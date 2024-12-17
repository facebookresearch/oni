# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import fasttext
import numpy as np
from torch import nn

import torch.nn.functional as F

# from sentence_transformers import SentenceTransformer
from torch.distributions import Categorical


from sample_factory.algorithms.appo.model_utils import create_encoder, create_core
from sample_factory.algorithms.appo.model import _ActorCriticSharedWeights
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import AttrDict

from utils.fpdb import ForkedPdb


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, device, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).to(device)
        self.var = torch.ones(shape).to(device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardSharedWeights(_ActorCriticSharedWeights):
    def __init__(self, make_encoder, make_core, seq_len, action_space, cfg, timing):
        super().__init__(make_encoder, make_core, action_space, cfg, timing)
        self.seq_len = seq_len

        self.core_output_size = self.encoder.encoder_out_size
        self.reward_fn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.core_output_size, 1),
        )

        self.device = torch.device("cpu" if cfg.device == "cpu" else "cuda")

        self.apply(self.initialize_weights)
        self.train()

    def add_mean_var(self, mean, var):
        self.mean = mean
        self.var = var

    def forward(self, mb, normalize=False, add_dim=False):
        for key, value in mb.items():
            if key in ["obs", "message", "glyphs", self.encoder.bl_stats_key]:
                if add_dim:
                    mb[key] = torch.tensor(value[None, ...]).to("cuda")
                else:
                    mb[key] = value.to("cuda")

        x = self.forward_head(mb, normalize=normalize)
        rewards = self.reward_fn(x)

        result = AttrDict(
            dict(
                rewards=rewards,
            )
        )
        return result

    def forward_pairs(self, mb, normalize=True):
        for key, value in mb.items():
            if key in ["obs", "message", "glyphs", self.encoder.bl_stats_key]:
                mb[key] = value.to("cuda")

        batch_size = len(
            mb["message"].reshape(-1, 2, self.seq_len, mb["message"].shape[-1])
        )
        x = self.forward_head(mb, normalize=normalize)
        x = x.reshape(
            batch_size * 2, self.seq_len, -1
        )  # Batch size x 2, sequence length, *
        x = x.transpose(0, 1)  # sequence length, Batch size x 2, *
        rewards = self.reward_fn(x)
        rewards = rewards.reshape(
            self.seq_len, batch_size, 2
        )  # sequence length, Batch size, 2

        result = AttrDict(
            dict(
                rewards=rewards,
            )
        )
        return result


def create_reward_model(cfg, obs_space, action_space, seq_len=1, timing=None):
    """Motif Reward Model"""
    if timing is None:
        timing = Timing()

    def make_encoder():
        return create_encoder(cfg, obs_space, timing, cfg.reward_encoder)

    def make_core(encoder):
        return create_core(cfg, encoder.get_encoder_out_size(), False)

    if cfg.actor_critic_share_weights:
        return RewardSharedWeights(
            make_encoder, make_core, seq_len, action_space, cfg, timing
        )
    else:
        raise NotImplementedError


# ======================
#      ONI Code
# ======================


class FastTextBoWEmbedding:

    # TODO: pass model path from config
    def __init__(
        self,
        goal_string,
        threshold=None,
        model_path="./fasttext/crawl-300d-2M-subword.bin",
    ):
        self.model = fasttext.load_model(model_path)
        self.threshold = threshold
        self.dim = self.model.get_dimension()
        self.vec_cache = torch.zeros(self.dim)
        self.goal_emb = self.embed(goal_string)
        self.rewards_dict = {}

    def embed(self, text):
        tokens = fasttext.tokenize(text)
        self.vec_cache.zero_()
        for tok in tokens:
            self.vec_cache.add_(torch.from_numpy(self.model[tok]))
        return self.vec_cache.clone()

    def compute_reward(self, message):
        if message in self.rewards_dict:
            return self.rewards_dict[message]
        else:
            message_emb = self.embed(message)
            sim = F.cosine_similarity(message_emb, self.goal_emb, dim=0).item()
            if self.threshold is not None:
                sim = sim if sim > self.threshold else 0.0
            self.rewards_dict[message] = sim
            return sim


def mlp(input_dim, output_dim, hidden_dims, dropout=0.0):

    layers = []
    for dim in hidden_dims:
        layers += [
            nn.Linear(input_dim, dim),
            nn.ReLU(inplace=True),
        ]
        input_dim = dim

    if dropout:
        layers += [nn.Dropout(dropout)]
    layers += [nn.Linear(input_dim, output_dim)]

    return layers


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ClassificationRewardModel(nn.Module):
    def __init__(
        self, n_cls, hidden_dims, classification_threshold, dropout=0.0, device="cuda"
    ):
        super().__init__()

        NUM_CHARS = 256  # TODO: move
        PAD_CHAR = 0  # TODO: move

        # We train the reward model using the cross entropy loss. For binary
        # classification, this is equivalent to logistic regression and the
        # corresponding probability decision boundary is 0.5 (which corresponds to the
        # decision boundary <w,x>+b = 0). However, at inference time, we want to control
        # the sparsity of intrinsic rewards, therefore, we would like to vary the
        # boundary.
        self.classification_threshold = classification_threshold
        self.softmax_func = torch.nn.Softmax(dim=1)
        self.msg_hdim = 64
        self.msg_edim = 32
        self.char_lt = nn.Embedding(NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR)
        self.conv1 = nn.Conv1d(self.msg_edim, self.msg_hdim, kernel_size=7)
        # remaining convolutions, relus, pools, and a small FC network
        self.conv2_6_fc = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            # conv2
            nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            # conv3
            nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
            nn.ReLU(),
            # conv4
            nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
            nn.ReLU(),
            # conv5
            nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
            nn.ReLU(),
            # conv6
            nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            # fc receives -- [ B x h_dim x 5 ]
            Flatten(),
            nn.Linear(5 * self.msg_hdim, 2 * self.msg_hdim),
            nn.ReLU(),
        )
        layers = mlp(2 * self.msg_hdim, n_cls, hidden_dims, dropout)
        self.net = nn.Sequential(*layers)
        self.device = device

    def forward(self, messages_raw):
        h = self.char_lt(messages_raw).transpose(1, 2)
        h = self.conv1(h)
        h = self.conv2_6_fc(h)
        logits = self.net(h)
        return logits

    def make_prediction(self, messages_raw):
        logits = self.forward(messages_raw)
        probs = self.softmax_func(logits)
        return (probs[:, 1] > self.classification_threshold).float()


class RewardModelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        llm_message_rewards,
    ):

        super(RewardModelDataset, self).__init__()
        self.data = list(llm_message_rewards.items())

    def update(self, new_msg_reward_dict, old_dict):

        for message, reward in new_msg_reward_dict.items():
            if message not in old_dict:
                self.data.append((message, reward))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_labels_only(self):
        vals = []
        for i in range(len(self.data)):
            y = self.data[i][1]
            vals.append(y)
        return vals
