# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import List, Optional, Callable
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import tqdm
import random
import nle
import gym
import copy
import hashlib
from collections import defaultdict
from nle import nethack
from rlaif.llms import AnnotationIdx
from rlaif.annotators_transforms import BlstatsTransform

from utils.preprocessing import ToTensorDict, GPT5BaselineTransform
from utils.decode import encode_message_batch, decode_message
from torchvision import transforms
from utils.fpdb import ForkedPdb


NETHACK_OBJ_CLASSES = {
    getattr(nethack, x): x for x in dir(nethack) if x.endswith("_CLASS")
}
NETHACK_GLYPHS_CLASSES = sorted(
    (getattr(nethack, x), x) for x in dir(nethack) if x.endswith("_OFF")
)


class InfiniteDataLoader:
    def __init__(self, dataloader_fn):
        self.dataloader_fn = dataloader_fn
        self.loader = iter(self.dataloader_fn())

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.loader)
        except StopIteration:
            self.loader = iter(self.dataloader_fn())
            return next(self.loader)


def dict_collate_fn(batch):
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        # stack along a new dimension
        if isinstance(batch[0][key], torch.Tensor):
            stacked_tensor = torch.stack([item[key] for item in batch])
        else:
            stacked_tensor = np.stack([item[key] for item in batch])
        collated_batch[key] = stacked_tensor
    return collated_batch


def tuple_dict_collate_fn(batch):
    use_tensor = isinstance(batch[0][1][list(batch[0][1].keys())[0]], torch.Tensor)
    collated_data = {}
    keys = batch[0][0].keys()
    for key in keys:
        # stack along a new dimension
        if use_tensor:
            stacked_tensor = torch.stack([item[0][key] for item in batch])
        else:
            stacked_tensor = np.stack([item[0][key] for item in batch])
        collated_data[key] = stacked_tensor
    collated_infos = {}
    keys = batch[0][1].keys()
    for key in keys:
        # stack along a new dimension
        if use_tensor:
            stacked_tensor = torch.stack([item[1][key] for item in batch])
        else:
            stacked_tensor = np.stack([item[1][key] for item in batch])
        collated_infos[key] = stacked_tensor
    return collated_data, collated_infos


def flatten_pair_collate_fn(batch, ignore_keys: List[str] = []):
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        # stack along a new dimension
        if key not in ignore_keys:
            # Batch size, 2, sequence length, ...
            stacked_tensor = torch.stack([item[key] for item in batch])
            collapsed_tensor = stacked_tensor.flatten(0, 2)
            collated_batch[key] = collapsed_tensor
        else:
            collated_batch[key] = [el[key] for el in batch]
            if isinstance(collated_batch[key][0], torch.Tensor):
                collated_batch[key] = torch.stack(collated_batch[key])
            else:
                collated_batch[key] = np.stack(collated_batch[key])
    return collated_batch


def pair_collate_fn(batch):
    # `batch` is a list of subepisode dictionaries
    # Group them into pairs
    pairs = [(batch[i], batch[i + 1]) for i in range(0, len(batch), 2)]

    # Turn each pair of dictionaries into a dictionary of tensors (batch_size, 2, subepisode_length, dims)
    collated_pairs = {
        key: np.array([(pair[0][0][key], pair[1][0][key]) for pair in pairs])
        for key in pairs[0][0][0]
    }

    collated_infos = {
        key: (
            np.array([((pair[0][1][key]), pair[1][1][key]) for pair in pairs])
            if type(pairs[0][0][1][key]) != str
            else [(pair[0][1][key], pair[1][1][key]) for pair in pairs]
        )
        for key in pairs[0][0][1]
    }
    return collated_pairs, collated_infos


class NetHackDirectoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        directories: List[str],
        subepisode_length: int,
        data_keys: Optional[List] = None,
        info_keys: Optional[List] = None,
        seed: int = 0,
    ):
        self.directories = directories
        self.subepisode_length = subepisode_length
        self.data_keys = data_keys
        self.info_keys = info_keys
        random.seed(seed)

        # Create a list of filenames and episode start indices for subepisodes
        self.subepisode_indices = []
        self.infos = {}
        for directory in self.directories:
            with open(os.path.join(directory, "info.json"), "r") as f:
                info = json.load(f)
                self.infos[directory] = info
                episode_lengths = list(zip(info["seed"], info["steps"]))

            for seed, length in episode_lengths:
                filename = f"{seed}.npz"
                num_subepisodes = length // self.subepisode_length
                # Calculate the range of valid starts
                valid_starts = range(length - self.subepisode_length + 1)
                # Sample a subset of valid starts
                subepisode_starts = random.sample(valid_starts, num_subepisodes)
                for start in subepisode_starts:
                    self.subepisode_indices.append((directory, filename, start))
        # Shuffle self.subepisode_indices
        self.subepisode_indices = random.sample(
            self.subepisode_indices, len(self.subepisode_indices)
        )
        new_infos = {}
        for directory in self.infos:
            new_infos[directory] = {}
            for i in range(len(self.infos[directory]["seed"])):
                # For each unique value, create a new dictionary with the corresponding values from the other arrays
                seed_val = str(self.infos[directory]["seed"][i])
                new_infos[directory][seed_val] = {
                    key: value[i]
                    for key, value in self.infos[directory].items()
                    if key != "seed"
                    if self.info_keys is None or key in self.info_keys
                }
        self.infos = new_infos

    def __len__(self):
        return len(self.subepisode_indices)

    def __getitem__(self, idx):
        directory, filename, start_idx = self.subepisode_indices[idx]
        full_name = os.path.join(directory, filename)
        with np.load(full_name) as episode:
            # Extract the subepisode
            subepisode = {
                key: values[start_idx : start_idx + self.subepisode_length]
                for key, values in episode.items()
                if self.data_keys is None or key in self.data_keys
            }
            info = self.infos[directory][filename.split(".")[0]]
            # Add path to episode file to info
            info["path"] = full_name
            return subepisode, info


class PairsDataset(torch.utils.data.Dataset):
    """Dataset for pairs of subepisodes.

    Args:
        directory (str): Path to the directory containing the subepisodes.
        data_keys (Optional[List[str]]): List of keys to extract from the subepisodes.
        info_keys (Optional[List[str]]): List of keys to extract from the info.
        preference_keys (Optional[List[str]]): List of keys to extract from the preferences.
        transform (Optional[Callable]): Transform to apply to the data.
        mode (str): Mode to open the subepisode files in. See `numpy.load` for details.
    """

    def __init__(
        self,
        directory: str,
        data_keys: Optional[List[str]] = None,
        info_keys: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        preference_keys: Optional[List[str]] = None,
        mode: str = "r",
    ):
        assert (
            data_keys is not None
            or info_keys is not None
            or preference_keys is not None
        )
        self.directory = directory
        self.data_keys = data_keys
        self.info_keys = info_keys
        self.preference_keys = preference_keys
        # Use directory structure to load arrays with the right keys
        if self.data_keys is not None:
            self.data_arrays = self.load_to_dict(
                self.directory, "data", mode, data_keys
            )
            self.valid_indices = np.arange(
                len(self.data_arrays[list(self.data_arrays.keys())[0]])
            )
        if self.info_keys is not None:
            self.info_arrays = self.load_to_dict(
                self.directory, "info", mode, info_keys
            )
            self.valid_indices = np.arange(
                len(self.info_arrays[list(self.info_arrays.keys())[0]])
            )
        final_mask = np.ones(
            self.data_arrays[list(self.data_arrays.keys())[0]].shape[0], dtype=bool
        )
        if self.preference_keys is not None:
            self.pref_arrays = self.load_to_dict(
                self.directory, "preference", mode, preference_keys
            )
            mask_arrays = {
                key: self.pref_arrays[key] != AnnotationIdx.UNKOWN
                for key in self.pref_arrays
            }
            for _, arr in mask_arrays.items():
                final_mask = np.logical_and(final_mask, arr)
        self.valid_indices = np.where(final_mask)[0]

        self.transform = transform

    def load_to_dict(
        self,
        dir_name: str,
        subdir_name: str,
        mode: str = "r",
        keys: Optional[List[str]] = None,
    ):
        running_keys = set(keys)
        array_dict = {}
        for filename in os.listdir(os.path.join(self.directory, subdir_name)):
            filename_key = filename.replace(".npy", "")
            if keys is None or filename_key in keys:
                array = np.load(
                    os.path.join(dir_name, subdir_name, filename), mmap_mode=mode
                )
                array_dict[filename_key] = array
                running_keys.remove(filename_key)
        if len(running_keys) > 0:
            raise ValueError(
                f"Keys {running_keys} not found in {os.path.join(dir_name,subdir_name)}"
            )
        return array_dict

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        data = {"idx": idx}
        if self.data_keys is not None:
            for key in self.data_arrays:
                data[key] = self.data_arrays[key][self.valid_indices[idx]]
        if self.info_keys is not None:
            for key in self.info_arrays:
                data[key] = self.info_arrays[key][self.valid_indices[idx]]
        if self.preference_keys is not None:
            for key in self.pref_arrays:
                data[key + "_pref"] = self.pref_arrays[key][self.valid_indices[idx]]

        if self.transform:
            data = self.transform(data)

        return data


class OnlinePairsDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        remove_duplicated_obs=True,
        device="cpu",
    ):
        self.remove_duplicated_obs = remove_duplicated_obs
        self.device = "cpu"

        self.labeled_pairs = []
        self.observations = []
        self.sample_hashes = set()
        self.obs_cpu = None
        self.sample = None

    def __len__(self):
        return len(self.labeled_pairs)

    def num_samples(self):
        return len(self.observations)

    def add_obs_batch(self, obs, subsample=1):

        bsz = obs["message"].shape[0]
        indx = torch.arange(0, bsz, subsample)

        obs_cpu = {k: v[indx].cpu() for k, v in obs.items() if k in ["message"]}
        obs_cpu["message"] = obs_cpu["message"].int().numpy()

        for i in range(0, len(indx)):
            sample = {k: v[i] for k, v in obs_cpu.items()}
            if self.remove_duplicated_obs:
                # use hashing to avoid duplicate observations
                sample_hash = self.sample_to_str(sample)
                if sample_hash not in self.sample_hashes:
                    self.observations.append(sample)
                    self.sample_hashes.add(sample_hash)
            else:
                self.observations.append(sample)

    def sample_to_str(self, sample):
        msg = decode_message(sample["message"])
        return msg

    def _sample_indices_uniformly(self, num_samples):
        indices = [
            random.randint(0, len(self.observations) - 1) for i in range(num_samples)
        ]
        return indices

    def sample_pairs_for_labeling(self, max_trials=100, device="cpu", num_pairs=1):

        self.device = device
        idx = self._sample_indices_uniformly(2 * num_pairs)
        pairs = [(idx[i], idx[i + 1]) for i in range(0, len(idx), 2)]

        msg_pairs = []
        for i1, i2 in pairs:
            llm_msg_1 = self.sample_to_str(self.observations[i1])
            llm_msg_2 = self.sample_to_str(self.observations[i2])
            msg_pairs.append(((i1, llm_msg_1), (i2, llm_msg_2)))

        return msg_pairs

    def update(self, new_labeled_pairs_dict, placeholder=None):
        for pair, label in new_labeled_pairs_dict.items():
            i1 = pair[0][0]
            i2 = pair[1][0]
            self.labeled_pairs.append({"pair": (i1, i2), "label": label})

    def __getitem__(self, idx: int):
        sample_1 = self.observations[self.labeled_pairs[idx]["pair"][0]]
        sample_2 = self.observations[self.labeled_pairs[idx]["pair"][1]]
        item = {}
        item["idx"] = idx
        item["label"] = torch.tensor(self.labeled_pairs[idx]["label"], dtype=torch.int8)
        item["message"] = torch.stack(
            (torch.tensor(sample_1["message"]), torch.tensor(sample_2["message"]))
        ).view(2, 1, -1)

        return item


if __name__ == "__main__":
    from torch.utils.data import random_split

    info_keys = ["score"]
    print("Loading dataset...")
    if False:
        dataset = PairsDataset(
            "./motif_dataset",
            data_keys=["blstats", "message", "tty_chars", "tty_colors"],
            info_keys=info_keys,
            preference_keys=["llama70b_msg_defaultgoal_default"],
            transform=transforms.Compose([GPT5BaselineTransform(), ToTensorDict()]),
            mode="r",
        )

    elif True:
        dataset = OnlinePairsDataset(data_keys=["message", "blstats"])
        env = gym.make("NetHackScore-v0")
        obs = env.reset()
        for i in range(100):
            obs, _, _, _ = env.step(env.action_space.sample())
            dataset.add_obs(obs)
        print(dataset.sample_pair_for_labeling())
        print(dataset.sample_pair_for_labeling())
        print(dataset.sample_pair_for_labeling())
        print(dataset.sample_pair_for_labeling())
    else:
        dataset = MessagePairsDataset()
        labeled_pairs_dict = {
            ("good message 1", "bad message 1"): 0,
            ("good message 2", "bad message 2"): 0,
            ("bad message 3", "good message 3"): 1,
            ("bad message 4", "good message 4"): 1,
            ("bad message 5", "good message 5"): 1,
            ("bad message 6", "good message 6"): 1,
        }
        dataset.update(labeled_pairs_dict)

    print("Dataset length", len(dataset))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=dict_collate_fn,
        num_workers=64,
    )
    start_time = time.time()
    for batch in tqdm.tqdm(data_loader):
        pass
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Time taken for one epoch: {epoch_time} seconds")
