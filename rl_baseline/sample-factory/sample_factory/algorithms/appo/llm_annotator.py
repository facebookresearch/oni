# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import shutil
import tqdm
import numpy as np
from numpy.lib.format import open_memmap

import torch
from torch.multiprocessing import Process as TorchProcess

import re
import dataclasses
import random
import pickle

from abc import ABC, abstractmethod
import itertools
from typing import Dict, List, Callable, Optional, Tuple, Sequence

import numpy as np
import torchvision

from sample_factory.utils.utils import log

from fastchat.model.model_adapter import (
    get_conversation_template as get_conversation_template_fastchat,
)
from fastchat.conversation import Conversation, SeparatorStyle
from vllm import LLM, SamplingParams
import openai
import time

from faster_fifo import Queue as MpQueue
from utils.fpdb import ForkedPdb

from rlaif.prompts import (
    system_prompts,
    ranking_regexes,
    generate_prompt_ranking,
    generate_prompt_classification,
)

from rlaif.llms import AnnotationIdx

# assert False, "manually merge"


def get_cls_label(response):
    # TODO: generalize and parameterize
    helpful_match = re.search("FOO", response)
    unhelpful_match = re.search("BAR", response)

    if helpful_match:
        return 1
    elif unhelpful_match:
        return 0
    else:
        # TODO: could log, save this better to debug the prompt
        log.warning(f"WARNING: no annotation found in response:\n\n{response}\n")
        return 0


# LLaMA-3 is not registered in fastchat, so handle this here
def get_conversation_template(model_name):
    if "llama-3" in model_name.lower():
        # taken from: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
        return Conversation(
            name="llama-3",
            system_template="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_message}",
            roles=(
                "<|start_header_id|>user<|end_header_id|>",
                "<|start_header_id|>assistant<|end_header_id|>",
            ),
            sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
            sep="<|eot_id|>",
            stop_str="<|eot_id|>",
        )
    else:
        return get_conversation_template_fastchat(model_name)


class LLMAnnotatorOpenAI:

    def __init__(
        self,
        model="Meta-Llama-3.1-8B-Instruct",
        server_port="9001",
        prompt_version="default",
        goal_key="defaultgoal",
        log_dir=None,
    ):
        self.model = model
        self.server_port = server_port
        self.prompt_version = prompt_version
        self.goal_key = goal_key
        self.log_dir = log_dir

    def setup_server(self, server_addr):
        self.llm_server = openai.OpenAI(
            base_url=f"http://{server_addr}:{self.server_port}/v1",
            api_key="EMPTY",
        )

    def __call__(self, message_batch):
        # TODO: could log more out for debugging
        responses = self._generate_llm_response(message_batch)
        annotations_batch = {
            message: self._get_label(response)
            for message, response in zip(message_batch, responses)
        }
        return annotations_batch

    def _generate_llm_response(self, message_batch):
        prompts = []
        conv = get_conversation_template(self.model)
        for message in message_batch:
            conv.set_system_message(system_prompts["default"])
            conv.messages = []
            # TODO: rename key of the online motif method in prompt templates (default vs online_motif)
            if self.reward_type in ["online_cls", "online_cls_reward_model"]:
                conv_message = generate_prompt_classification(message, self.goal_key)
            elif self.reward_type == "online_motif":
                conv_message = generate_prompt_ranking(
                    message[0][1], message[1][1], self.goal_key, self.prompt_version
                )
            conv.append_message(conv.roles[0], conv_message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)

        completions = self.llm_server.completions.create(
            model=self.model,
            prompt=prompts,
            temperature=0.1,
            # top_p=0.95, # don't use it together with temperature as recommended by OpenAI API
            max_tokens=4096,
            # stop=["HELPFUL"],
        )
        responses = []
        for choice in completions.choices:
            text = choice.text
            # if choice.finish_reason == "stop":
            #     text += "HELPFUL"
            responses.append(text)

        if self.log_dir:
            for prompt, response in zip(prompts, responses):
                if random.random() < 0.1:
                    fname = f"{self.log_dir}/{self.log_cnt:08d}.txt"
                    full_conv = prompt + "\n" + response
                    with open(fname, "w") as f:
                        f.write(full_conv)
                    self.log_cnt += 1

        return responses

    def _get_label(self, response):
        if self.reward_type in ["online_cls", "online_cls_reward_model"]:
            label = self._get_cls_label(response)
        elif self.reward_type in ["online_motif"]:
            result = re.search(ranking_regexes["default"], response)
            if result:
                try:
                    best_sequence = int(result.group(1))
                    if best_sequence == 1:
                        best_sequence = AnnotationIdx.FIRST
                    elif best_sequence == 2:
                        best_sequence = AnnotationIdx.SECOND
                except Exception as e:
                    # log.debug(f"Exception {e} occured when parsing response: \n{response}\n treating as a TIE")
                    best_sequence = AnnotationIdx.TIE
            else:
                best_sequence = AnnotationIdx.TIE
            label = best_sequence

        return label

    def _get_cls_label(self, response):
        # TODO: get rid of this function (_get_cls_label)
        return get_cls_label(response)


@dataclasses.dataclass
class LLMWorker:
    model: str = "Meta-Llama-3.1-8B-Instruct"
    reward_type: str = "online_cls_reward_model"
    llm_batch_size: int = 500
    log_dir: str = ""
    message_queue = MpQueue(10 * 1000 * 1000)
    reward_queue = MpQueue(2 * 1000 * 1000)
    process = None
    prompt_version: str = "default"
    goal_key: str = "defaultgoal"

    def start_process(self, server_addr):
        # annotator needs to be created here to correctly read self.model
        self.annotator = LLMAnnotatorOpenAI(
            self.model, prompt_version=self.prompt_version, goal_key=self.goal_key
        )
        self.annotator.reward_type = self.reward_type
        if self.log_dir:
            self.annotator.log_dir = f"{self.log_dir}/llm_logs"
            os.makedirs(self.annotator.log_dir, exist_ok=True)
            # TODO: could move init back into the class
            self.annotator.log_cnt = 0
        self.annotator.setup_server(server_addr)
        self.process = TorchProcess(target=self._run, daemon=True)
        self.process.start()

    def _run(self):
        messages_to_process = []
        while True:
            log.info(f"llmworker; queue size: {self.message_queue.qsize()}")
            if self.message_queue.qsize() > 0:
                # get most new messages in the queue
                # don't go all the way to 0 since there are so many actors writing into it
                min_messages_queue = 3  # MH: this does not work with 100
                while self.message_queue.qsize() > min_messages_queue:
                    new_messages = self.message_queue.get_many()
                    messages_to_process += new_messages

                # annotate a batch of newest messages
                log.info(
                    f"llmworker: {len(messages_to_process)} messages to process, running the latest batch"
                )
                message_batch = messages_to_process[-self.llm_batch_size :]
                messages_to_process = messages_to_process[: -self.llm_batch_size]
                if (
                    self.reward_type == "online_motif"
                    and len(messages_to_process) > 10000
                ):
                    messages_to_process = messages_to_process[-10000:]
                # if len(message_batch) > 0: # (TODO: check for online_motif?)
                annotations_batch = self.annotator(message_batch)
                self.reward_queue.put(annotations_batch)
            else:
                log.info("llmworker sleep")
                time.sleep(5)


if __name__ == "__main__":
    messages = [
        "you are hit by a huge chunk of metal",
        "you dishonorably attack the innocent!",
        "you find gold on the ground",
    ]
    annotator = LLMAnnotatorOpenAI()
    annotator.setup_server("localhost")
    annotations = annotator(messages)
    print(annotations)
