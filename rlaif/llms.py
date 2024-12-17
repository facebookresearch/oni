# Copyright (c) Meta Platforms, Inc. and affiliates.

import re
from fastchat.model.model_adapter import (
    get_conversation_template as get_conversation_template_fastchat,
)
from fastchat.conversation import Conversation, SeparatorStyle
from vllm import LLM, SamplingParams
import os
from typing import List, Optional, Sequence
import numpy as np


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


class AnnotationIdx:
    FIRST = 0
    SECOND = 1
    TIE = 2
    UNKOWN = 3


class LocalLanguageModel:
    def __init__(
        self,
        system_prompt: str,
        answer_regex: str,
        retry_prompt: str,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        num_gpus: int = 8,
        logdir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.answer_regex = answer_regex
        self.retry_prompt = retry_prompt
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=num_gpus,
            dtype="float16",
            max_num_batched_tokens=8192,
            max_model_len=8192,
            # if we don't set this to false, crashes with llama3.1
            # can use bigger batch sizes if this is set to true.
            enable_chunked_prefill=False,
        )
        self.logdir = logdir
        self.system_prompt = system_prompt
        if self.logdir is not None:
            # Create directory
            os.makedirs(self.logdir, exist_ok=True)

    def generate(
        self, messages: List[str], logging_indices: Sequence[int] = None
    ) -> List[int]:
        assert len(messages) == len(logging_indices)
        prompts = []
        convs = []
        for message in messages:
            conv = get_conversation_template(self.model_name)
            conv.set_system_message(self.system_prompt)
            conv.messages = []
            conv.append_message(conv.roles[0], message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
            convs.append(conv)
        sampling_params = SamplingParams(
            top_k=50, max_tokens=4096, temperature=0.1, top_p=0.95, stop=conv.stop_str
        )
        outputs = self.llm.generate(prompts, sampling_params)
        # Parse all the outputs
        cleaned_outputs = np.full(len(messages), AnnotationIdx.UNKOWN)
        indexes_to_retry = []
        prompts_to_retry = []
        print("Retrying prompts")
        for i, output in enumerate(outputs):
            text_answer = output.outputs[0].text
            result = re.search(self.answer_regex, text_answer)
            conv = convs[i]
            conv.append_message("", text_answer)
            if result:
                try:
                    best_sequence = int(result.group(1))
                    if best_sequence == 1:
                        best_sequence = AnnotationIdx.FIRST
                    elif best_sequence == 2:
                        best_sequence = AnnotationIdx.SECOND
                except ValueError:
                    best_sequence = AnnotationIdx.TIE
                cleaned_outputs[i] = best_sequence
            else:
                # Ask the model again
                conv.append_message(conv.roles[0], self.retry_prompt)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                prompts_to_retry.append(prompt)
                indexes_to_retry.append(i)
        # Retry the prompts that were not good
        second_batch = self.llm.generate(prompts_to_retry, sampling_params)
        for i, output in zip(indexes_to_retry, second_batch):
            text_answer = output.outputs[0].text
            convs[i].append_message("", text_answer)
            result = re.search(self.answer_regex, text_answer)
            if result:
                try:
                    best_sequence = int(result.group(1))
                    if best_sequence == 1:
                        best_sequence = AnnotationIdx.FIRST
                    elif best_sequence == 2:
                        best_sequence = AnnotationIdx.SECOND
                except ValueError:
                    best_sequence = AnnotationIdx.TIE
                cleaned_outputs[i] = best_sequence
        if self.logdir is not None and logging_indices is not None:
            # Log the conversations
            for conv, idx in zip(convs, logging_indices):
                text_conv = conv.get_prompt()
                with open(os.path.join(self.logdir, str(idx) + ".txt"), "w") as f:
                    f.write(text_conv)
        return cleaned_outputs


class MockLanguageModel:
    def __init__(
        self,
        system_prompt: str,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",  # 'vicuna-13b-v1.3',
        num_gpus: int = 8,
        logdir: Optional[str] = None,
    ) -> None:
        pass

    def generate(
        self, messages: List[str], logging_indices: Optional[List[int]]
    ) -> List[int]:
        return np.random.randint(0, 2, len(messages))
