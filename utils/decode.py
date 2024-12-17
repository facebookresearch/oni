# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np


def decode_message(message):
    """Decode NetHack's message from the environment into a string."""
    return bytes(message[message != 0].tolist()).decode("latin-1")


def encode_message_no_pad(message):
    return np.array(list(message.encode("latin-1")), dtype=np.int64)


def encode_message_batch(messages, message_length=256):
    msg = np.zeros((len(messages), message_length), dtype=np.int64)
    for i, m in enumerate(messages):
        msg[i, : len(m)] = encode_message_no_pad(m)
    return msg
