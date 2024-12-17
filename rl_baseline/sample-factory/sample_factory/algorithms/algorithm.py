# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from abc import ABC
from os.path import join

from sample_factory.utils.utils import str2bool


class AlgorithmBase:
    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def add_cli_args(cls, parser):
        p = parser

    def initialize(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()


class ReinforcementLearningAlgorithm(AlgorithmBase, ABC):
    """Basic things that most RL algorithms share."""

    @classmethod
    def add_cli_args(cls, parser):
        p = parser
        super().add_cli_args(p)

    def __init__(self, cfg):
        super().__init__(cfg)
