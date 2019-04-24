# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class N2(Regularizer):
    def __init__(self, weight: float):
        super(N2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.norm(f, 2, 1) ** 3
            )
        return norm / factors[0].shape[0]


class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]
