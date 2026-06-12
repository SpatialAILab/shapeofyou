import torch
from torch import nn


class DummyAggregationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones([]))

    def forward(self, batch, pose=None):
        return batch * self.dummy
