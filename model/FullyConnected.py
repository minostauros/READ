#!/usr/bin/env python3
import torch
import torch.nn as nn

class FullyConnected(nn.Module):
  def __init__(self, in_size, out_size):
    super(FullyConnected, self).__init__()

    self.fc = nn.Linear(in_size, out_size)

  def forward(self, x):
    x = self.fc(x)

    return x