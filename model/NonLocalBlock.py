#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlock(nn.Module):
  def __init__(self, fc_dim=None):
    super(NonLocalBlock, self).__init__()
    pass

  def forward(self, query_key, query_value, memory_key, memory_value):
    """
      This operation support batch of queries with memories
    Input size should be
      query: (query_batch_size[Bq] x Tq x C/2 or C/8 x H x W based on value/key)
      memory: (num_memories[Bm] x Tm x C/2 or C/8 x H x W)
    
    Return: 
      result: (Bq, Bm, Tq, C, H, W)
    """
    
    query_batch_size = query_key.size(0)
    tq_size = query_key.size(1)
    memory_batch_size = memory_key.size(0)
    key_size = query_key.size(2)
    value_size = query_value.size(2)

    query_key = query_key.view(query_batch_size, tq_size, key_size, -1) # Bq x Tq x C/8 x HW
    query_key = torch.transpose(query_key, 2, 3) # Bq x Tq x HW x C/8
    query_key = query_key.unsqueeze(dim=1) # Bq x 1 x Tq x HW x C/8
    memory_key = memory_key.view(memory_batch_size, 1, key_size, -1) # Bm x 1 x C/8 x TmHW
    
    key_attention = torch.matmul(query_key, memory_key) # Bq x Bm x Tq x HW x TmHW
    key_attention = F.softmax(key_attention, dim=-1) # softmax over memory space-time location

    H = query_value.size(3)
    W = query_value.size(4)

    memory_value = memory_value.view(memory_batch_size, 1, value_size, -1) # Bm x 1 x C/2 x TmHW
    memory_value = torch.transpose(memory_value, 2, 3) # Bm x 1 x TmHW x C/2

    attention = torch.matmul(key_attention, memory_value) # Bq x Bm x Tq x HW x C/2
    attention = torch.transpose(attention, 3, 4) # Bq x Bm x Tq x C/2 x HW
    attention = attention.view(
      query_batch_size, memory_batch_size, tq_size, value_size, H, W
      ) # Bq x Bm x Tq x C/2 x H x W

    query_value_rep = query_value.unsqueeze(dim=1) # Bq x 1 x Tq x C/2 x H x W
    query_value_rep = query_value_rep.expand(
      -1, memory_batch_size, -1, -1, -1, -1
      ) # Bq x Bm x Tq x C/2 x H x W

    result = torch.cat([query_value_rep, attention], dim=3) # Bq x Bm x Tq x C x H x W

    return result
