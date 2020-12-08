#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

from .FullyConnected import FullyConnected
from .Encoder import Encoder
from .NonLocalBlock import NonLocalBlock
from .ResNetNonLocal import CNN

class MemoryEncoder(nn.Module):
  def __init__(self):
    super(MemoryEncoder, self).__init__()

    self.memory_encoder = CNN(2048, model_type='resnet50_NL',
                              non_layers=[0,2,3,0],
                              stripes=[], temporal='NoPool')

  def forward(self, period):
    key, value = self.memory_encoder(period)

    return key, value

class MemoryNetwork(nn.Module):
  def __init__(self, input_shape, feat_dim=1024, out_size=2):
    super(MemoryNetwork, self).__init__()
    # input_shape must be (C, H, W); without batch dimension
    self.out_size = out_size

    self.query_encoder = Encoder()

    self.memory_reader = NonLocalBlock()

    # Get classifier input size
    tmp = torch.rand(1, *input_shape)
    tmp_key, tmp_value = self.query_encoder(tmp)
    tmp_key = tmp_key.unsqueeze(dim=0)
    tmp_value = tmp_value.unsqueeze(dim=0)
    tmp = self.memory_reader(tmp_key, tmp_value, tmp_key, tmp_value)
    classifier_input_size = tmp.view(1,-1).size()[1]
    print('MemoryNetwork: Memory reader output size: {} ({} if flattened)'
          ''.format(tmp.size(), tmp.view(1,-1).size() ))
    
    self.feature_layer = FullyConnected(classifier_input_size, feat_dim)

    self.classifier = FullyConnected(feat_dim, self.out_size) 

  def forward(self, query, memory_key, memory_value):
    q_batch = query.size(0) # query_batch or Bq
    q_len = query.size(1) # Tq
    m_batch = memory_key.size(0) # memory_batch of Bm

    query = query.view((-1,) + query.size()[2:]) # (batch*Tq x C x H x W)
    query_key, query_value = self.query_encoder(query) 
    query_key = query_key.view((q_batch, q_len) + query_key.size()[1:])
    query_value = query_value.view((q_batch, q_len) + query_value.size()[1:])

    attention = self.memory_reader(
      query_key, 
      query_value, 
      memory_key, # (Bm x Tm x C/8 x H x W)
      memory_value # (Bm x Tm x C/2 x H x W)
      ) # Bq x Bm x Tq x C x H x W
    attention = attention.view(q_batch, m_batch, q_len, -1) # Bq x Bm x Tq x CHW
    feature = self.feature_layer(attention) # Bq x Bm x Tq x feat_dim
    feature = F.relu(feature)
    logit = self.classifier(feature) # Bq x Bm x Tq x classifier_out_size

    return logit, feature

