#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    resnet = models.resnet50(pretrained=True)
    self.conv1 = resnet.conv1
    self.bn1 = resnet.bn1
    self.relu = resnet.relu 
    self.maxpool = resnet.maxpool

    self.res2 = resnet.layer1 
    self.res3 = resnet.layer2 
    self.res4 = resnet.layer3 

    self.bottleneck_key = nn.Conv2d(1024, 128, kernel_size=1, stride=1, bias=False)
    self.bottleneck_value = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)

    for m in self.modules():
      if isinstance(m, nn.BatchNorm2d):
        for p in m.parameters():
          p.requires_grad = False

  def forward(self, in_f):
    x = self.conv1(in_f)
    x = self.bn1(x)
    c1 = self.relu(x) 
    x = self.maxpool(c1)  
    r2 = self.res2(x)  
    r3 = self.res3(r2) 
    r4 = self.res4(r3) 

    key = self.bottleneck_key(r4)
    value = self.bottleneck_value(r4)

    return key, value
