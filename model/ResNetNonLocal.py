#!/usr/bin/env python3
"""
Non-Local ResNet-50 modified from 
  https://github.com/jackie840129/STE-NVAN/blob/master/net/models.py
READ can achieve similar results without non-local resnet.
See supplementary material for more details.
"""
import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CNN(nn.Module):
  def __init__(self,out_dim,model_type='resnet50_s1',num_class=710,non_layers=[1,2,2],stripes=[16,16,16,16], temporal = 'Done',stride=1):
    super(CNN,self).__init__()
    self.model_type = model_type
    if model_type == 'resnet50_s1':
      self.features = Resnet50_s1(stride=stride)
    elif model_type == 'resnet50_NL':
      self.features = Resnet50_NL(non_layers=non_layers,temporal=temporal,non_type='normal')
    elif model_type == 'resnet50_NL_stripe':
      self.features = Resnet50_NL(non_layers=non_layers,stripes=stripes,temporal=temporal,non_type='stripe')
    elif model_type == 'resnet50_NL_hr':
      self.features = Resnet50_NL(non_layers=non_layers,stripes=stripes,temporal=temporal,non_type='hr')
    elif model_type == 'resnet50_NL_stripe_hr':
      self.features = Resnet50_NL(non_layers=non_layers,stripes=stripes,temporal=temporal,non_type='stripe_hr')

    self.temporal = temporal

    if self.temporal == 'NoPool':
      # Bottleneck layers: keep spatial size, reduce channel size
      self.bottleneck_key = nn.Conv2d(1024, 128, kernel_size=1, stride=1, bias=False)
      self.bottleneck_value = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
    else:
      self.bottleneck = nn.BatchNorm1d(out_dim)
      self.bottleneck.bias.requires_grad_(False)  # no shift
      self.bottleneck.apply(weights_init_kaiming)    

      # self.classifier = nn.Linear(out_dim,num_class, bias=False)
      # self.classifier.apply(weights_init_classifier)
      self.id_classifier = nn.Linear(out_dim,num_class, bias=False)
      self.id_classifier.apply(weights_init_classifier)

  def forward(self,x,seg=None):
    if self.model_type == 'resnet50_s1':
      x = self.features(x)
      bn = self.bottleneck(x)
      if self.training == True:
        output = self.id_classifier(bn)
        return x,output
      else:
        return bn
    elif self.model_type == 'resnet50_NL' or self.model_type == 'resnet50_NL_stripe' or \
      self.model_type=='resnet50_NL_hr' or self.model_type == 'resnet50_NL_stripe_hr':
      if self.temporal == 'NoPool':
        x = self.features(x) # (B,C,T,H,W)
        x = x.permute(0,2,1,3,4) # (B,T,C,H,W)
        B,T,C,H,W = x.shape
        x = x.reshape(B*T,C,H,W)

        key = self.bottleneck_key(x)
        value = self.bottleneck_value(x)
        
        C_key = key.size(1)
        C_value = value.size(1)
        
        key = key.reshape(B,T,C_key,H,W)
        value = value.reshape(B,T,C_value,H,W)
        
        return key, value

      else:
        x = self.features(x)
        bn = self.bottleneck(x)
        if self.training == True:
          output = self.id_classifier(bn)
          return x,output
        else:
          return bn

def weights_init_kaiming(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
    nn.init.constant_(m.bias, 0.0)
  elif classname.find('Conv') != -1:
    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    if m.bias is not None:
      nn.init.constant_(m.bias, 0.0)
  elif classname.find('BatchNorm') != -1:
    if m.affine:
      nn.init.constant_(m.weight, 1.0)
      nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    nn.init.normal_(m.weight, std=0.001)
    if m.bias:
      nn.init.constant_(m.bias, 0.0)


class Resnet50_NL(nn.Module):
  def __init__(self,non_layers=[0,1,1,1],stripes=[16,16,16,16],non_type='normal',temporal=None):
    super(Resnet50_NL,self).__init__()
    original = models.resnet50(pretrained=True).state_dict()
    if non_type == 'normal':
      self.backbone = ResNet_Video_nonlocal(last_stride=1,non_layers=non_layers)
    elif non_type == 'stripe':
      self.backbone = ResNet_Video_nonlocal_stripe(last_stride = 1, non_layers=non_layers, stripes=stripes)
    elif non_type == 'hr':
      self.backbone = ResNet_Video_nonlocal_hr(last_stride = 1, non_layers=non_layers, stripes=stripes)
    elif non_type == 'stripe_hr':
      self.backbone = ResNet_Video_nonlocal_stripe_hr(last_stride = 1, non_layers=non_layers, stripes=stripes)
    for key in original:
      if key.find('fc') != -1:
        continue
      elif key.find('layer4') != -1:
        continue
      self.backbone.state_dict()[key].copy_(original[key])
    del original

    self.temporal = temporal
    if self.temporal == 'Done':
      self.avgpool = nn.AdaptiveAvgPool3d(1)


  def forward(self,x):
    if self.temporal == 'Done':
      maps = self.backbone(x) # (B,C,T,H,W)
      x = self.avgpool(maps) # (B,C,1,1,1) not sure
      x = x.reshape(x.shape[0],-1) # (B,C)
      return x, maps
    elif self.temporal == 'NoPool':
      x = self.backbone(x) # (B,C,T,H,W)
      return x

class Resnet50_s1(nn.Module):
  def __init__(self,pooling=True,stride=1):
    super(Resnet50_s1,self).__init__()
    original = models.resnet50(pretrained=True).state_dict()
    self.backbone = ResNet(last_stride=stride)
    for key in original:
      if key.find('fc') != -1:
        continue
      self.backbone.state_dict()[key].copy_(original[key])
    del original
    if pooling == True:
      self.add_module('avgpool',nn.AdaptiveAvgPool2d(1))
    else:
      self.avgpool = None

    self.out_dim = 2048

  def forward(self,x):
    x = self.backbone(x)
    if self.avgpool is not None:
      x = self.avgpool(x)
      x = x.view(x.shape[0],-1)
    return x


##################### Small Block ###################################
class NonLocalBlock(nn.Module):
  def __init__(self, in_channels, inter_channels=None,sub_sample=False, bn_layer=True,instance='soft'):
    super(NonLocalBlock, self).__init__()
    self.sub_sample = sub_sample
    self.instance = instance
    self.in_channels = in_channels
    self.inter_channels = inter_channels

    if self.inter_channels is None:
      self.inter_channels = in_channels // 2
      if self.inter_channels == 0:
        self.inter_channels = 1

    conv_nd = nn.Conv3d
    max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
    bn = nn.BatchNorm3d

    self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
             kernel_size=1, stride=1, padding=0)
    if bn_layer:
      self.W = nn.Sequential(
        conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
            kernel_size=1, stride=1, padding=0),
        bn(self.in_channels)
      )
      nn.init.constant_(self.W[1].weight, 0)
      nn.init.constant_(self.W[1].bias, 0)
    else:
      self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
               kernel_size=1, stride=1, padding=0)
      nn.init.constant_(self.W.weight, 0)
      nn.init.constant_(self.W.bias, 0)

    self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
               kernel_size=1, stride=1, padding=0)
    self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
               kernel_size=1, stride=1, padding=0)
    if sub_sample:
      self.g = nn.Sequential(self.g, max_pool_layer)
      self.phi = nn.Sequential(self.phi, max_pool_layer)
    
  def forward(self, x):
    '''
    :param x: (b, c, t, h, w)
    :return:
    '''
    batch_size = x.size(0)

    g_x = self.g(x).view(batch_size, self.inter_channels, -1)
    g_x = g_x.permute(0, 2, 1)

    theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
    theta_x = theta_x.permute(0, 2, 1)
    phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
    f = torch.matmul(theta_x, phi_x)
    if self.instance == 'soft':
      f_div_C = F.softmax(f, dim=-1)
    elif self.instance == 'dot':
      f_div_C = f / f.shape[1]

    y = torch.matmul(f_div_C, g_x)
    y = y.permute(0, 2, 1).contiguous()
    y = y.view(batch_size, self.inter_channels, *x.size()[2:])
    W_y = self.W(y)
    z = W_y + x
    
    return z

class Stripe_NonLocalBlock(nn.Module):
  def __init__(self,stripe,in_channels,inter_channels=None,pool_type='mean',instance='soft'):
    super(Stripe_NonLocalBlock,self).__init__()
    self.instance = instance
    self.stripe=stripe
    self.in_channels = in_channels
    self.pool_type = pool_type
    if pool_type == 'max':
      self.pool = nn.AdaptiveMaxPool2d(1)
    elif pool_type == 'mean':
      self.pool = nn.AdaptiveAvgPool2d(1)
    elif pool_type == 'meanmax':
      self.avgpool = nn.AdaptiveAvgPool2d(1)
      self.maxpool = nn.AdaptiveMaxPool2d(1)
      self.in_channels*=2
    if inter_channels == None:
      self.inter_channels = in_channels//2 
    else:
      self.inter_channels = inter_channels

    self.g = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
             kernel_size=1, stride=1, padding=0)
    self.theta = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
               kernel_size=1, stride=1, padding=0)
    self.phi = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
               kernel_size=1, stride=1, padding=0)
    if pool_type == 'meanmax':
      self.in_channels //=2
    
    self.W = nn.Sequential(
      nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
          kernel_size=1, stride=1, padding=0),
      nn.BatchNorm3d(self.in_channels)
    )
    nn.init.constant_(self.W[1].weight, 0)
    nn.init.constant_(self.W[1].bias, 0)

  def forward(self,x):
    # x.shape = (b,c,t,h,w)
    b,c,t,h,w = x.shape
    assert self.stripe * (h//self.stripe) == h

    if self.pool_type == 'meanmax': 
      discri_a = self.avgpool(x.reshape(b*c*t,self.stripe,(h//self.stripe),w)).reshape(b,c,t,self.stripe,1)
      discri_m = self.maxpool(x.reshape(b*c*t,self.stripe,(h//self.stripe),w)).reshape(b,c,t,self.stripe,1)
      discri = torch.cat([discri_a,discri_m],dim=1)
    else:
      discri = self.pool(x.reshape(b*c*t,self.stripe,(h//self.stripe),w)).reshape(b,c,t,self.stripe,1)
    g = self.g(discri).reshape(b,self.inter_channels,-1)
    g = g.permute(0,2,1)
    theta = self.theta(discri).reshape(b, self.inter_channels, -1)
    theta = theta.permute(0,2,1)
    phi = self.phi(discri).reshape(b, self.inter_channels, -1)

    f = torch.matmul(theta, phi)
    if self.instance == 'soft':
      f_div_C = F.softmax(f, dim=-1)
    elif self.instance == 'dot':
      f_div_C = f / f.shape[1]

    y = torch.matmul(f_div_C, g)
    y = y.permute(0, 2, 1).contiguous()
    y = y.reshape(b, self.inter_channels, *discri.size()[2:])
    W_y = self.W(y)

    W_y = W_y.repeat(1,1,1,1,h//self.stripe*w).reshape(b,c,t,h,w)

    z = W_y + x
    return z 

class Bottleneck(nn.Module):
  expansion = 4
  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)
    out += residual
    out = self.relu(out)

    return out
##############################################################################

############################ backbone model ##################################
class ResNet(nn.Module):
  def __init__(self, last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3]):
    self.inplanes = 64
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(
      block, 512, layers[3], stride=last_stride)

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x

class ResNet_Video_nonlocal(nn.Module):
  def __init__(self,last_stride=1,block=Bottleneck,layers=[3,4,6,3],non_layers=[0,1,1,1]):
    self.inplanes = 64
    super().__init__()
    self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    non_idx = 0
    self.NL_1 = nn.ModuleList([NonLocalBlock(self.inplanes,self.inplanes//2,sub_sample=True) for i in range(non_layers[non_idx])])
    self.NL_1_idx = sorted([layers[0]-(i+1) for i in range(non_layers[non_idx])])
    non_idx += 1
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.NL_2 = nn.ModuleList([NonLocalBlock(self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    self.NL_2_idx = sorted([layers[1]-(i+1) for i in range(non_layers[non_idx])])
    non_idx += 1
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)       
    self.NL_3 = nn.ModuleList([NonLocalBlock(self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    self.NL_3_idx =sorted( [layers[2]-(i+1) for i in range(non_layers[non_idx])])
    non_idx += 1
    #self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
    #self.NL_4 = nn.ModuleList([NonLocalBlock(self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    #self.NL_4_idx = sorted([layers[3]-(i+1) for i in range(non_layers[non_idx])])

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))
    return nn.ModuleList(layers)

  def forward(self, x):
    # x 's shape (B,T,C,H,W)
    B,T,C,H,W = x.shape
    x = x.reshape(B*T,C,H,W)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.maxpool(x)
    
    # Layer 1
    NL1_counter = 0
    if len(self.NL_1_idx)==0: self.NL_1_idx=[-1]
    for i in range(len(self.layer1)):
      x = self.layer1[i](x)
      if i == self.NL_1_idx[NL1_counter]:
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        x = self.NL_1[NL1_counter](x)
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        NL1_counter+=1
    # Layer 2
    NL2_counter = 0
    if len(self.NL_2_idx)==0: self.NL_2_idx=[-1]
    for i in range(len(self.layer2)):
      x = self.layer2[i](x)
      if i == self.NL_2_idx[NL2_counter]:
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        x = self.NL_2[NL2_counter](x)
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        NL2_counter+=1
    # Layer 3
    NL3_counter = 0
    if len(self.NL_3_idx)==0: self.NL_3_idx=[-1]
    for i in range(len(self.layer3)):
      x = self.layer3[i](x)
      if i == self.NL_3_idx[NL3_counter]:
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        x = self.NL_3[NL3_counter](x)
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        NL3_counter+=1
    _,C,H,W = x.shape
    x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
    return x
    # Layer 4
    NL4_counter = 0
    if len(self.NL_4_idx)==0: self.NL_4_idx=[-1]
    for i in range(len(self.layer4)):
      x = self.layer4[i](x)
      if i == self.NL_4_idx[NL4_counter]:
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        x = self.NL_4[NL4_counter](x)
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        NL4_counter+=1
    _,C,H,W = x.shape
    x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
    # Return is (B,C,T,H,W)
    return x

class ResNet_Video_nonlocal_stripe(nn.Module):
  def __init__(self,last_stride=1,block=Bottleneck,layers=[3,4,6,3],non_layers=[0,1,1,1],stripes=[16,16,16,16]):
    self.inplanes = 64
    super().__init__()
    self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    non_idx = 0
    self.NL_1 = nn.ModuleList([Stripe_NonLocalBlock(stripes[non_idx],self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    self.NL_1_idx = sorted([layers[0]-(i+1) for i in range(non_layers[non_idx])])
    non_idx += 1
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.NL_2 = nn.ModuleList([Stripe_NonLocalBlock(stripes[non_idx],self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    self.NL_2_idx = sorted([layers[1]-(i+1) for i in range(non_layers[non_idx])])
    non_idx += 1
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)       
    self.NL_3 = nn.ModuleList([Stripe_NonLocalBlock(stripes[non_idx],self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    self.NL_3_idx =sorted( [layers[2]-(i+1) for i in range(non_layers[non_idx])])
    non_idx += 1
    self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
    self.NL_4 = nn.ModuleList([Stripe_NonLocalBlock(stripes[non_idx],self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    self.NL_4_idx = sorted([layers[3]-(i+1) for i in range(non_layers[non_idx])])

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))
    return nn.ModuleList(layers)

  def forward(self, x):
    # x 's shape (B,T,C,H,W)
    B,T,C,H,W = x.shape
    x = x.reshape(B*T,C,H,W)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.maxpool(x)
    
    # Layer 1
    NL1_counter = 0
    if len(self.NL_1_idx)==0: self.NL_1_idx=[-1]
    for i in range(len(self.layer1)):
      x = self.layer1[i](x)
      if i == self.NL_1_idx[NL1_counter]:
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        x = self.NL_1[NL1_counter](x)
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        NL1_counter+=1
    # Layer 2
    NL2_counter = 0
    if len(self.NL_2_idx)==0: self.NL_2_idx=[-1]
    for i in range(len(self.layer2)):
      x = self.layer2[i](x)
      if i == self.NL_2_idx[NL2_counter]:
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        x = self.NL_2[NL2_counter](x)
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        NL2_counter+=1
    # Layer 3
    NL3_counter = 0
    if len(self.NL_3_idx)==0: self.NL_3_idx=[-1]
    for i in range(len(self.layer3)):
      x = self.layer3[i](x)
      if i == self.NL_3_idx[NL3_counter]:
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        x = self.NL_3[NL3_counter](x)
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        NL3_counter+=1
    # Layer 4
    NL4_counter = 0
    if len(self.NL_4_idx)==0: self.NL_4_idx=[-1]
    for i in range(len(self.layer4)):
      x = self.layer4[i](x)
      if i == self.NL_4_idx[NL4_counter]:
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        x = self.NL_4[NL4_counter](x)
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        NL4_counter+=1
    _,C,H,W = x.shape
    x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
    # Return is (B,C,T,H,W)
    return x

class ResNet_Video_nonlocal_hr(nn.Module):
  def __init__(self,last_stride=1,block=Bottleneck,layers=[3,4,6,3],non_layers=[0,1,1,1],stripes=[16,16,16,16]):
    self.inplanes = 64
    super().__init__()
    self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    non_idx = 0
    self.NL_1 = nn.ModuleList([NonLocalBlock(self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    self.NL_1_idx = sorted([layers[0]-(i+1) for i in range(non_layers[non_idx])])
    non_idx += 1
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.NL_2 = nn.ModuleList([NonLocalBlock(self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    self.NL_2_idx = sorted([layers[1]-(i+1) for i in range(non_layers[non_idx])])
    non_idx += 1
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)       
    self.NL_3 = nn.ModuleList([NonLocalBlock(self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    self.NL_3_idx =sorted( [layers[2]-(i+1) for i in range(non_layers[non_idx])])
    non_idx += 1
    self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
    self.NL_4 = nn.ModuleList([NonLocalBlock(self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    self.NL_4_idx = sorted([layers[3]-(i+1) for i in range(non_layers[non_idx])])

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))
    return nn.ModuleList(layers)

  def forward(self, x):
    # x 's shape (B,T,C,H,W)
    B,T,C,H,W = x.shape
    x = x.reshape(B*T,C,H,W)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.maxpool(x)
    # x 's shape (B*T,C,H,W)
    
    # Layer 1
    NL1_counter = 0
    if len(self.NL_1_idx)==0: self.NL_1_idx=[-1]
    for i in range(len(self.layer1)):
      x = self.layer1[i](x)
      if i == self.NL_1_idx[NL1_counter]:
        _,C,H,W = x.shape
        x = x.reshape(-1,2,C,H,W).permute(0,2,1,3,4)
        x = self.NL_1[NL1_counter](x)
        x = x.permute(0,2,1,3,4).reshape(-1,C,H,W)
        # x's shape (B*T//2,2,C,H,W)
        NL1_counter+=1
    # Max pool
    # _,C,H,W = x.shape
    # x = torch.max(x.reshape(-1,2,C,H,W),dim=1)[0]
    # T  = T//2
    # Layer 2
    NL2_counter = 0
    if len(self.NL_2_idx)==0: self.NL_2_idx=[-1]
    for i in range(len(self.layer2)):
      x = self.layer2[i](x)
      if i == self.NL_2_idx[NL2_counter]:
        _,C,H,W = x.shape
        x = x.reshape(-1,T,C,H,W).permute(0,2,1,3,4)
        x = self.NL_2[NL2_counter](x)
        x = x.permute(0,2,1,3,4).reshape(-1,C,H,W)
        # x's shape (B*T//2,2,C,H,W)
        NL2_counter+=1
    # Max pool
    _,C,H,W = x.shape
    x = torch.max(x.reshape(-1,2,C,H,W),dim=1)[0]
    T  = T//2
    # Layer 3
    NL3_counter = 0
    if len(self.NL_3_idx)==0: self.NL_3_idx=[-1]
    for i in range(len(self.layer3)):
      x = self.layer3[i](x)
      if i == self.NL_3_idx[NL3_counter]:
        _,C,H,W = x.shape
        x = x.reshape(-1,T,C,H,W).permute(0,2,1,3,4)
        x = self.NL_3[NL3_counter](x)
        x = x.permute(0,2,1,3,4).reshape(-1,C,H,W)
        # x's shape (B*T//2,2,C,H,W)
        NL3_counter+=1
    # Max pool
    _,C,H,W = x.shape
    x = torch.max(x.reshape(-1,2,C,H,W),dim=1)[0]
    T  = T//2
    # Layer 4
    NL4_counter = 0
    if len(self.NL_4_idx)==0: self.NL_4_idx=[-1]
    for i in range(len(self.layer4)):
      x = self.layer4[i](x)
      if i == self.NL_4_idx[NL4_counter]:
        _,C,H,W = x.shape
        x = x.reshape(-1,T,C,H,W).permute(0,2,1,3,4)
        x = self.NL_4[NL4_counter](x)
        x = x.permute(0,2,1,3,4).reshape(-1,C,H,W)
        NL4_counter+=1
    _,C,H,W = x.shape
    x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
    # Return is (B,C,T,H,W)
    return x

class ResNet_Video_nonlocal_stripe_hr(nn.Module):
  def __init__(self,last_stride=1,block=Bottleneck,layers=[3,4,6,3],non_layers=[0,1,1,1],stripes=[16,16,16,16]):
    self.inplanes = 64
    super().__init__()
    self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    non_idx = 0
    self.NL_1 = nn.ModuleList([Stripe_NonLocalBlock(stripes[non_idx],self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    self.NL_1_idx = sorted([layers[0]-(i+1) for i in range(non_layers[non_idx])])
    non_idx += 1
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.NL_2 = nn.ModuleList([Stripe_NonLocalBlock(stripes[non_idx],self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    self.NL_2_idx = sorted([layers[1]-(i+1) for i in range(non_layers[non_idx])])
    non_idx += 1
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)       
    self.NL_3 = nn.ModuleList([Stripe_NonLocalBlock(stripes[non_idx],self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    self.NL_3_idx =sorted( [layers[2]-(i+1) for i in range(non_layers[non_idx])])
    non_idx += 1
    self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
    self.NL_4 = nn.ModuleList([Stripe_NonLocalBlock(stripes[non_idx],self.inplanes,self.inplanes//2) for i in range(non_layers[non_idx])])
    self.NL_4_idx = sorted([layers[3]-(i+1) for i in range(non_layers[non_idx])])

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))
    return nn.ModuleList(layers)

  def forward(self, x):
    # x 's shape (B,T,C,H,W)
    B,T,C,H,W = x.shape
    x = x.reshape(B*T,C,H,W)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.maxpool(x)
    # x 's shape (B*T,C,H,W)
    
    # Layer 1
    NL1_counter = 0
    if len(self.NL_1_idx)==0: self.NL_1_idx=[-1]
    for i in range(len(self.layer1)):
      x = self.layer1[i](x)
      if i == self.NL_1_idx[NL1_counter]:
        _,C,H,W = x.shape
        x = x.reshape(-1,2,C,H,W).permute(0,2,1,3,4)
        x = self.NL_1[NL1_counter](x)
        x = x.permute(0,2,1,3,4).reshape(-1,C,H,W)
        # x's shape (B*T//2,2,C,H,W)
        NL1_counter+=1
    # Max pool
    # _,C,H,W = x.shape
    # x = torch.max(x.reshape(-1,2,C,H,W),dim=1)[0]
    # T  = T//2
    # Layer 2
    NL2_counter = 0
    if len(self.NL_2_idx)==0: self.NL_2_idx=[-1]
    for i in range(len(self.layer2)):
      x = self.layer2[i](x)
      if i == self.NL_2_idx[NL2_counter]:
        _,C,H,W = x.shape
        x = x.reshape(-1,2,C,H,W).permute(0,2,1,3,4)
        x = self.NL_2[NL2_counter](x)
        x = x.permute(0,2,1,3,4).reshape(-1,C,H,W)
        # x's shape (B*T//2,2,C,H,W)
        NL2_counter+=1
    # Max pool
    _,C,H,W = x.shape
    x = torch.max(x.reshape(-1,2,C,H,W),dim=1)[0]
    T  = T//2
    # Layer 3
    NL3_counter = 0
    if len(self.NL_3_idx)==0: self.NL_3_idx=[-1]
    for i in range(len(self.layer3)):
      x = self.layer3[i](x)
      if i == self.NL_3_idx[NL3_counter]:
        _,C,H,W = x.shape
        x = x.reshape(-1,2,C,H,W).permute(0,2,1,3,4)
        x = self.NL_3[NL3_counter](x)
        x = x.permute(0,2,1,3,4).reshape(-1,C,H,W)
        # x's shape (B*T//2,2,C,H,W)
        NL3_counter+=1
    # Max pool
    _,C,H,W = x.shape
    x = torch.max(x.reshape(-1,2,C,H,W),dim=1)[0]
    T  = T//2
    # Layer 4
    NL4_counter = 0
    if len(self.NL_4_idx)==0: self.NL_4_idx=[-1]
    for i in range(len(self.layer4)):
      x = self.layer4[i](x)
      if i == self.NL_4_idx[NL4_counter]:
        _,C,H,W = x.shape
        x = x.reshape(-1,2,C,H,W).permute(0,2,1,3,4)
        x = self.NL_4[NL4_counter](x)
        x = x.permute(0,2,1,3,4).reshape(-1,C,H,W)
        NL4_counter+=1
    _,C,H,W = x.shape
    x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
    # Return is (B,C,T,H,W)
    return x
if __name__ == "__main__":
  net = ResNet(last_stride=1)
  print(net)
  import torch

  x = net(torch.zeros(1, 3, 256, 128))
  print(x.shape)
