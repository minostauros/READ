#!/usr/bin/env python3
import os
import json
import math
import h5py
import random
import operator
import collections
import numpy as np
from PIL import Image
from itertools import groupby
from operator import itemgetter

import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class MARS(Dataset):
  """
  MARS Dataset

  Reference:
      Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

  URL: `<http://www.liangzheng.com.cn/Project/project_mars.html>`_
  
  Dataset statistics:
      - identities: 1261.
      - tracklets: 8298 (train) + 1980 (query) + 11310 (gallery).
      - cameras: 6.
  """
  def __init__(self, 
    json_file,
    h5_dir, 
    subset,
    same_id_prob=1.0,
    max_sample_length=4,
    samples_per_batch=1,
    visual_transform='normalize'):
    """
    Args:
      json_file (string): Path to the json file with tracklet names and subset 
        that each video belongs to.
      h5_dir (string): Directory with h5 files containing data.
      subset (string): 'train', 'test', or 'test_gallery'. ('test' is for query)
      same_id_prob (float): Only works for TRAINING set. Probability of 
        returning same id for images to be put into memory. MARS Set dataset 
        returns a sample for query, alongside with another sample to be put into
        memory. See return values. 
      max_sample_length (int): Each group or period of frames may contain too 
        many frames. Limit the number of frames for each group with this 
        parameter.
      samples_per_batch (int): sample multiple samples from a single ID.
        Only used for subset train.
      visual_transform (string): None or 'normalize'. If 'normalize', rgb images
        are normalized with ImageNet statistics.
    """

    with open(json_file, 'r', encoding='utf-8') as fp:
      self.meta = json.load(fp, object_pairs_hook=collections.OrderedDict)

    self.h5_dir = h5_dir
    self.subset = subset
    if self.subset == 'test':
      self.subset = 'test_query'
    self.same_id_prob = same_id_prob
    self.max_sample_length = max_sample_length
    self.samples_per_batch = samples_per_batch
    self.visual_transform = visual_transform

    self.rng = random.SystemRandom()

    # divide TRAINING set separated by person_ids
    if self.subset == 'train':
      self.set_train = {}
      for path in self.meta['train']:
        person_id = self.meta['train'][path]['person_id']
        if person_id not in self.set_train:
          self.set_train[person_id] = {}
        self.set_train[person_id][path] = self.meta['train'][path]

    self.h5_path = os.path.join(
      self.h5_dir, 'mars_tracklets_{}.h5'.format(self.subset)
      )

    self.height = 256 # height to be returned
    self.width = 128 # width to be returned

    self.tracklets = [ path for path in self.meta[self.subset] ]
    # One eternity later, I discovered that MARS evaluation has a following 
    # problem: https://github.com/liangzheng06/MARS-evaluation/issues/4#issuecomment-469406381
    # So as to keep comparison fair, gallery set should contain query set
    # For backward compatibility of code, issue above will be handled in this
    # code rather than fixing input JSON file.
    if self.subset == 'test_gallery':
      self.tracklets += [ path for path in self.meta['test_query'] ]
      self.h5_path_test_query = os.path.join(
        self.h5_dir, 'mars_tracklets_{}.h5'.format('test_query')
        )

    self.tracklets = sorted(self.tracklets) # important for maintaining order

    self.rgb_transform = transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.Resize([self.height,self.width]),
                           transforms.ToTensor()
                           ])
    if self.visual_transform == 'normalize':
      self.rgb_transform = transforms.Compose([
                             self.rgb_transform,
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                             ])

    print('MARS is initialized. '
          '(Subset: {}, Length: {})'.format(self.subset, len(self.tracklets)) )

  def __len__(self):
    return len(self.tracklets)

  def _get_item_by_path(self, tracklet):
    # Again, handle https://github.com/liangzheng06/MARS-evaluation/issues/4#issuecomment-469406381
    if self.subset == 'test_gallery' and tracklet not in self.meta[self.subset]:
      info = self.meta['test_query'][tracklet]
      h5_path = self.h5_path_test_query
    else:
      h5_path = self.h5_path
      info = self.meta[self.subset][tracklet]
    length = info['length']
    person_id = info['person_id']
    camera_id = info['camera_id']
    track_id = info['track_id']

    segment_len = info['length']
    frame_inds = []
    if self.subset == 'test_query':
      frame_inds = [0] # Image to Video ReID setting
    elif segment_len <= self.max_sample_length:
      frame_inds = list(range(segment_len))
      # pad frames if length is shorter than sample length
      if len(frame_inds) < self.max_sample_length:
        frame_inds_to_pad = []
        if self.subset == 'train': # random sample padding
          while (len(frame_inds) + len(frame_inds_to_pad)) < self.max_sample_length:
            frame_inds_to_pad.append(self.rng.sample(frame_inds, k=1)[0])
        else: # when testing just repeat last frame
          frame_inds_to_pad = [frame_inds[-1]]*(self.max_sample_length - len(frame_inds))
        frame_inds = sorted(frame_inds + frame_inds_to_pad)
    else:
      # restricted random sampling: divide tracklet into N(wanted #frames) strips, sample from each segment
      range_list = list(range(segment_len))
      interval = math.ceil(segment_len/self.max_sample_length)
      strip = range_list+[range_list[-1]]*(interval*self.max_sample_length-segment_len)
      for s in range(self.max_sample_length):
        pool = strip[s*interval:(s+1)*interval]
        if self.subset == 'train':
          frame_inds.append( self.rng.sample(pool, k=1)[0] )
        else:
          frame_inds.append( pool[0] )

    h5 = h5py.File(h5_path, 'r')

    # Get images
    images = []
    for img_ind in frame_inds:
      img_byte = h5['/tracklets' + tracklet][img_ind]
      img = cv2.imdecode(img_byte, flags=cv2.IMREAD_COLOR)
      img = self.rgb_transform(img)
      images.append(img)
    images = torch.stack(images, dim=0)

    h5.close()

    # random horizontal flip for training
    if self.subset == 'train':
      if self.rng.random()  < 0.5: # 50% probability
        images = torch.flip(images, dims=[3])

    sample = {
      'camera': camera_id,
      'id': person_id,
      'images': images
    }

    return sample

  def __getitem__(self, idx):
    tracklet = self.tracklets[idx]
    sample = self._get_item_by_path(tracklet)

    if self.subset == 'train':
      samples = []
      samples.append(sample)
      if self.samples_per_batch > 1:
        # sample multiple samples from a single ID
        tracks_to_add = []
        paths = [ path for path in self.set_train[sample['id']] if path != tracklet]
        if len(paths) < self.samples_per_batch-1:
          # if num of same id tracks is not enough, get from other ids
          tracks_to_add.extend(paths)
          paths = []
          p_ids = [p_id for p_id in self.set_train if p_id != sample['id']]
          for p_id in p_ids:
            paths.extend([path for path in self.set_train[p_id]])
          tracks_to_add.extend(
            self.rng.sample(
              paths, 
              k=self.samples_per_batch - 1 - len(tracks_to_add)
              )
            )
        else:
          tracks_to_add.extend(
            self.rng.sample(paths, k=self.samples_per_batch-1)
            )
        for path in tracks_to_add:
          samples.append(self._get_item_by_path(path))

      memory_samples = []
      for sample in samples:
        # sample counterpart memory sample with same ID
        paths = [ path for path in self.set_train[sample['id']] if path != tracklet]

        if len(paths) < 1:
          p_ids = [p_id for p_id in self.set_train if p_id != sample['id']]
          for p_id in p_ids:
            paths.extend([path for path in self.set_train[p_id]])

        path = self.rng.sample(paths, k=1)[0]
        memory_samples.append( self._get_item_by_path(path) )

      sample = {
        'images': torch.stack([s['images'] for s in samples], dim=0), # (samples_per_batch, max_sample_length, C, H, W)
        'id': torch.stack([torch.LongTensor([s['id']]) for s in samples], dim=0).squeeze(dim=-1),
        'camera': torch.stack([torch.IntTensor([s['camera']]) for s in samples], dim=0).squeeze(dim=-1),
        'memory_images': torch.stack([s['images'] for s in memory_samples], dim=0),
        'memory_id': torch.stack([torch.LongTensor([s['id']]) for s in memory_samples], dim=0).squeeze(dim=-1),
        'memory_camera': torch.stack([torch.IntTensor([s['camera']]) for s in memory_samples], dim=0).squeeze(dim=-1)
      }

    return sample

def MARSLoader(
    json_file, 
    h5_dir, 
    subset,
    same_id_prob=1.0,
    max_sample_length=4,
    samples_per_batch=1,
    visual_transform='normalize',
    batch_size=1, 
    shuffle=True, 
    num_workers=1, 
    pin_memory=False,
    drop_last=False,
    **kwargs # unkown arguments are ignored
    ):

  if len(kwargs) > 0:
    print('MARSLoader: arguments {} are ignored.'.format(kwargs))

  # load dataset
  dataset = MARS(
    json_file=json_file, 
    h5_dir=h5_dir, 
    subset=subset,
    same_id_prob=same_id_prob,
    max_sample_length=max_sample_length,
    samples_per_batch=samples_per_batch,
    visual_transform=visual_transform
    )

  # data loader for custom dataset
  data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=drop_last
    # collate_fn=custom_collate_fn
    )

  return data_loader, dataset
