#!/usr/bin/env python3
"""
READ: Reciprocal Attention Discriminator for Image-to-Video Re-Identification
"""
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')
from model.MemoryNetwork import MemoryEncoder, MemoryNetwork
from utils.args import get_args
from utils.ckpt_utils import setCheckpointFileDict
from utils.iter_utils import testIters
from utils.misc_utils import correctK
from utils.metric_utils import auprc_average_precision_score

RGB_INPUT_SHAPE = (3,256,128)
ALL_MODELS = ['memory_encoder', 'memory_network'] 
LOG_PREFIX = 'READ'

def main(args):
  margs = {}

  # Use cuda device if available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  margs['device'] = device

  # Set dataloader for requested dataset
  if args.dataset == 'mars':
    from dataloader.MARSLoader import MARSLoader
    margs['dataloader'] = MARSLoader
    margs['gallery_available'] = True
  elif args.dataset == 'duke':
    from dataloader.DukeVideoReIDLoader import DukeVideoReIDLoader
    margs['dataloader'] = DukeVideoReIDLoader
    margs['gallery_available'] = True
  else:
    raise NotImplementedError('Dataset {} is not supported in this code'
                              ''.format(args.dataset))

  models = build_models(args, device=device)

  checkpoint_files = setCheckpointFileDict(ALL_MODELS, args.checkpoint_files)

  if args.for_what == 'test':
    main_test(checkpoint_files, args, margs, models)

  else:
    raise NotImplementedError(
      'Given "{}" mode is not implemented'.format(args.for_what)
      )

def build_models(args, device='cuda'):
  models = {}

  models['memory_network'] = MemoryNetwork(input_shape=RGB_INPUT_SHAPE,
                                           out_size=args.classifier_out_size)
 
  models['memory_encoder'] = MemoryEncoder()
  if torch.cuda.is_available():
    for m in models:
      models[m] = nn.DataParallel(models[m].to(device))

  return models

def main_test(checkpoint_files, args, margs, models):
  test_loader, test_dataset = margs['dataloader'](
    json_file=args.json_path, 
    h5_dir=args.h5_dir, 
    subset='test',
    max_sample_length=args.max_sample_length,
    visual_transform=args.visual_transform,
    batch_size=args.batch_size, 
    shuffle=True,
    num_workers=args.num_workers, 
    pin_memory=False,
    drop_last=False
    ) 

  if margs['gallery_available']:
    gallery_loader, gallery_dataset = margs['dataloader'](
      json_file=args.json_path, 
      h5_dir=args.h5_dir, 
      subset='test_gallery',
      max_sample_length=args.max_sample_length,
      visual_transform=args.visual_transform,
      batch_size=args.batch_size, 
      shuffle=False,
      num_workers=args.num_workers, 
      pin_memory=False,
      drop_last=False
    ) 
  else:
    gallery_loader, gallery_dataset = None, None

  testIters(run, test_loader, test_dataset,  
            gallery_loader=gallery_loader, gallery_dataset=gallery_dataset,
            models=models, checkpoint_files=checkpoint_files, 
            args=args, device=margs['device'])

def run(split, models, memories, identities, sample=None, target_modules=[], device='cuda',
        optimizers=None, criteria=None, args=None, memory_op=False):
  if split == 'train':
    training = True
    for m in models:
      if m in target_modules:
        models[m].train()
        optimizers[m].zero_grad()
      else:
        models[m].eval()
  else:
    training = False
    for m in models:
      models[m].eval()

  result = {}
  result['logs'] = {}
  result['output'] = {}

  if len(sample['images'].size()) == 6: # (batch_size, samples_per_batch, sample_length, C, H, W)
    # if there's no samples_per_batch, sample['images'] will be (batch_size, sample_length, C, H, W)
    # so match the size if samples_per_batch is given
    for key in sample: # key: images, id, camera, memory_images, memory_id, memory_camera
      sample[key] = sample[key].view((-1,) + sample[key].size()[2:]) # (batch_size*samples_per_batch, ...)

  if memory_op == 'compute':
    key, value = models['memory_encoder'](
      period=sample['images'].to(device)
      )
    for mem_ind, memory_id in enumerate(sample['id']):
      identities.append(memory_id)
      new_memory = {
        'key': key[mem_ind].to('cpu'), 
        'value': value[mem_ind].to('cpu'),
        'camera': sample['camera'][mem_ind]
        }
      memories.append(new_memory)
    return 0

  batch_size = len(sample['id'])

  assert len(memories) >= 10, "Memory length (batch size when training) " \
                              "should be >= 10 for top10 accuracy evaluation"
  logits = []
  features = []
  # cycle through memories and calculate probabilities
  memory_key_batch = []
  memory_value_batch = []
  for id_ind, p_id in enumerate(identities):
    memory_key_batch.append(memories[id_ind]['key'])
    memory_value_batch.append(memories[id_ind]['value'])
    if len(memory_key_batch) < batch_size and id_ind < len(identities) - 1:
      continue
    memory_key_batch = torch.stack(memory_key_batch, dim=0) # expand batch dim
    memory_value_batch = torch.stack(memory_value_batch, dim=0) # expand batch dim
    
    # Input:  sample['images'] (query_batch x Tq x C x H x W)
    #         memory_key: Tensor(memory_batch x Tm x C/8 x H x W)
    #         memory_value: Tensor(memory_batch x Tm x C/2 x H x W)
    # Output: logit (query_batch x memory_batch x Tq x classifier_out_size)
    #         feature (query_batch x memory_batch x Tq x feat_dim)
    logit, feature = models['memory_network'](
      sample['images'].to(device),
      memory_key_batch.repeat(torch.cuda.device_count(),1,1,1,1),
      memory_value_batch.repeat(torch.cuda.device_count(),1,1,1,1)
      ) # torch.cuda.device_count() to avoid splitting by DataParallel module

    logits.append(logit)
    features.append(feature)
    
    memory_key_batch = []
    memory_value_batch = []
  logits = torch.cat(logits, dim=1) # Bq x num_memory x Tq x classifier_out_size
  features = torch.cat(features, dim=1) # Bq x num_memory x Tq x feat_dim
  result['output']['logits'] = logits

  # Make answers or targets for evaluation
  num_memory = len(memories)
  memory_identities = np.array(identities) # (num_memory,)
  memory_identities = np.expand_dims(memory_identities, axis=0) # 1 x num_memory
  memory_identities = memory_identities.repeat(batch_size, axis=0) # b x num_memory
  sample_identities = np.stack(num_memory*[sample['id']], axis=1) # b x num_memory
  equal_identities = (sample_identities == memory_identities)
  ce_target = torch.Tensor(
    np.array(equal_identities, dtype=np.uint8)
    ).to(device, dtype=torch.long) # b x num_memory

  if not training and 'camera' in sample and sample['camera'][0] != -1: # under multi-camera setting
    memory_cameras = np.array([m['camera'] for m in memories]) # (num_memory,)
    memory_cameras = np.expand_dims(memory_cameras, axis=0) # 1 x num_memory
    memory_cameras = memory_cameras.repeat(batch_size, axis=0) # b x num_memory
    sample_cameras = np.stack(num_memory*[sample['camera']], axis=1) # b x num_memory

  # Evaluation
  result['logs']['average_precision'] = []
  if models['memory_network'].module.out_size == 1:
    logits_mean = torch.mean(logits, dim=2, keepdim=False).squeeze(-1).detach().cpu() # Bq x num_memory
  else:
    logits_mean = torch.mean(logits, dim=2, keepdim=False)[:,:,1].detach().cpu() 
      # Bq x num_memory, two-class output to binary values with [:,:,1]
  logits_softmax_batch_wise = F.softmax(logits_mean, dim=1)
  ce_target_cpu = ce_target.cpu().numpy()
  if not training and 'camera' in sample and sample['camera'][0] != -1:
    result['logs']['top10'] = []
    result['logs']['top5'] = []
    result['logs']['top1'] = []
  num_no_groundtruth = 0
  for logits_ind, logits_batch in enumerate(logits_softmax_batch_wise.numpy()):
    if ce_target_cpu[logits_ind].any() == False: # no matching ground truth counterpart in gallery
      num_no_groundtruth += 1
      continue
    if not training and 'camera' in sample and sample['camera'][0] != -1: # under multi-camera setting
      # Filter out the same id and same camera
      valid = ((memory_identities[logits_ind] != sample_identities[logits_ind]) |
               (memory_cameras[logits_ind] != sample_cameras[logits_ind]))
      logits_mean_valid = logits_mean[logits_ind].masked_select(torch.from_numpy(valid.astype(np.uint8)))
      result['logs']['average_precision'].append(
        auprc_average_precision_score(ce_target_cpu[logits_ind][valid], logits_batch[valid])
      )
      correct_top10 = correctK(
        output_scores=logits_mean_valid.unsqueeze(dim=0), 
        output_classes=torch.from_numpy(memory_identities[logits_ind][valid].astype(np.int32)).unsqueeze(dim=0),
        target_classes=torch.from_numpy(np.array([sample['id'][logits_ind]], dtype=np.int32)).unsqueeze(-1),
        topk=10
      )
      result['logs']['top10'].append(correct_top10[:,:10].any(dim=1).item())
      result['logs']['top5'].append(correct_top10[:,:5].any(dim=1).item())
      result['logs']['top1'].append(correct_top10[:,:1].any(dim=1).item())
    else:
      result['logs']['average_precision'].append(
        auprc_average_precision_score(ce_target_cpu[logits_ind], logits_batch)
      )
  if num_no_groundtruth > 0:
    print('{} queries do not have groundtruth.'.format(num_no_groundtruth))
    
  return result

if __name__ == "__main__":
  args = get_args(ALL_MODELS)
  
  main(args)
