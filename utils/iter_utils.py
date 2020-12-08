#!/usr/bin/env python3
import math
import time
from tqdm import tqdm

import torch

from .time_utils import timeSince
from .ckpt_utils import loadCheckpoints

def testIters(run, test_loader, test_dataset, 
              gallery_loader=None, gallery_dataset=None,
              models={}, checkpoint_files={},
              args=None, device='cuda'):
  assert len(models) > 0, 'No model is given for testing. '

  for m in models:
    for param in models[m].parameters():
      param.requires_grad = False

  loadCheckpoints(models, [], {}, checkpoint_files)

  print('Test ongoing...')
  test_memories = []
  test_identities = []
  test_size = math.ceil(len(test_dataset) / args.batch_size)

  with torch.no_grad():
    """
    Query & Gallery
      If we have dedicated gallery_loader, test_loader is for queries.
    First put all gallery into memory, and then classify queries. Mind GPU usage 
    depends on the size of IDs in the whole gallery. Length of each segment in
    memory could be variable, since space-time memory reading is length-ignorant
    in temporal dimension.
    """
    if gallery_loader is not None and len(test_identities) < 1:
      #   Add memories to RAM first, and send to GPU just before Memory Read, to
      # save GPU memory usage. 
      print('Adding galleries to memory...')
      for gal_batch_ind, gal_sample in enumerate(tqdm(gallery_loader)):
        run(split='test',
            sample=gal_sample, 
            memories=test_memories,
            identities=test_identities,
            models=models, 
            device=device,
            memory_op='compute') # add_memory
      print('In galleries: {} tracklets are present.'.format(len(test_memories)))
    
    test_start = time.time()
    test_logs = {}
    for test_batch_ind, test_sample in enumerate(test_loader):
      test_result = run(split='test',
                        sample=test_sample, 
                        memories=test_memories,
                        identities=test_identities,
                        models=models, 
                        args=args,
                        device=device)

      if gallery_loader is None:
        raise NotImplementedError()

      for log in test_result['logs']:
        if log not in test_logs:
          test_logs[log] = []
        if isinstance(test_result['logs'][log], list):
          for log_ind, a_log in enumerate(test_result['logs'][log]):
            if a_log != a_log: # isnan
              test_result['logs'][log][log_ind] = 0
          test_logs[log].extend(test_result['logs'][log])  
        else:
          if test_result['logs'][log] != test_result['logs'][log]: # isnan
            test_result['logs'][log] = 0
          test_logs[log].append(test_result['logs'][log])

      print('{}: Test Sample: {:d}, ET: {}'.format(
        'READ', test_batch_ind + 1, 
        timeSince(test_start, (test_batch_ind + 1) / test_size)
        ) )
      for metric in ['average_precision', 'top10', 'top5', 'top1']:
        print('batch_{}: {}'.format(metric, torch.Tensor(test_result['logs'][metric]).mean()))

    test_log_mean = {}
    for log in test_logs:
      test_log_mean[log] = torch.Tensor(test_logs[log]).mean()

    print('\nTest Results:')
    for metric in ['average_precision', 'top10', 'top5','top1']:
      print('mean_{}: {}'.format(metric, test_log_mean[metric]))
