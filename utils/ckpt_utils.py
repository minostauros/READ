#!/usr/bin/env python3
import torch

def setCheckpointFileDict(all_models, checkpoint_files):
  # Assign given checkpoint files to each model
  output = {}
  if len(checkpoint_files) > 0:
    models_name = all_models + ['iter']
    if 'all' in checkpoint_files:
      for m in models_name:
        output[m] = checkpoint_files['all']
      checkpoint_files.pop('all')
    elif 'else' in checkpoint_files:
      for m in models_name:
        output[m] = checkpoint_files['else']
      checkpoint_files.pop('else')
    if len(checkpoint_files) > 0:
      for m in checkpoint_files:
        output[m] = checkpoint_files[m]

  return output

def loadCheckpoints(models, modules, optimizers, checkpoint_files={}, 
                    load_optimizers=True):
  # checkpoint_files should specify model name as key and to specify starting 
  # iteration number, 'iter' key should be in checkpoint_files
  if len(checkpoint_files) > 0:
    for m in checkpoint_files:
      if m == 'iter':
        continue
      checkpoint = torch.load(checkpoint_files[m])
      trained_dict = checkpoint[m]
      model_dict = models[m].state_dict()

      # 1. filter out unnecessary keys
      not_used_keys = {k: v for k, v in trained_dict.items() if k not in model_dict}
      trained_dict = {k: v for k, v in trained_dict.items() if k in model_dict}
      if len(not_used_keys.keys()) > 0:
        print('Unused keys in checkpoint for model {}:'.format(m), not_used_keys.keys())
      # 2. overwrite entries in the existing state dict
      model_dict.update(trained_dict) 
      # 3. load the new state dict
      models[m].load_state_dict(trained_dict)
      print('{} checkpoint file loaded ({})'.format(m, checkpoint_files[m]))

      if load_optimizers:
        if m in modules:
          if '{}_optimizer'.format(m) in checkpoint:
            optimizers[m].load_state_dict(checkpoint['{}_optimizer'.format(m)])
            print('{}_optimizer checkpoint file loaded ({})'.format(m, checkpoint_files[m]))
          else:
            print('{}_optimizer is not in the checkpoint file. '
                  'Not loaded ({})'.format(m, checkpoint_files[m]))
      del model_dict
      del trained_dict
      del checkpoint
      
    if 'iter' in checkpoint_files:
      checkpoint = torch.load(checkpoint_files['iter'], map_location=torch.device('cpu'))
      if 'epoch' in checkpoint:
        epoch_start = checkpoint['epoch'] + 1
      else:
        epoch_start = 1
      if 'iter' in checkpoint:
        iteration = checkpoint['iter'] + 1
      else:
        iteration = 1
      del checkpoint

      return epoch_start, iteration

  return 1, 1
