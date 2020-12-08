#!/usr/bin/env python3

def cleanList(mylist):
  if len(mylist) > 0:
    del mylist[:]

def correctK(output_scores, output_classes, target_classes, topk=3):
  """
  Computes the precision@k for the specified values of k

  :params:
  output_scores: (b x num_instances) torch.Tensor
  output_classes: (b x num_instances) torch.Tensor
  target_classes: (b x 1) torch.Tensor

  :returns:
  correct_k: (b x topk) dtype=torch.uint8
  """
  assert len(target_classes.size()) == 2 and target_classes.size(1) == 1
  batch_size = output_scores.size(0)

  _, pred_indices = output_scores.topk(topk, dim=1, largest=True, sorted=True)
  predictions = output_classes.gather(dim=1,index=pred_indices)
  correct_k = predictions.eq(target_classes.expand_as(predictions))

  return correct_k