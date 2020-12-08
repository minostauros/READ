#!/usr/bin/env python3
import time
import math

def asDays(s):
  d = s // 86400
  if d > 0:
    d = '%dd ' % d
  else:
    d = ''
  s = s % 86400
  h = s // 3600
  if h > 0:
    h = '%dh ' % h
  else:
    h = ''
  s = s % 3600
  m = s // 60
  if m > 0:
    m = '%dm ' % m
  else:
    m = ''
  s = int(s % 60)
  
  return '{}{}{}{:d}s'.format(d, h, m, s)

def timeSince(since, percent=None):
  now = time.time()
  s = now - since
  if percent is not None:
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asDays(s), asDays(rs))
  else:
    return '%s' % (asDays(s))