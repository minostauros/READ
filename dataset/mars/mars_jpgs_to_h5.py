#!/usr/bin/env python3
# Reference: https://kaiyangzhou.github.io/deep-person-reid/_modules/torchreid/data/datasets/video/mars.html
import os
import h5py
import time
import math
import argparse
import numpy as np
from scipy.io import loadmat

def main():
  args = get_args()

  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  ids = os.listdir(args.top_dir)
  ids = [d for d in ids if not d.startswith('.') and os.path.isdir(d)]

  # Retrieve data from info directory
  # From: https://github.com/liangzheng06/MARS-evaluation/tree/master/info
  train_name_path = os.path.join(args.top_dir, 'info/train_name.txt')
  test_name_path = os.path.join(args.top_dir, 'info/test_name.txt')
  track_train_info_path = os.path.join(args.top_dir, 'info/tracks_train_info.mat')
  track_test_info_path = os.path.join(args.top_dir, 'info/tracks_test_info.mat')
  query_IDX_path = os.path.join(args.top_dir, 'info/query_IDX.mat')

  train_names = get_names(train_name_path)
  test_names = get_names(test_name_path)
  track_train = loadmat(track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
  track_test = loadmat(track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
  query_IDX = loadmat(query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
  query_IDX -= 1 # index from 0
  track_query = track_test[query_IDX,:]
  gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
  track_gallery = track_test[gallery_IDX,:]

  subsets = []

  train = process_data(train_names, track_train, args.top_dir, home_dir='bbox_train', relabel=True)
  query = process_data(test_names, track_query, args.top_dir, home_dir='bbox_test', relabel=False)
  gallery = process_data(test_names, track_gallery, args.top_dir, home_dir='bbox_test', relabel=False)

  subsets.append(('train', train))
  subsets.append(('test_query', query))
  subsets.append(('test_gallery', gallery))

  if args.num_worker < 1:
    for split, subset in subsets:
      tracklets_to_h5(args, split, subset)
  else:
    from joblib import Parallel, delayed
    Parallel(n_jobs=args.num_worker)(
      delayed(tracklets_to_h5)(args, split, subset) for split, subset in subsets
      )

def get_names(fpath):
  names = []
  with open(fpath, 'r') as f:
    for line in f:
      new_line = line.rstrip()
      names.append(new_line)
  return names

def process_data(names, meta_data, top_dir, home_dir=None, relabel=False, min_seq_len=0):
  assert home_dir in ['bbox_train', 'bbox_test']
  num_tracklets = meta_data.shape[0]
  pid_list = list(set(meta_data[:,2].tolist()))
  num_pids = len(pid_list)

  if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
  tracklets = []

  for tracklet_idx in range(num_tracklets):
    data = meta_data[tracklet_idx,...]
    start_index, end_index, pid, camid = data
    if pid == -1:
      continue # junk images are just ignored
    assert 1 <= camid <= 6
    if relabel: pid = pid2label[pid]
    camid -= 1 # index starts from 0
    img_names = names[start_index - 1:end_index]

    # make sure image names correspond to the same person
    pnames = [img_name[:4] for img_name in img_names]
    assert len(set(pnames)) == 1, 'Error: a single tracklet contains different person images'

    # make sure all images are captured under the same camera
    camnames = [img_name[5] for img_name in img_names]
    assert len(set(camnames)) == 1, 'Error: images are captured under different cameras!'
    
    # get tracklet id and check sanity
    trackletid = [img_name[7:11] for img_name in img_names]
    assert len(set(camnames)) == 1, 'Error: images have different tracklet numbers!'
    trackletid = int(trackletid[0]) - 1 # index starts from 0

    # append image names with directory information
    img_paths = [os.path.join(top_dir, home_dir, img_name[:4], img_name) for img_name in img_names]
    if len(img_paths) >= min_seq_len:
      img_paths = tuple(img_paths)
      tracklets.append((img_paths, pid, camid, trackletid))

  return tracklets


def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

def timeSince(since):
  now = time.time()
  s = now - since
  return '%s' % (asMinutes(s))

def tracklets_to_h5(args, split, tracklets):
  dt = h5py.special_dtype(vlen=np.uint8)
  
  start_time = time.time()

  outfile_path = os.path.join(args.output_dir, 'mars_tracklets_{}.h5'.format(split))
  outfile = h5py.File(outfile_path, 'w')
  tracklets_group = outfile.create_group('tracklets')
  pid_groups = {}
  camid_groups = {}
  for tracklet in tracklets:
    img_paths = tracklet[0]
    pid = '{:04d}'.format(tracklet[1])
    camid = str(tracklet[2])
    trackletid = '{:04d}'.format(tracklet[3])
    if "/tracklets/{}".format(pid) not in outfile:
      pid_groups[pid] = tracklets_group.create_group(pid)
    if "/tracklets/{}/{}".format(pid, camid) not in outfile:
      if pid not in camid_groups:
        camid_groups[pid] = {}
      camid_groups[pid][camid] = pid_groups[pid].create_group(camid)
    dset = camid_groups[pid][camid].create_dataset(trackletid, 
      (len(img_paths),), maxshape=(len(img_paths),), chunks=True, dtype=dt)

    for f_ind, f in enumerate(img_paths):
      # read jpg as binary and put into h5
      jpg = open(f, 'rb')
      binary_data = jpg.read()
      dset[f_ind] = np.fromstring(binary_data, dtype=np.uint8)
      jpg.close()

  outfile.close()
  print('Converting jpgs of MARS {} set to h5 done...took {}'.format(
    split, timeSince(start_time)) )

def get_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

  # Input Directories and Files
  parser.add_argument('--top-dir', dest='top_dir',
    default='./dataset/mars/',
    help='Top Directory with subdirectories containing MARS.')

  # Output Directories
  parser.add_argument('--output-dir', dest='output_dir',
    default='./dataset/mars_h5/',
    help='Directory for outputs, extracted 3D flows in HDF5 format.')

  # Parallelism
  parser.add_argument('--num-worker', dest='num_worker',
    type=int, default=0, 
    help='Number of parallel jobs for resizing. 0 for no parallelism. '
         'Choose this number wisely for speed and I/O bottleneck tradeoff.')

  args = parser.parse_args()

  return args

if __name__ == "__main__":
  main()
