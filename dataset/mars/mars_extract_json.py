#!/usr/bin/env python3
import os
import h5py
import json
import argparse

def main():
  args = get_args()

  splits = ['train', 'test_query', 'test_gallery']

  output = {}
  for split in splits:
    output[split] = extract_length(
      os.path.join(args.h5_dir, 'mars_tracklets_{}.h5'.format(split))
      )

  with open(args.output_json_path, 'w', encoding="utf-8") as fp:
    json.dump(output, fp, indent=args.json_indent)

def extract_length(h5_path):
  tracklets = {}
  h5 = h5py.File(h5_path, 'r')
  for person_id in h5['tracklets']:
    for cam_id in h5['tracklets'][person_id]:
      for track_id in h5['tracklets'][person_id][cam_id]:
        path = '/{}/{}/{}'.format(person_id,cam_id,track_id)
        tracklets[path] = {}
        tracklets[path]['person_id'] = int(person_id)
        tracklets[path]['camera_id'] = int(cam_id)
        tracklets[path]['track_id'] = int(track_id)
        tracklets[path]['length'] = len(h5['tracklets'][person_id][cam_id][track_id])
  h5.close()

  return tracklets


def get_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

  # Input Directories and Files
  parser.add_argument('-d', '--h5-dir', dest='h5_dir', required=True,
    help='Path to directory with input H5 files. H5 files must end with '
          '\'_dataset.h5\'')

  # Parameters
  parser.add_argument('-i', '--json-indent', dest='json_indent',
    default=None, type=int,
    help='If None, generates JSON with no identation. If integer >0, json will '
         'be indented')

  # Output Directories and Files
  parser.add_argument('-o', '--output-json-path', dest='output_json_path',
    default='./mars.json',
    help='Path to dataset JSON directory')


  args = parser.parse_args()
  
  return args

if __name__ == "__main__":
  main()
