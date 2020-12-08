#!/usr/bin/env python3
from run.read import main, ALL_MODELS
from utils.args import get_args

def play_main():
  args = get_args(ALL_MODELS)

  try:
    main(args)
  except Exception as e:
    raise

if __name__ == "__main__":
  play_main()