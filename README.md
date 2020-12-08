# READ: Reciprocal Attention Discriminator for Image-to-Video Re-Identification
#### [Minho Shim](https://research.minhoshim.com), [Hsuan-I Ho](https://azuxmioy.github.io), Jinhyung Kim, and Dongyoon Wee

This repository contains demo software for the READ (ECCV 2020).

[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590324.pdf)] [[Supp](https://drive.google.com/file/d/1S8u7qzzZz6STP0U6Rx5qLvcJ5pwUwswP/view?usp=sharing)]

### Environment
- CUDA 9.0
- Python 3.5.2
- PyTorch 0.4.1
- OpenCV 3.4.7
- H5py
- SciPy

### Dataset Preparation (MARS)
Unzip `bbox_train.zip` and `bbox_test.zip`, alongside with `mat`/`txt` files.
(Links: [zips](https://drive.google.com/drive/u/0/folders/0B6tjyrV1YrHeMVV2UFFXQld6X1E), [mats/txts](https://github.com/liangzheng06/MARS-evaluation/tree/6477ae919dc3a48ec6e879a548b9940602b78f26/info) from the MARS authors' [repo](https://github.com/liangzheng06/MARS-evaluation))

The directory will look like:
```
path/to/mars_raw/
|-- bbox_train/
|-- bbox_test/
|-- info/
```

Then run:
```bash
python dataset/mars/mars_jpgs_to_h5.py --top-dir path/to/mars_raw/ --output-dir path/to/mars_h5/
python dataset/mars/mars_extract_json.py -d path/to/mars_h5/ -o path/to/mars_h5/mars.min.json
```

### Download weights
Download links to weight files for [MARS](https://drive.google.com/file/d/1TMWGFzz-O64UfN9PxGUfYJiQ0Y5z02n-/view?usp=sharing), and [DukeMTMC-VideoReID](https://drive.google.com/file/d/1LjOBvgBBOoceHV590ttFYH-Kb2rnqWrT/view?usp=sharing).

### Run (MARS)
```bash
python play.py --test --dataset mars --json-path path/to/mars_h5/mars.min.json --h5-dir path/to/mars_h5/ --checkpoint-files '{"all":"[downloaded_checkpoint_path]"}'
```
By default, it uses all available GPUs in the machine. 
It could be limited with `CUDA_VISIBLE_DEVICES`, *e.g.* `CUDA_VISIBLE_DEVICES=0,1,2,3 python ...`

Adjust `--batch-size` accordingly to your GPU capacity. In our case, P40 or V100 GPUs are used with the default batch size is 32.

### Sample Results (MARS)
```bash
python play.py --test --dataset mars --checkpoint-files '{"all":"/volume/READ/mars_demo.tar"}' --json-path /volume/dataset/mars_h5/mars.min.json --h5-dir /volume/mino/dataset/mars_h5/

...

Test Results:
mean_average_precision: 0.7039292454719543
mean_top10: 0.9383838176727295
mean_top5: 0.9207070469856262
mean_top1: 0.8146464824676514
```

### Citation
```
@InProceedings{shim2020read,
author = {Shim, Minho and Ho, Hsuan-I and Kim, Jinhyung and Wee, Dongyoon},
title = {{READ}: {R}eciprocal Attention Discriminator for Image-to-Video Re-Identification},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2020}
}
```

### Lincense and References

This software is for non-commercial use only.
Imported/modified codes contain their own references inside each code.
