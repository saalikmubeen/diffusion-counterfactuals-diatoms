"""Sanity checks before diffusion training. CPU only, nothing is trained.

  * the model built with the DVCE flags loads the ImageNet checkpoint strictly
  * the DIATLAS train+val loader works and contains no test images

Run:  CUDA_VISIBLE_DEVICES= python diatoms/check_setup.py
"""
import os
import sys

import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_diffusion import REPO, DiatlasDataset, model_config  # noqa: E402
from guided_diffusion.script_util import create_model_and_diffusion  # noqa: E402

ckpt = os.path.join(REPO, 'checkpoints', '256x256_diffusion_uncond.pt')
model, _ = create_model_and_diffusion(**model_config(use_fp16=True, use_checkpoint=True))
model.load_state_dict(th.load(ckpt, map_location='cpu'), strict=True)
print(f'checkpoint loads strictly: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params')

ds = DiatlasDataset(os.path.join(REPO, 'data', 'diatlas'), ['train', 'val'], shard=0, num_shards=1)
x, _ = ds[0]
print(f'train+val images: {len(ds)}, sample {tuple(x.shape)} in [{x.min():.2f}, {x.max():.2f}]')
assert not any('/test/' in p for p in ds.paths), 'test images in training set!'
print('no test images in training set')
