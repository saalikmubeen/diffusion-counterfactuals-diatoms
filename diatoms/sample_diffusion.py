"""Check a diffusion checkpoint on DIATLAS (plan section 3, "Done when").

Two modes, both writing a PNG grid:
  sample  unconditional samples from pure noise
  recon   take held-out test images, noise them to the DVCE start point
          (timestep_respacing=200, skip_timesteps=100, as in configs/default.yml)
          and denoise without classifier guidance. Grid rows alternate
          original / reconstruction. A good prior keeps the diatom's identity.

Examples:
  python diatoms/sample_diffusion.py --ckpt checkpoints/256x256_diffusion_uncond.pt --mode recon --out recon_imagenet.png
  python diatoms/sample_diffusion.py --ckpt checkpoints/diatlas/<run>/ema_0.9995_030000.pt --mode sample --out samples.png --save_individual
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
import torch as th
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_diffusion import REPO, model_config  # noqa: E402
from guided_diffusion.script_util import create_model_and_diffusion  # noqa: E402


def load_test_images(data_root, n, seed):
    with open(os.path.join(data_root, 'splits.csv')) as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'test']
    rows = random.Random(seed).sample(rows, n)
    imgs = [np.asarray(Image.open(os.path.join(data_root, r['path'])).convert('RGB')) for r in rows]
    x = th.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 127.5 - 1
    return x, [r['taxon'] for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--mode', choices=['sample', 'recon'], default='sample')
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--timestep_respacing', default='200')
    p.add_argument('--skip_timesteps', type=int, default=100, help='recon only')
    p.add_argument('--data_root', default=os.path.join(REPO, 'data', 'diatlas'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    p.add_argument('--save_individual', action='store_true',
                   help='also save every generated/reconstructed image as its own PNG in <out without .png>/')
    args = p.parse_args()

    cfg = model_config(use_fp16=True, use_checkpoint=False)
    cfg['timestep_respacing'] = args.timestep_respacing
    model, diffusion = create_model_and_diffusion(**cfg)
    model.load_state_dict(th.load(args.ckpt, map_location='cpu'))
    model.convert_to_fp16()
    model.cuda().eval()

    if args.mode == 'recon':
        x0, taxa = load_test_images(args.data_root, args.n, args.seed)
        print('test taxa:', ' '.join(taxa))
        skip = args.skip_timesteps
    else:
        x0, skip = th.zeros(args.n, 3, 256, 256), 0

    # The vendored p_sample/p_sample_loop crash without a cond_fn (factor_ is
    # unset) and print debug output every step, so run the same ancestral loop
    # here: noise x0 to step (T - skip - 1), then denoise down to 0.
    gen = th.Generator(device='cuda').manual_seed(args.seed)
    steps = list(range(diffusion.num_timesteps - skip))[::-1]
    outs = []
    with th.no_grad():
        for i in range(0, args.n, args.batch_size):
            x0_b = x0[i:i + args.batch_size].cuda()
            t = th.full((x0_b.shape[0],), steps[0], dtype=th.long, device='cuda')
            img = diffusion.q_sample(x0_b, t, noise=th.randn(x0_b.shape, device='cuda', generator=gen))
            for step in tqdm(steps, desc=f'batch {i // args.batch_size}'):
                t = th.full((x0_b.shape[0],), step, dtype=th.long, device='cuda')
                out = diffusion.p_mean_variance(model, img, t, clip_denoised=True)
                img = out['mean']
                if step > 0:
                    img = img + th.exp(0.5 * out['log_variance']) * th.randn(img.shape, device='cuda', generator=gen)
            outs.append(img.float().cpu())
    out = th.cat(outs)

    if args.mode == 'recon':
        cols = 8
        rows = []
        for i in range(0, args.n, cols):
            rows += [x0[i:i + cols], out[i:i + cols]]
        grid, nrow = th.cat(rows), cols
    else:
        grid, nrow = out, 8
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_image((grid + 1) / 2, args.out, nrow=nrow, padding=2, pad_value=1)
    print('saved', args.out)
    if args.save_individual:
        folder = os.path.splitext(args.out)[0]
        os.makedirs(folder, exist_ok=True)
        for i, img in enumerate(out):
            name = f'{i:02d}_{taxa[i]}_recon.png' if args.mode == 'recon' else f'{i:02d}.png'
            save_image((img + 1) / 2, os.path.join(folder, name))
        print(f'saved {len(out)} individual images in {folder}/')


if __name__ == '__main__':
    main()
