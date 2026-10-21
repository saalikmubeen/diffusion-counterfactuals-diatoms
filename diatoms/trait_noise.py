"""Measurement-noise floor for the trait check.

Each original specimen is re-measured after small perturbations that should not change
its traits (shift by up to 2 px, rotation by up to 2 degrees, brightness +-3%, Gaussian
noise sigma 0.01). The spread of these measurements is the measurement noise of each
trait. Counterfactual trait changes are then compared with it: what share exceeds two
noise standard deviations, and does the direction test still hold for those changes?

  python diatoms/trait_noise.py --run full:<wid_0 pth>,<wid_1 pth> --out results/traits/noise_floor.txt
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trait_check import TRAITS, load_run, measure  # noqa: E402


def jitter(g, rng):
    bg = np.median(np.concatenate([g[:4].ravel(), g[-4:].ravel()]))
    out = ndimage.rotate(g, rng.uniform(-2, 2), reshape=False, order=1, mode='constant', cval=bg)
    out = ndimage.shift(out, rng.uniform(-2, 2, 2), order=1, mode='constant', cval=bg)
    out = out * rng.uniform(0.97, 1.03) + rng.normal(0, 0.01, out.shape)
    return np.clip(out, 0, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='name:pth,pth')
    ap.add_argument('--ckpt', default='DiatlasModels/r50madry_random_eps1.0/best.pt')
    ap.add_argument('--n_jitter', type=int, default=8)
    ap.add_argument('--ref', default='results/traits/reference_traits.csv', help='reference trait table from trait_check.py')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import pandas as pd
    name, p = args.run.split(':', 1)
    gt, cf, y, t = load_run(p)
    classes = torch.load(args.ckpt, map_location='cpu')['classes']
    ref = pd.read_csv(args.ref, keep_default_na=False, na_values=['', 'nan'])
    stats = {(sp, k): (g[k].dropna().mean(), g[k].dropna().std()) for sp, g in ref.groupby('species') for k in TRAITS}
    rng = np.random.default_rng(0)
    n = len(t)

    noise_sd = {k: [] for k in TRAITS}                       # per source, relative SD
    per_src_sd = []
    for s in range(0, n, 2):
        g = gt[s]
        m0 = measure(g)
        crop = np.isfinite(m0['length']) and m0['length'] >= 222
        vals = [measure(jitter(g, rng), crop) for _ in range(args.n_jitter)]
        sd = {}
        for k in TRAITS:
            v = np.array([x[k] for x in vals], float)
            v = v[np.isfinite(v)]
            sd[k] = v.std(ddof=1) if len(v) >= 4 else np.nan
            mu = v.mean() if len(v) else np.nan
            if np.isfinite(sd[k]) and mu:
                noise_sd[k].append(sd[k] / mu)
        per_src_sd.append((crop, sd))

    lines = [f'{name}: {n} counterfactuals; measurement noise from {args.n_jitter} jittered copies of each of {n // 2} originals', '',
             f'{"trait":14s} {"noise (rel. SD)":>16s} {"cf changes > 2 SD":>18s} {"direction, changes > 2 SD":>30s}']
    for k in TRAITS:
        big = tot = hits = n_dir = 0
        for i in range(n):
            crop, sd = per_src_sd[i // 2]
            o, c = measure(gt[i], crop)[k], measure(cf[i], crop)[k]
            if not np.isfinite([o, c, sd[k]]).all():
                continue
            tot += 1
            if abs(c - o) > 2 * sd[k]:
                big += 1
                mu, sdb = stats.get((classes[t[i]], k), (np.nan, np.nan))
                if np.isfinite([mu, sdb]).all() and abs(mu - o) > sdb:
                    n_dir += 1
                    hits += int(np.sign(c - o) == np.sign(mu - o))
        d = f'{hits / n_dir:.0%} ({hits}/{n_dir}, p={binomtest(hits, n_dir, .5).pvalue:.1e})' if n_dir else 'n/a'
        lines.append(f'{k:14s} {np.median(noise_sd[k]):15.1%} {big / max(tot, 1):17.0%} {d:>30s}')
    lines += ['', 'noise: median over specimens of SD/mean across jittered copies; "> 2 SD": |change| larger than twice',
              "that specimen's measurement SD; direction: same test as trait_check.py, restricted to those changes."]
    txt = '\n'.join(lines)
    print(txt)
    open(args.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
