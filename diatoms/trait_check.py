"""Do counterfactuals move measurable diatom traits towards the target species?

Traits, measured automatically on a 256x256 native-size image (long axis vertical):
  length       vertical extent of the specimen mask (px)
  width        median horizontal extent of the mask over the central half of the valve (px)
  elongation   length / width
  stria_period dominant period (px) of the intensity profile along the valve's long
               axis (striae run across the valve), from the FFT of the detrended
               row-mean profile over the central 60% of the valve; NaN if no clear peak
Length and elongation are skipped for specimens cropped at 224 px (their true length
is unknown).

Reference distributions: up to --n_ref training images (random split) per species.

Tests per trait, over counterfactuals x (source A -> target B):
  direction    among pairs where B's mean differs from the original by > 1 SD of B,
               does the counterfactual move the trait towards B's mean? (chance 50%)
  specificity  for the two counterfactuals of one source (targets B1, B2) where
               B1 and B2 differ by > 1 pooled SD: do the two counterfactuals differ
               in the same direction as B1 and B2? (chance 50%)
Also reports measurement reliability: the share of trait variance explained by
species in the reference images (higher = the trait is a real species property
under this measurement).

Example:
  python diatoms/trait_check.py --run full:<wid_0 pth>,<wid_1 pth> --out_dir results/traits
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_metrics import diatom_mask  # noqa: E402
from train_classifier import DATA_NATIVE  # noqa: E402

TRAITS = ['length', 'width', 'elongation', 'stria_period']


def measure(gray, cropped=False):
    """gray: (H, W) float in [0, 1]. Returns dict of traits (NaN when not measurable)."""
    m = diatom_mask(np.repeat(gray[..., None], 3, 2))
    rows = np.where(m.any(1))[0]
    out = dict.fromkeys(TRAITS, np.nan)
    if len(rows) < 10:
        return out
    top, bot = rows[0], rows[-1]
    length = bot - top + 1
    mid = slice(top + length // 4, bot - length // 4 + 1)
    widths = m[mid].sum(1)
    width = float(np.median(widths[widths > 0])) if (widths > 0).any() else np.nan
    if not cropped:
        out['length'] = float(length)
        out['elongation'] = float(length / width) if width else np.nan
    out['width'] = width
    # stria period: row-mean intensity inside the (eroded) mask over the central 60% of the valve
    inner = ndimage.binary_erosion(m, iterations=4)
    r0, r1 = top + int(0.2 * length), bot - int(0.2 * length)
    prof = np.array([gray[r][inner[r]].mean() if inner[r].sum() > 3 else np.nan for r in range(r0, r1)])
    if len(prof) >= 32 and np.isfinite(prof).mean() > 0.9:
        prof = np.nan_to_num(prof, nan=np.nanmean(prof))
        prof = prof - ndimage.gaussian_filter1d(prof, 6)
        prof = prof * np.hanning(len(prof))
        spec = np.abs(np.fft.rfft(prof, n=512)) ** 2
        freqs = np.fft.rfftfreq(512)
        band = (freqs >= 1 / 20) & (freqs <= 1 / 2.5)       # periods 2.5-20 px
        if band.any():
            k = np.argmax(np.where(band, spec, 0))
            period = 1 / freqs[k]
            # clear peak only, and not at the edge of the search band (edge maxima are
            # mostly smooth valves or a slow trend, checked visually)
            if spec[k] > 4 * np.median(spec[band]) and 2.7 < period < 18:
                out['stria_period'] = float(period)
    return out


def load_run(spec):
    pths = spec.split(',')
    ds = [torch.load(p, map_location='cpu') for p in pths]
    gt = torch.cat([d['gt_imgs'].float() for d in ds]).mean(1).numpy()
    cf = torch.cat([d['counterfactuals'][:, 0, 0].float().clamp(0, 1) for d in ds]).mean(1).numpy()
    y = torch.cat([d['gt_labels'][:len(d['gt_imgs'])] for d in ds]).numpy()
    t = torch.cat([d['targets'] for d in ds]).numpy()
    return gt, cf, y, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, help='name:pth,pth')
    ap.add_argument('--ckpt', default='DiatlasModels/r50madry_random_eps1.0/best.pt', help='for class names')
    ap.add_argument('--n_ref', type=int, default=40)
    ap.add_argument('--sd_thresh', type=float, default=1.0)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    classes = torch.load(args.ckpt, map_location='cpu')['classes']

    runs = {n: load_run(p) for n, p in (r.split(':', 1) for r in args.run)}
    species = {classes[i] for _, (_, _, y, t) in runs.items() for i in np.concatenate([y, t])}

    with open(os.path.join(DATA_NATIVE, 'splits.csv')) as f:
        cropped_of = {r['path']: r['cropped'] == '1' for r in csv.DictReader(f)}
    with open(os.path.join(DATA_NATIVE, 'splits_random.csv')) as f:
        pool = collections.defaultdict(list)
        for r in csv.DictReader(f):
            if r['split_random'] == 'train' and r['species_code'] in species:
                pool[r['species_code']].append(r['path'])
    rng = np.random.default_rng(0)
    ref = {}
    ref_rows = []
    for sp in sorted(species):
        paths = pool[sp]
        pick = rng.choice(len(paths), min(args.n_ref, len(paths)), replace=False) if paths else []
        vals = []
        for i in pick:
            pth = paths[i]
            g = np.asarray(Image.open(os.path.join(DATA_NATIVE, pth)).convert('L'), dtype=np.float32) / 255
            v = measure(g, cropped_of.get(pth, False))
            vals.append(v)
            ref_rows.append(dict(species=sp, path=pth, **v))
        ref[sp] = {k: np.array([v[k] for v in vals], dtype=float) for k in TRAITS}
    with open(os.path.join(args.out_dir, 'reference_traits.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['species', 'path'] + TRAITS)
        w.writeheader()
        w.writerows(ref_rows)

    lines = ['MEASUREMENT RELIABILITY (reference training images)',
             f'{"trait":14s} {"measured":>9s} {"species R2":>11s}  (share of variance explained by species)']
    for k in TRAITS:
        groups = [v[k][np.isfinite(v[k])] for v in ref.values()]
        groups = [g for g in groups if len(g) >= 3]
        allv = np.concatenate(groups)
        ss_tot = ((allv - allv.mean()) ** 2).sum()
        ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
        n_meas = sum(np.isfinite(v[k]).sum() for v in ref.values())
        n_all = sum(len(v[k]) for v in ref.values())
        lines.append(f'{k:14s} {n_meas / n_all:9.0%} {ss_b / ss_tot:11.2f}')
    lines.append('')

    def stats(sp, k):
        v = ref[sp][k]
        v = v[np.isfinite(v)]
        return (v.mean(), v.std(ddof=1)) if len(v) >= 3 else (np.nan, np.nan)

    for name, (gt, cf, y, t) in runs.items():
        n = len(t)
        # sources come in pairs (two targets); cropped status of the source from its length
        rows = []
        for i in range(n):
            crop = False
            mo = measure(gt[i])
            if np.isfinite(mo['length']) and mo['length'] >= 222:
                crop = True
            mo = measure(gt[i], crop)
            mc = measure(cf[i], crop)
            rows.append(dict(idx=i, source=classes[y[i]], target=classes[t[i]], cropped=int(crop),
                             **{f'{k}_orig': mo[k] for k in TRAITS}, **{f'{k}_cf': mc[k] for k in TRAITS}))
        with open(os.path.join(args.out_dir, f'traits_{name}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        lines.append(f'RUN {name}: {n} counterfactuals')
        lines.append(f'{"trait":14s} {"direction: towards target":>30s} {"specificity: B1 vs B2":>28s}')
        for k in TRAITS:
            hits = tot = 0
            dir_hit, dir_src, gap = [], [], []       # per counted counterfactual
            for r in rows:
                o, c = r[f'{k}_orig'], r[f'{k}_cf']
                mu, sd = stats(r['target'], k)
                if not np.isfinite([o, c, mu, sd]).all() or sd == 0 or abs(mu - o) <= args.sd_thresh * sd:
                    continue
                tot += 1
                hits += int(np.sign(c - o) == np.sign(mu - o))
                dir_hit.append(int(np.sign(c - o) == np.sign(mu - o)))
                dir_src.append(r['idx'] // 2)
                gap.append((c - o) / (mu - o))       # 0 = unchanged, 1 = reaches the target mean
            sh = st = 0
            for i in range(0, n - 1, 2):
                r1, r2 = rows[i], rows[i + 1]
                (m1, s1), (m2, s2) = stats(r1['target'], k), stats(r2['target'], k)
                c1, c2 = r1[f'{k}_cf'], r2[f'{k}_cf']
                if not np.isfinite([m1, m2, s1, s2, c1, c2]).all():
                    continue
                if abs(m1 - m2) <= args.sd_thresh * np.sqrt((s1 ** 2 + s2 ** 2) / 2):
                    continue
                st += 1
                sh += int(np.sign(c1 - c2) == np.sign(m1 - m2))
            fmt = lambda h, n_: (f'{h / n_:6.1%} ({h}/{n_}, p={binomtest(h, n_, 0.5).pvalue:.1e})' if n_ else '      n/a')
            lines.append(f'{k:14s} {fmt(hits, tot):>30s} {fmt(sh, st):>28s}')
            if dir_hit:
                # cluster bootstrap over source specimens (their two counterfactuals are not independent)
                dh, ds, gp = np.array(dir_hit), np.array(dir_src), np.array(gap)
                srcs = np.unique(ds)
                idx_of = {sv: np.where(ds == sv)[0] for sv in srcs}
                brng = np.random.default_rng(0)
                boot_rate, boot_gap = [], []
                for _ in range(2000):
                    take = np.concatenate([idx_of[sv] for sv in brng.choice(srcs, len(srcs))])
                    boot_rate.append(dh[take].mean())
                    boot_gap.append(np.median(gp[take]))
                lo, hi = np.quantile(boot_rate, [0.025, 0.975])
                glo, ghi = np.quantile(boot_gap, [0.025, 0.975])
                lines.append(f'{"":14s}   direction 95% CI (cluster bootstrap over {len(srcs)} sources): [{lo:.1%}, {hi:.1%}];'
                             f' median share of gap to target mean closed {np.median(gp):.2f} [{glo:.2f}, {ghi:.2f}]')
        lines.append('')
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
