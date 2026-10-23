"""Counterfactuals towards an expert's look-alike species vs. a random species of the same genus.

Input: the DVCE run on evaluation/vce_pairs_expert.json (diatoms/select_pairs_expert.py): each
source has target 1 = expert look-alike, target 2 = random same-genus species. Reports, for the
two kinds of target (paired over the same sources): target reached, LPIPS, held-out-page
transfer, how likely the classifier already found the target on the original, and trait
direction; per expert pair as well. Also how often the classifier confuses each pair on all
its test images. Writes one figure sheet per pair for the expert:
  original (A) | counterfactual -> look-alike (B) | 2 real photos of A | 2 real photos of B

  python diatoms/expert_pairs_analysis.py --pth <wid_0 pth> <wid_1 pth> --out_dir results/dvce/expert_pairs
"""
import argparse
import collections
import csv
import json
import os
import random
import sys

import lpips
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import binomtest, wilcoxon  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diatom_classifier import load_diatom_classifier  # noqa: E402
from train_classifier import DATA_NATIVE  # noqa: E402
from trait_check import TRAITS, measure  # noqa: E402


def mcnemar(a, b):
    only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return 1.0 if only_a + only_b == 0 else binomtest(only_a, only_a + only_b, .5).pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pth', nargs='+', required=True)
    ap.add_argument('--pairs', default='evaluation/vce_pairs_expert.json')
    ap.add_argument('--ckpt', default='DiatlasModels/r50madry_random_eps1.0/best.pt')
    ap.add_argument('--eval_ckpt', default='DiatlasModels/r50imagenet_page_eps0/best.pt')
    ap.add_argument('--taxa', default='/mnt/diatoms/datasets/diatlas/taxonomic_code.csv')
    ap.add_argument('--n_ref', type=int, default=40)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = [torch.load(p, map_location='cpu') for p in args.pth]
    x = torch.cat([d['gt_imgs'].float() for d in ds])
    cf = torch.cat([d['counterfactuals'][:, 0, 0].float().clamp(0, 1) for d in ds])
    t = torch.cat([d['targets'] for d in ds]).numpy()
    p_in = torch.cat([d['in_probabilities'][:, 0, 0] for d in ds])
    p_out = torch.cat([d['out_probabilities'][:, 0, 0] for d in ds])
    clf, classes, _ = load_diatom_classifier(args.ckpt)
    idx = {c: i for i, c in enumerate(classes)}
    n = len(t)

    entries = json.load(open(args.pairs))['entries']
    files = {e['path']: np.asarray(Image.open(os.path.join(DATA_NATIVE, e['path'])).convert('L'), np.float32) / 255
             for e in entries}
    by_path = {e['path']: e for e in entries}
    kind, pair, src_path = [], [], []
    for k in range(n):
        g = x[k].mean(0).numpy()
        pth = min(files, key=lambda q: np.abs(files[q] - g).mean())
        e = by_path[pth]
        j = e['targets'].index(classes[t[k]])
        kind.append(e['target_kinds'][j]); pair.append(e['expert_pair']); src_path.append(pth)
    kind, pair = np.array(kind), np.array(pair)
    names = {r['taxonomic_code']: (r['infra_species'] or r['species'] or r['genus'])
             for r in csv.DictReader(open(args.taxa), delimiter=';')}

    lp = lpips.LPIPS(net='alex', verbose=False).eval()
    with torch.no_grad():
        d_lp = torch.cat([lp(x[i:i + 20] * 2 - 1, cf[i:i + 20] * 2 - 1).flatten() for i in range(0, n, 20)]).numpy()
    ev, _, _ = load_diatom_classifier(args.eval_ckpt)
    with torch.no_grad():
        pe = torch.cat([ev(cf[i:i + 50]).softmax(1) for i in range(0, n, 50)])
    reached = (p_out.argmax(1).numpy() == t).astype(int)
    transfer = (pe.argmax(1).numpy() == t).astype(int)
    p_before = p_in[range(n), t].numpy()

    # traits: direction towards the target species' mean (references: training photos, seeded per species)
    with open(os.path.join(DATA_NATIVE, 'splits_random.csv')) as f:
        rows = list(csv.DictReader(f))
    pool = collections.defaultdict(list)
    for r in rows:
        if r['split_random'] == 'train':
            pool[r['species_code']].append(r['path'])
    ref_cache = {}

    def ref_stats(sp):
        if sp not in ref_cache:
            import zlib
            paths = pool[sp]
            pick = np.random.default_rng(zlib.crc32(sp.encode())).choice(len(paths), min(args.n_ref, len(paths)), replace=False)
            vals = [measure(np.asarray(Image.open(os.path.join(DATA_NATIVE, paths[i])).convert('L'), np.float32) / 255) for i in pick]
            ref_cache[sp] = {k: (np.nanmean([v[k] for v in vals]), np.nanstd([v[k] for v in vals], ddof=1)) for k in TRAITS}
        return ref_cache[sp]
    hits, tot = np.zeros(n), np.zeros(n)
    for k in range(n):
        mo = measure(x[k].mean(0).numpy())
        crop = np.isfinite(mo['length']) and mo['length'] >= 222
        mo, mc = measure(x[k].mean(0).numpy(), crop), measure(cf[k].mean(0).numpy(), crop)
        st = ref_stats(classes[t[k]])
        for tr in TRAITS:
            mu, sd = st[tr]
            if np.isfinite([mo[tr], mc[tr], mu, sd]).all() and sd > 0 and abs(mu - mo[tr]) > sd:
                tot[k] += 1
                hits[k] += np.sign(mc[tr] - mo[tr]) == np.sign(mu - mo[tr])

    E, R = kind == 'expert_look_alike', kind == 'random_same_genus'
    L = [f'{E.sum()} counterfactuals towards the expert look-alike, {R.sum()} towards a random same-genus species '
         f'(same {E.sum()} sources)', '',
         f'{"":26s} {"look-alike":>11s} {"random":>8s}  paired test']
    L.append(f'{"classifier p(target) on original":26s} {p_before[E].mean():11.3f} {p_before[R].mean():8.3f}  '
             f'Wilcoxon p = {wilcoxon(p_before[E], p_before[R]).pvalue:.2g}')
    L.append(f'{"target reached":26s} {reached[E].mean():11.1%} {reached[R].mean():8.1%}  McNemar p = {mcnemar(reached[E], reached[R]):.2g}')
    L.append(f'{"LPIPS (edit size)":26s} {d_lp[E].mean():11.3f} {d_lp[R].mean():8.3f}  Wilcoxon p = {wilcoxon(d_lp[E], d_lp[R]).pvalue:.2g}')
    L.append(f'{"held-out-page transfer":26s} {transfer[E].mean():11.1%} {transfer[R].mean():8.1%}  McNemar p = {mcnemar(transfer[E], transfer[R]):.2g}')
    L.append(f'{"traits toward target":26s} {hits[E].sum() / max(tot[E].sum(), 1):11.1%} {hits[R].sum() / max(tot[R].sum(), 1):8.1%}  '
             f'(n traits {int(tot[E].sum())} / {int(tot[R].sum())})')
    L += ['', 'PER EXPERT PAIR (counterfactuals towards the look-alike)',
          f'{"pair":14s} {"n":>3s} {"reached":>8s} {"LPIPS":>6s} {"transfer":>9s} {"p(B) before":>12s} {"clf confusion A->B on test":>28s}']

    # how often the classifier confuses the pair on all test images (random split)
    test_rows = [r for r in rows if r['split_random'] == 'test']
    for pr in sorted(set(pair)):
        a, b = pr.split('<->')
        m = E & (pair == pr)
        conf = []
        for s1, s2 in ((a, b), (b, a)):
            paths = [r['path'] for r in test_rows if r['species_code'] == s1]
            if not paths:
                continue
            imgs = torch.stack([torch.from_numpy(np.asarray(Image.open(os.path.join(DATA_NATIVE, q)).convert('RGB'), np.float32) / 255).permute(2, 0, 1) for q in paths])
            with torch.no_grad():
                pred = clf(imgs).argmax(1).numpy()
            conf.append(f'{s1}->{s2} {np.mean(pred == idx[s2]):.0%} of {len(paths)}')
        L.append(f'{pr:14s} {m.sum():3d} {reached[m].mean():8.1%} {d_lp[m].mean():6.3f} {transfer[m].mean():9.1%} '
                 f'{p_before[m].mean():12.3f}   ' + '; '.join(conf))
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(args.out_dir, 'summary.txt'), 'w').write(txt + '\n')

    # figure sheets for the expert, one per pair
    rng = random.Random(0)
    refs = lambda sp, k: [np.asarray(Image.open(os.path.join(DATA_NATIVE, q)).convert('L')) for q in rng.sample(sorted(pool[sp]), min(k, len(pool[sp])))]
    for pr in sorted(set(pair)):
        ks = [k for k in range(n) if E[k] and pair[k] == pr]
        fig, ax = plt.subplots(len(ks), 6, figsize=(6 * 2.2, len(ks) * 2.5), squeeze=False)
        for row, k in enumerate(ks):
            a, b = classes[int(torch.cat([d['gt_labels'][:len(d['gt_imgs'])] for d in ds])[k])], classes[t[k]]
            panels = [(x[k].mean(0).numpy(), f'original\n{names.get(a, a)}'),
                      (cf[k].mean(0).numpy(), f'counterfactual ->\n{names.get(b, b)}')]
            panels += [(r_ / 255, f'real {names.get(a, a)}') for r_ in refs(a, 2)]
            panels += [(r_ / 255, f'real {names.get(b, b)}') for r_ in refs(b, 2)]
            for c, (im, title) in enumerate(panels):
                ax[row, c].imshow(im, cmap='gray', vmin=0, vmax=1)
                ax[row, c].set_title(title, fontsize=7)
            for a_ in ax[row]:
                a_.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'pair_{pr.replace("<->", "_")}.png'), dpi=110)
        plt.close(fig)
    print('figure sheets:', args.out_dir)


if __name__ == '__main__':
    main()
