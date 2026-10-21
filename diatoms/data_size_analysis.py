"""Does counterfactual quality depend on how many training images a species has?

For every counterfactual of a DVCE run, joins the number of training images (random split,
the explained classifier's training set) of its source and target species with its quality:
  reached       explained classifier predicts the target
  lpips         distance to the original (lower = smaller edit)
  locality      mean change on the specimen / in the background (224 crop)
  transfer      the held-out-page classifier predicts the target
  traits        share of measured traits (length, width, elongation, stria period) that move
                towards the target species' mean, among traits where the target differs from
                the original by > 1 SD (same rule as trait_check.py)
Reports quality per bin of target-species training size, Spearman correlations with
log(training size), and a lowest-vs-highest quartile comparison.

  python diatoms/data_size_analysis.py --pth <wid_0 pth> <wid_1 pth> \
      --traits results/traits/traits_full.csv --out results/dvce/full_r50eps1_ftv2/data_size.txt
"""
import argparse
import collections
import csv
import os
import sys

import lpips
import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diatom_classifier import load_diatom_classifier  # noqa: E402
from recon_metrics import diatom_mask  # noqa: E402
from train_classifier import DATA_NATIVE  # noqa: E402

TRAITS = ['length', 'width', 'elongation', 'stria_period']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pth', nargs='+', required=True)
    ap.add_argument('--ckpt', default='DiatlasModels/r50madry_random_eps1.0/best.pt')
    ap.add_argument('--eval_ckpt', default='DiatlasModels/r50imagenet_page_eps0/best.pt')
    ap.add_argument('--traits', required=True, help='per-counterfactual trait table from trait_check.py (same run)')
    ap.add_argument('--ref', default='results/traits/reference_traits.csv')
    ap.add_argument('--bins', type=int, nargs='+', default=[20, 40, 80, 150, 10 ** 6])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ds = [torch.load(p, map_location='cpu') for p in args.pth]
    x = torch.cat([d['gt_imgs'].float() for d in ds])
    cf = torch.cat([d['counterfactuals'][:, 0, 0].float().clamp(0, 1) for d in ds])
    y = torch.cat([d['gt_labels'][:len(d['gt_imgs'])] for d in ds]).numpy()
    t = torch.cat([d['targets'] for d in ds]).numpy()
    p_out = torch.cat([d['out_probabilities'][:, 0, 0] for d in ds])
    classes = torch.load(args.ckpt, map_location='cpu')['classes']
    n = len(t)

    with open(os.path.join(DATA_NATIVE, 'splits_random.csv')) as f:
        n_train = collections.Counter(r['species_code'] for r in csv.DictReader(f) if r['split_random'] == 'train')

    lp = lpips.LPIPS(net='alex', verbose=False).eval()
    with torch.no_grad():
        d_lp = torch.cat([lp(x[i:i + 20] * 2 - 1, cf[i:i + 20] * 2 - 1).flatten() for i in range(0, n, 20)]).numpy()
    crop = np.zeros((256, 256), bool)
    crop[16:240, 16:240] = True
    diff = (cf - x).abs().mean(1).numpy()
    loc_in, loc_out = [], []
    for i in range(n):
        m = diatom_mask(x[i].permute(1, 2, 0).numpy())
        loc_in.append(diff[i][m].mean())
        loc_out.append(diff[i][~m & crop].mean())
    clf, _, _ = load_diatom_classifier(args.eval_ckpt)
    with torch.no_grad():
        pe = torch.cat([clf(cf[i:i + 50]).softmax(1) for i in range(0, n, 50)])

    tr = pd.read_csv(args.traits, keep_default_na=False, na_values=['', 'nan'])
    ref = pd.read_csv(args.ref, keep_default_na=False, na_values=['', 'nan'])
    st = {(sp, k): (g[k].dropna().mean(), g[k].dropna().std()) for sp, g in ref.groupby('species') for k in TRAITS}
    trait_hits, trait_tot = np.zeros(n), np.zeros(n)
    for _, r in tr.iterrows():
        i = int(r['idx'])
        for k in TRAITS:
            o, c = r[f'{k}_orig'], r[f'{k}_cf']
            mu, sd = st.get((r['target'], k), (np.nan, np.nan))
            if np.isfinite([o, c, mu, sd]).all() and sd > 0 and abs(mu - o) > sd:
                trait_tot[i] += 1
                trait_hits[i] += np.sign(c - o) == np.sign(mu - o)

    df = pd.DataFrame(dict(source=[classes[i] for i in y], target=[classes[i] for i in t],
                           reached=(p_out.argmax(1).numpy() == t).astype(int), lpips=d_lp,
                           loc_in=loc_in, loc_out=loc_out, transfer=(pe.argmax(1).numpy() == t).astype(int),
                           trait_hits=trait_hits, trait_tot=trait_tot))
    df['n_target'] = df.target.map(n_train)
    df['n_source'] = df.source.map(n_train)
    df.to_csv(args.out.replace('.txt', '.csv'), index=False)

    def summary(g):
        return dict(n=len(g), reached=g.reached.mean(), lpips=g.lpips.mean(), locality=g.loc_in.mean() / g.loc_out.mean(),
                    transfer=g.transfer.mean(), traits=g.trait_hits.sum() / max(g.trait_tot.sum(), 1), trait_n=int(g.trait_tot.sum()))

    L = [f'{n} counterfactuals; training images per species (random split): target median {df.n_target.median():.0f} '
         f'(range {df.n_target.min()}-{df.n_target.max()}), source median {df.n_source.median():.0f}', '']
    for who in ['n_target', 'n_source']:
        L.append(f'BY {who.upper()} (training images of the {who[2:]} species)')
        L.append(f'{"bin":>10s} {"n":>4s} {"reached":>8s} {"LPIPS":>6s} {"locality":>9s} {"transfer":>9s} {"traits toward target":>22s}')
        cut = pd.cut(df[who], args.bins, right=False)
        for b, g in df.groupby(cut):
            if not len(g):
                continue
            s = summary(g)
            label = f'{int(b.left)}-{int(b.right) - 1}' if b.right < 10 ** 6 else f'>={int(b.left)}'
            L.append(f'{label:>10s} {s["n"]:4d} {s["reached"]:8.1%} {s["lpips"]:6.3f} {s["locality"]:8.0f}x '
                     f'{s["transfer"]:9.1%} {s["traits"]:12.1%} (n={s["trait_n"]})')
        lx = np.log(df[who])
        tr_rate = np.where(df.trait_tot > 0, df.trait_hits / df.trait_tot.clip(lower=1), np.nan)
        cors = []
        for name, v in [('reached', df.reached), ('LPIPS', df.lpips), ('transfer', df.transfer),
                        ('locality', np.log(np.array(loc_in) / np.array(loc_out))), ('traits', tr_rate)]:
            ok = np.isfinite(v)
            rho, p = spearmanr(lx[ok], np.asarray(v)[ok])
            cors.append(f'{name} {rho:+.2f} (p={p:.2g})')
        L.append('  Spearman with log(training images): ' + ', '.join(cors))
        q1, q4 = df[who].quantile(0.25), df[who].quantile(0.75)
        lo, hi = df[df[who] <= q1], df[df[who] >= q4]
        ft = lambda col: fisher_exact([[lo[col].sum(), len(lo) - lo[col].sum()], [hi[col].sum(), len(hi) - hi[col].sum()]])[1]
        L.append(f'  lowest quartile (<= {q1:.0f} images, n={len(lo)}) vs highest (>= {q4:.0f}, n={len(hi)}): '
                 f'reached {lo.reached.mean():.1%} vs {hi.reached.mean():.1%} (Fisher p={ft("reached"):.2g}), '
                 f'transfer {lo.transfer.mean():.1%} vs {hi.transfer.mean():.1%} (p={ft("transfer"):.2g}), '
                 f'LPIPS {lo.lpips.mean():.3f} vs {hi.lpips.mean():.3f} (Mann-Whitney p={mannwhitneyu(lo.lpips, hi.lpips).pvalue:.2g})')
        L.append('')
    txt = '\n'.join(L)
    print(txt)
    open(args.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
