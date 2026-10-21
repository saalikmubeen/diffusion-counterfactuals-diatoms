"""Compare DVCE runs on the same pair list (e.g. the main run and its ablations).

For each run: target reached, LPIPS, change on the diatom vs background, and
transfer to a fixed set of evaluator classifiers (the same for every run, so
the rows are comparable). Paired Wilcoxon tests on LPIPS against the first run.
Optionally a figure: the same source diatoms, one row per run.

Example:
  python diatoms/compare_dvce_runs.py \
      --run main:DiatlasVCEs/clf_r50madry_random_eps1.0__diff_ft_v2_nativesize/<settings>_wid_{0,1}_2_dvces/50.pth \
      --run imagenet_prior:... --run nonrobust_clf:... \
      --eval_ckpt DiatlasModels/resnet18_native_eps0.5/best.pt DiatlasModels/r50madry_random_eps0.5/best.pt \
      --eval_name indep_robust_r18 our_robust_r50_eps0.5 --out_dir results/dvce/ablations
"""
import argparse
import os
import sys

import lpips
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import binomtest, wilcoxon  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diatom_classifier import load_diatom_classifier  # noqa: E402
from recon_metrics import diatom_mask  # noqa: E402


def load_run(pths):
    ds = [torch.load(p, map_location='cpu') for p in pths]
    x = torch.cat([d['gt_imgs'].float() for d in ds])
    return dict(x=x,
                xcf=torch.cat([d['counterfactuals'][:, 0, 0].float().clamp(0, 1) for d in ds]),
                y=torch.cat([d['gt_labels'][:len(d['gt_imgs'])].long() for d in ds]),
                t=torch.cat([d['targets'].long() for d in ds]),
                p_out=torch.cat([d['out_probabilities'][:, 0, 0] for d in ds]),
                p_in=torch.cat([d['in_probabilities'][:, 0, 0] for d in ds]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='append', required=True, help='name:pth[,pth]  (first run = reference)')
    p.add_argument('--eval_ckpt', nargs='*', default=[])
    p.add_argument('--eval_name', nargs='*', default=[])
    p.add_argument('--fig_sources', type=int, nargs='*', default=[0, 3, 7, 12, 20, 33])
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = {}
    for r in args.run:
        name, pths = r.split(':', 1)
        runs[name] = load_run(pths.split(','))
    ref = runs[next(iter(runs))]
    for name, r in runs.items():
        assert torch.equal(r['y'], ref['y']) and torch.equal(r['t'], ref['t']), f'{name}: different pairs'
        assert (r['x'] - ref['x']).abs().max() < 1e-3, f'{name}: different source images'

    lp = lpips.LPIPS(net='alex', verbose=False).eval()
    evals = [(n, load_diatom_classifier(c)[0].eval()) for n, c in zip(args.eval_name, args.eval_ckpt)]
    masks = torch.stack([torch.from_numpy(diatom_mask(ref['x'][i].permute(1, 2, 0).numpy())) for i in range(len(ref['x']))])

    stats = {}
    for name, r in runs.items():
        with torch.no_grad():
            d_lp = torch.cat([lp(r['x'][i:i + 20] * 2 - 1, r['xcf'][i:i + 20] * 2 - 1).flatten()
                              for i in range(0, len(r['x']), 20)]).numpy()
        diff = (r['xcf'] - r['x']).abs().mean(1)
        l1_in = np.array([diff[i][masks[i]].mean().item() for i in range(len(diff))])
        l1_out = np.array([diff[i][~masks[i]].mean().item() for i in range(len(diff))])
        crop = torch.zeros(256, 256, dtype=torch.bool)
        crop[16:240, 16:240] = True                   # what the classifier sees
        l1_out_crop = np.array([diff[i][~masks[i] & crop].mean().item() for i in range(len(diff))])
        n = len(r['t'])
        s = dict(success=(r['p_out'].argmax(1) == r['t']).float().numpy(),
                 p_after=r['p_out'][range(n), r['t']].numpy(), lpips=d_lp, l1=diff.mean((1, 2)).numpy(),
                 l1_in=l1_in, l1_out=l1_out, l1_out_crop=l1_out_crop)
        for en, clf in evals:
            with torch.no_grad():
                pr = torch.cat([clf(r['xcf'][i:i + 50]).softmax(1) for i in range(0, n, 50)])
            s[en] = (pr.argmax(1) == r['t']).float().numpy()
            s[en + '_p'] = pr[range(n), r['t']].numpy()
        stats[name] = s

    head = f'{"run":18s} {"reached":>8s} {"med p":>6s} {"LPIPS":>6s} {"L1":>7s} {"in/out":>7s} {"in/out224":>9s}' + \
        ''.join(f' {en:>22s}' for en, _ in evals)
    lines = [f'{len(ref["t"])} counterfactuals ({len(ref["t"]) // 2} source diatoms), same pairs for every run', '', head]
    for name, s in stats.items():
        lines.append(f'{name:18s} {s["success"].mean():8.1%} {np.median(s["p_after"]):6.3f} {s["lpips"].mean():6.3f} '
                     f'{s["l1"].mean():7.4f} '
                     f'{s["l1_in"].mean() / max(s["l1_out"].mean(), 1e-9):6.0f}x '
                     f'{s["l1_in"].mean() / max(s["l1_out_crop"].mean(), 1e-9):8.0f}x'
                     + ''.join(f' {s[en].mean():8.1%} (p {s[en + "_p"].mean():.3f})' for en, _ in evals))
    lines.append('')
    ref_name = next(iter(stats))
    for name, s in list(stats.items())[1:]:
        same = np.allclose(stats[ref_name]['lpips'], s['lpips'])
        pv = 1.0 if same else wilcoxon(stats[ref_name]['lpips'], s['lpips']).pvalue
        lines.append(f'LPIPS {ref_name} vs {name}: {stats[ref_name]["lpips"].mean():.3f} vs {s["lpips"].mean():.3f}, '
                     f'paired Wilcoxon p = {pv:.1e}; {name} edit larger in '
                     f'{int((s["lpips"] > stats[ref_name]["lpips"]).sum())}/{len(s["lpips"])}')
        for key in ['success'] + [en for en, _ in evals]:
            a, b = stats[ref_name][key].astype(int), s[key].astype(int)
            only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
            pm = 1.0 if only_a + only_b == 0 else binomtest(only_a, only_a + only_b, 0.5).pvalue
            lines.append(f'  {key:22s} {a.mean():6.1%} vs {b.mean():6.1%}  (only {ref_name}: {only_a}, only {name}: {only_b}, '
                         f'exact McNemar p = {pm:.1e})')
    lines.append('')
    lines.append('L1: mean absolute pixel change; in/out: mean change inside the diatom / outside it (whole 256 canvas);')
    lines.append('in/out224: same, background restricted to the central 224x224 crop the classifier sees.')
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(args.out_dir, 'comparison.txt'), 'w') as f:
        f.write(txt + '\n')

    # figure: for each chosen source, original + counterfactual (target 1) from each run
    if args.fig_sources:
        names = list(runs)
        ks = [2 * s for s in args.fig_sources]
        fig, ax = plt.subplots(len(ks), 1 + len(names), figsize=(2.6 * (1 + len(names)), 2.75 * len(ks)), squeeze=False)
        to_np = lambda t: t.permute(1, 2, 0).numpy().mean(2)
        for i, k in enumerate(ks):
            ax[i, 0].imshow(to_np(ref['x'][k]), cmap='gray', vmin=0, vmax=1)
            ax[i, 0].set_title('original' if i == 0 else '', fontsize=10)
            for j, n in enumerate(names):
                s = stats[n]
                ax[i, j + 1].imshow(to_np(runs[n]['xcf'][k]), cmap='gray', vmin=0, vmax=1)
                ax[i, j + 1].set_title((n + '\n' if i == 0 else '') +
                                       f'p(target) {s["p_after"][k]:.2f}, LPIPS {s["lpips"][k]:.2f}', fontsize=8.5)
        for a in ax.ravel():
            a.set_xticks([]); a.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, 'comparison.png'), dpi=110)
        print('figure:', os.path.join(args.out_dir, 'comparison.png'))


if __name__ == '__main__':
    main()
