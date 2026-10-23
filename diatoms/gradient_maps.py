"""Where does the classifier push during DVCE, and does the image change there?

Reads the gradient logs written by diatlas_VCEs.py --log_gradients
(<run dir>/gradient_logs/batch_*.pth): per DVCE step, the raw classifier gradient
d log p(target)/d x_t (w.r.t. the NOISY image, through the denoiser; before
renormalisation) and the net guidance actually applied (classifier push minus distance pull).

Per counterfactual it makes one figure:
  original | counterfactual | actual change |cf - original| | classifier push, all steps |
  classifier push in 4 phases of the sampling (early = high noise ... late = low noise)
and reports:
  - share of the classifier push (each step's gradient rescaled to the same norm, as DVCE
    applies it) that lies on the specimen; the net guidance (push minus the L1 distance pull,
    which is spread over all pixels) is reported too
  - corr(guidance, change): Pearson correlation between the smoothed accumulated |guidance|
    and the smoothed actual change, inside the dilated specimen region; per phase as well
  - the raw classifier gradient (before renormalisation): how its size is spread over the
    steps (mass_per_step.png) and its correlation with the change
|gradient| is sensitivity, not change; the actual change is shown next to it.

  python diatoms/gradient_maps.py --log_dir <run dir>/gradient_logs --out_dir results/dvce/gradient_maps
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_metrics import diatom_mask  # noqa: E402

N_PHASES = 4


def smooth(a, sigma=4):
    return ndimage.gaussian_filter(a, sigma)


def norm01(a):
    hi = np.quantile(a, 0.99)
    return np.clip(a / hi, 0, 1) if hi > 0 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir', required=True)
    ap.add_argument('--ckpt', default='DiatlasModels/r50madry_random_eps1.0/best.pt', help='for class names')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    classes = torch.load(args.ckpt, map_location='cpu')['classes']

    rows, masses = [], []
    k = 0
    for f in sorted(glob.glob(os.path.join(args.log_dir, 'batch_*.pth'))):
        d = torch.load(f, map_location='cpu')
        G = d['grad_class'].float().abs().numpy()            # (T, B, H, W) raw classifier gradient
        Gs = d['grad_class'].float()
        # classifier push as DVCE applies it: each step's gradient rescaled to unit L2 norm (enforce_same_norms)
        N = (Gs / Gs.flatten(2).norm(dim=2)[..., None, None].clamp_min(1e-20)).abs().numpy()
        D = (d['grad_net'].float()).abs().numpy()           # applied push minus distance pull (net)
        T = G.shape[0]
        phases = np.array_split(np.arange(T), N_PHASES)
        for b in range(G.shape[1]):
            init = d['init_image'][b].mean(0).numpy()
            fin = d['final_image'][b].mean(0).numpy()
            change = smooth(np.abs(fin - init))
            acc = smooth(G[:, b].sum(0))
            net = smooth(N[:, b].sum(0))
            ph = [smooth(N[p, b].sum(0)) for p in phases]          # phases of the APPLIED guidance
            mask = diatom_mask(np.repeat(init[..., None], 3, 2))
            region = ndimage.binary_dilation(mask, iterations=10)
            corr = lambda a: float(np.corrcoef(a[region], change[region])[0, 1])
            tot = sum(p.sum() for p in ph)
            rows.append(dict(i=k, target=classes[int(d['targets'][b])], steps=T,
                             mass_in=float(net[mask].sum() / net.sum()), area=float(mask.mean()), corr=corr(net),
                             corr_raw=corr(acc), raw_early=float(G[phases[0], b].sum() / G[:, b].sum()),
                             net_in=float(D[:, b].sum(0)[mask].sum() / D[:, b].sum()),
                             phase_share=[float(p.sum() / tot) for p in ph], phase_corr=[corr(p) for p in ph]))
            masses.append(G[:, b].reshape(T, -1).sum(1))

            fig, ax = plt.subplots(1, 4 + N_PHASES, figsize=(2.4 * (4 + N_PHASES), 2.9))
            ax[0].imshow(init, cmap='gray', vmin=0, vmax=1); ax[0].set_title('original', fontsize=8)
            ax[1].imshow(fin, cmap='gray', vmin=0, vmax=1); ax[1].set_title(f'counterfactual\n-> {rows[-1]["target"]}', fontsize=8)
            ax[2].imshow(norm01(change), cmap='magma'); ax[2].set_title('actual change\n|cf - original|', fontsize=8)
            ax[3].imshow(init, cmap='gray', vmin=0, vmax=1); ax[3].imshow(norm01(net), cmap='jet', alpha=0.5)
            ax[3].set_title(f'classifier push, all steps\ncorr with change {rows[-1]["corr"]:.2f}', fontsize=8)
            for j, p in enumerate(ph):
                ax[4 + j].imshow(init, cmap='gray', vmin=0, vmax=1); ax[4 + j].imshow(norm01(p), cmap='jet', alpha=0.5)
                ax[4 + j].set_title(f'steps {phases[j][0] + 1}-{phases[j][-1] + 1}\ncorr {rows[-1]["phase_corr"][j]:.2f}', fontsize=8)
            for a in ax:
                a.axis('off')
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f'grad_{k:03d}.png'), dpi=110)
            plt.close(fig)
            k += 1

    if not rows:
        raise SystemExit('no gradient logs found')
    M = np.stack(masses)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(1, M.shape[1] + 1), (M / M.sum(1, keepdims=True)).T, color='gray', alpha=0.3, lw=0.8)
    ax.plot(np.arange(1, M.shape[1] + 1), (M / M.sum(1, keepdims=True)).mean(0), color='k', lw=2, label='mean')
    ax.set_xlabel('DVCE step (high noise -> low noise)'); ax.set_ylabel('share of total raw |grad|'); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'mass_per_step.png'), dpi=120); plt.close(fig)

    pc = np.array([r['phase_corr'] for r in rows]); ps = np.array([r['phase_share'] for r in rows])
    phases_all = np.array_split(np.arange(rows[0]['steps']), N_PHASES)
    L = [f'{len(rows)} counterfactuals, {rows[0]["steps"]} logged steps each. Maps use the classifier push as DVCE applies it',
         '(each step\'s gradient rescaled to the same norm); raw gradients and the net guidance are reported separately.', '',
         f'raw classifier gradient: {np.mean([r["raw_early"] for r in rows]):.0%} of its total size falls in the first quarter of the steps (high noise);',
         f'  corr(accumulated raw |grad|, actual change) {np.mean([r["corr_raw"] for r in rows]):.2f}', '',
         f'net guidance (push minus distance pull) on the specimen: {np.mean([r["net_in"] for r in rows]):.0%} (the L1 distance pull is spread over all pixels)',
         f'share of the classifier push on the specimen: mean {np.mean([r["mass_in"] for r in rows]):.0%} '
         f'(the specimen covers {np.mean([r["area"] for r in rows]):.0%} of the image on average)',
         f'corr(classifier push, actual change) in the specimen region: mean {np.mean([r["corr"] for r in rows]):.2f} '
         f'(min {min(r["corr"] for r in rows):.2f}, max {max(r["corr"] for r in rows):.2f})', '',
         'phase (classifier push)    share   corr with actual change']
    for j in range(N_PHASES):
        L.append(f'  steps {phases_all[j][0] + 1:3d}-{phases_all[j][-1] + 1:3d}            {ps[:, j].mean():5.0%}   {pc[:, j].mean():.2f}')
    L += ['', 'per counterfactual: ' + ', '.join(f'{r["target"]} corr {r["corr"]:.2f}' for r in rows)]
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(args.out_dir, 'summary.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
