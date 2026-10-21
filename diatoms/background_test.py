"""Does the classifier decide from the specimen or from the background?

For the source specimens of a DVCE run (and their counterfactuals), with the specimen
mask from recon_metrics.diatom_mask:
  1. swap      the specimen pasted onto the background of a photo from another atlas page
               (that photo's own specimen region filled with its background median);
               does the classifier still predict the true species?
  2. bg_only   the specimen region filled with the background median (background alone);
               how often is the true species still predicted? (chance 1/759)
  3. cf_spec   counterfactual specimen on the ORIGINAL background: still the target?
     orig_spec original specimen on the COUNTERFACTUAL's background: target, or back to source?
If decisions come from the specimen, (1) stays close to the unmodified accuracy, (2) is
near chance, cf_spec keeps the target and orig_spec loses it.

  python diatoms/background_test.py --pth <wid_0 pth> <wid_1 pth> --pairs evaluation/vce_pairs_paper_all.json \
      --ckpt DiatlasModels/r50madry_random_eps1.0/best.pt --eval_ckpt DiatlasModels/r50imagenet_page_eps0/best.pt \
      --out results/dvce/full_r50eps1_ftv2/background_test.txt
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diatom_classifier import load_diatom_classifier  # noqa: E402
from recon_metrics import diatom_mask  # noqa: E402
from train_classifier import DATA_NATIVE  # noqa: E402


def predict(clf, imgs):
    with torch.no_grad():
        return torch.cat([clf(imgs[i:i + 50]).softmax(1) for i in range(0, len(imgs), 50)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pth', nargs='+', required=True)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--ckpt', required=True, help='explained classifier')
    ap.add_argument('--eval_ckpt', nargs='*', default=[], help='further classifiers to report')
    ap.add_argument('--n_swaps', type=int, default=3)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ds = [torch.load(p, map_location='cpu') for p in args.pth]
    x = torch.cat([d['gt_imgs'].float() for d in ds])[::2]          # one per source
    cf = torch.cat([d['counterfactuals'][:, 0, 0].float().clamp(0, 1) for d in ds])
    x_all = torch.cat([d['gt_imgs'].float() for d in ds])
    y = torch.cat([d['gt_labels'][:len(d['gt_imgs'])] for d in ds])
    t = torch.cat([d['targets'] for d in ds])
    ys = y[::2]
    n = len(x)

    # match each source to its file to know its atlas page (for "another page" backgrounds)
    entries = json.load(open(args.pairs))['entries']
    files = {e['path']: np.asarray(Image.open(os.path.join(DATA_NATIVE, e['path'])).convert('L'), np.float32) / 255
             for e in entries}
    with open(os.path.join(DATA_NATIVE, 'splits.csv')) as f:
        meta = {r['path']: r for r in csv.DictReader(f)}
    page = []
    for i in range(n):
        g = x[i].mean(0).numpy()
        pth = min(files, key=lambda q: np.abs(files[q] - g).mean())
        page.append((meta[pth]['source'], meta[pth]['page']))

    gray = x.mean(1).numpy()
    masks = [diatom_mask(np.repeat(gray[i][..., None], 3, 2)) for i in range(n)]
    bg_med = np.array([np.median(gray[i][~masks[i]]) for i in range(n)])
    bg_only = np.stack([np.where(masks[i], bg_med[i], gray[i]) for i in range(n)])   # specimen removed

    rng = np.random.default_rng(0)
    swaps, swap_src = [], []
    for i in range(n):
        others = [j for j in range(n) if page[j] != page[i]]
        for j in rng.choice(others, args.n_swaps, replace=False):
            # specimen i on background j; shift i's specimen to j's background level
            spec = gray[i] - bg_med[i] + bg_med[j]
            swaps.append(np.where(masks[i], spec, bg_only[j]))
            swap_src.append(i)
    swaps, swap_src = np.stack(swaps), np.array(swap_src)

    # counterfactual composites (two counterfactuals per source)
    gcf = cf.mean(1).numpy()
    gx = x_all.mean(1).numpy()
    mk = [masks[k // 2] for k in range(len(cf))]
    cf_spec = np.stack([np.where(mk[k], gcf[k], gx[k]) for k in range(len(cf))])      # cf specimen, original bg
    orig_spec = np.stack([np.where(mk[k], gx[k], gcf[k]) for k in range(len(cf))])    # original specimen, cf bg

    to3 = lambda a: torch.from_numpy(np.clip(a, 0, 1)).float()[:, None].repeat(1, 3, 1, 1)
    lines = [f'{n} source specimens, {len(cf)} counterfactuals; {args.n_swaps} background swaps per specimen '
             f'(backgrounds from other atlas pages)', '']
    for name, ck in [('explained', args.ckpt)] + [(os.path.basename(os.path.dirname(c)), c) for c in args.eval_ckpt]:
        clf, _, desc = load_diatom_classifier(ck)
        clf.eval()
        p_x = predict(clf, x)
        p_sw = predict(clf, to3(swaps))
        p_bg = predict(clf, to3(bg_only))
        acc = (p_x.argmax(1) == ys).float().mean().item()
        acc_sw = (p_sw.argmax(1) == ys[swap_src]).float().mean().item()
        acc_bg = (p_bg.argmax(1) == ys).float().mean().item()
        p_cf = predict(clf, cf)
        p_cs = predict(clf, to3(cf_spec))
        p_os = predict(clf, to3(orig_spec))
        tgt = lambda p: (p.argmax(1) == t).float().mean().item()
        src = lambda p: (p.argmax(1) == y).float().mean().item()
        lines += [f'== {name}: {desc}',
                  f'  originals: true species {acc:.1%} | on another page\'s background {acc_sw:.1%} | background alone {acc_bg:.1%} (chance 0.1%)',
                  f'  counterfactuals: target {tgt(p_cf):.1%} | cf specimen on original background: target {tgt(p_cs):.1%} | '
                  f'original specimen on cf background: target {tgt(p_os):.1%}, source {src(p_os):.1%}', '']
    txt = '\n'.join(lines)
    print(txt)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, 'w').write(txt + '\n')
    # a few examples for visual checking
    ex = np.concatenate([np.concatenate([gray[i], bg_only[i], swaps[i * args.n_swaps]], 1) for i in range(4)], 0)
    Image.fromarray((np.clip(ex, 0, 1) * 255).astype(np.uint8)).save(args.out.replace('.txt', '_examples.png'))


if __name__ == '__main__':
    main()
