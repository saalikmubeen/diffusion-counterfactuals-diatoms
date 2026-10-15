"""Compute the Table 1 metrics of Augustin et al. (NeurIPS 2022) for a set of generated DVCEs.

Inputs are the `<num_imgs>.pth` files written by imagenet_VCEs.py (one per --world_id shard).

Closeness  : l1, l1.5, l2 between VCE and original (256x256x3 images in [0, 1]), LPIPS (AlexNet)
Validity   : mean confidence of the guiding classifier (as stored by imagenet_VCEs.py) in the target class
Realism    : small-sample version of the paper's "crossover" FID (App. E), using the A/B class split
             stored in the --vce_list manifest: the VCEs with a target in A are compared with ImageNet
             *train* images of their target classes (label distribution matching the targets), likewise
             for B, and the two FIDs are averaged. The same is done for the original images as the paper's
             sanity baseline. KID (unbiased) is reported as well because FID is strongly biased at small n.

Usage:
    python evaluation/compute_metrics.py --pth <shard0.pth> <shard1.pth> \
        --vce_list evaluation/vce_list_30.json --imagenet_root /path/to/imagenet --out_dir logs/eval30
"""
import argparse
import csv
import json
import os
import sys

import lpips
import numpy as np
import torch
from PIL import Image
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_svces.datasets.augmentations.imagenet_augmentation import get_imageNet_augmentation

parser = argparse.ArgumentParser()
parser.add_argument('--pth', nargs='+', required=True)
parser.add_argument('--vce_list', required=True)
parser.add_argument('--imagenet_root', required=True)
parser.add_argument('--out_dir', required=True)
parser.add_argument('--ref_per_vce', type=int, default=100,
                    help='ImageNet train images per VCE in the FID reference set')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
device = torch.device(args.device)

# ---------------------------------------------------------------- load generated VCEs
keys = ['gt_imgs', 'counterfactuals', 'targets', 'gt_labels', 'out_probabilities', 'in_probabilities',
        'dataset_idcs']
data = {k: [] for k in keys}
for p in args.pth:
    d = torch.load(p, map_location='cpu')
    n = len(d['targets'])
    for k in keys:
        data[k].append(d[k][:n])
data = {k: torch.cat(v) for k, v in data.items()}
orig = data['gt_imgs'].clamp(0, 1)
vce = data['counterfactuals'][:, 0, 0].clamp(0, 1)
targets, labels, idcs = data['targets'], data['gt_labels'], data['dataset_idcs']
n = len(targets)

with open(args.vce_list) as f:
    entries = json.load(f)['entries']
expected = [(e['idx'], e['label'], t) for e in entries for t in e['targets']]
got = list(zip(idcs.tolist(), labels.tolist(), targets.tolist()))
assert sorted(got) == sorted(expected), f'generated VCEs do not match {args.vce_list}'
names = {e['label']: e['label_name'] for e in entries}
names.update({t: tn for e in entries for t, tn in zip(e['targets'], e['target_names'])})

# ---------------------------------------------------------------- closeness
diff = (vce - orig).flatten(1).abs()
l1 = diff.sum(1)
l15 = diff.pow(1.5).sum(1).pow(1 / 1.5)
l2 = diff.pow(2).sum(1).sqrt()

lpips_alex = lpips.LPIPS(net='alex').to(device).eval()
with torch.no_grad():
    lp = lpips_alex(vce.to(device), orig.to(device), normalize=True).flatten().cpu()

# ---------------------------------------------------------------- validity
ar = torch.arange(n)
out_p = data['out_probabilities'][:, 0, 0]
in_p = data['in_probabilities'][:, 0, 0]
conf = out_p[ar, targets]
init_conf = in_p[ar, targets]
pred = out_p.argmax(1)

# ---------------------------------------------------------------- realism (mini crossover FID)
inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()


@torch.no_grad()
def feats(x, bs=50):
    out = []
    for i in range(0, len(x), bs):
        out.append(inception(x[i:i + bs].to(device))[0].flatten(1).cpu())
    return torch.cat(out).double().numpy()


def fid(a, b):
    return float(calculate_frechet_distance(a.mean(0), np.cov(a, rowvar=False),
                                            b.mean(0), np.cov(b, rowvar=False)))


def kid(a, b):
    """Unbiased MMD^2 with the cubic polynomial kernel (Binkowski et al. 2018), on the full sets."""
    d = a.shape[1]
    k = lambda x, y: (x @ y.T / d + 1) ** 3
    kxx, kyy, kxy = k(a, a), k(b, b), k(a, b)
    m, r = len(a), len(b)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                 + (kyy.sum() - np.trace(kyy)) / (r * (r - 1)) - 2 * kxy.mean())


transform = get_imageNet_augmentation(type='crop_0.875', out_size=256)
wnids = sorted(os.listdir(os.path.join(args.imagenet_root, 'val')))
rng = np.random.RandomState(args.seed)


def reference_images(target_list):
    """ref_per_vce distinct train images per occurrence of each target class (label-distribution matched)."""
    imgs, ref_files = [], []
    for t, count in zip(*np.unique(target_list, return_counts=True)):
        cls_dir = os.path.join(args.imagenet_root, 'train', wnids[t])
        files = sorted(os.listdir(cls_dir))
        for fn in rng.choice(files, args.ref_per_vce * count, replace=False):
            ref_files.append(f'{wnids[t]}/{fn}')
            imgs.append(transform(Image.open(os.path.join(cls_dir, fn)).convert('RGB')))
    return torch.stack(imgs), ref_files


# paper: one FID for the VCEs with a target in A (vs train images of A), one for targets in B, averaged
class_group = {int(c): g for c, g in json.load(open(args.vce_list))['class_group'].items()}
half = np.array([class_group[t] for t in targets.tolist()])
realism, all_ref_files, ref_feats = {}, {}, {}
f_vce, f_orig = feats(vce), feats(orig)
for h in ('A', 'B'):
    m = half == h
    leaked = sorted(set(labels[m].tolist()) & set(targets[m].tolist()))
    assert not leaked, f'source classes {leaked} appear in the reference set of {h}'
    ref, ref_files = reference_images(targets[m].numpy())
    all_ref_files[h] = ref_files
    ref_feats[h] = feats(ref)
    realism[f'target_in_{h}'] = dict(
        n_vces=int(m.sum()), n_ref=len(ref), ref_classes=sorted(set(targets[m].tolist())),
        fid_vce=fid(f_vce[m], ref_feats[h]), fid_orig=fid(f_orig[m], ref_feats[h]),
        kid_vce=kid(f_vce[m], ref_feats[h]), kid_orig=kid(f_orig[m], ref_feats[h]))
for k in ('fid_vce', 'fid_orig', 'kid_vce', 'kid_orig'):
    realism[f'avg_{k}'] = float(np.mean([realism[f'target_in_{h}'][k] for h in ('A', 'B')]))
pooled_ref = np.concatenate([ref_feats['A'], ref_feats['B']])
realism['pooled'] = dict(fid_vce=fid(f_vce, pooled_ref), fid_orig=fid(f_orig, pooled_ref),
                         kid_vce=kid(f_vce, pooled_ref), kid_orig=kid(f_orig, pooled_ref))

# ---------------------------------------------------------------- report
summary = dict(
    n_vces=n, n_source_images=len(entries),
    l1=float(l1.mean()), l1_5=float(l15.mean()), l2=float(l2.mean()), lpips_alex=float(lp.mean()),
    mean_target_conf=float(conf.mean()), mean_initial_target_conf=float(init_conf.mean()),
    frac_pred_is_target=float((pred == targets).float().mean()),
    frac_conf_gt_0_9=float((conf > 0.9).float().mean()),
    realism=realism,
    paper_table1_dvce=dict(l1=12799, l1_5=293, l2=48, lpips_alex=0.35, mean_target_conf=0.932, avg_fid=17.6,
                           avg_fid_sanity_originals=41.5),
)
os.makedirs(args.out_dir, exist_ok=True)
with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
    json.dump(summary, f, indent=1)
with open(os.path.join(args.out_dir, 'fid_reference_files.json'), 'w') as f:
    json.dump(all_ref_files, f)
with open(os.path.join(args.out_dir, 'per_vce.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['dataset_idx', 'gt', 'target', 'target_set','l1', 'l1.5', 'l2', 'lpips_alex',
                'target_conf', 'initial_target_conf', 'pred'])
    for i in range(n):
        w.writerow([int(idcs[i]), names[int(labels[i])], names[int(targets[i])], half[i],
                    f'{l1[i]:.1f}', f'{l15[i]:.2f}', f'{l2[i]:.2f}', f'{lp[i]:.4f}',
                    f'{conf[i]:.4f}', f'{init_conf[i]:.4f}', int(pred[i])])

print(json.dumps(summary, indent=1))
