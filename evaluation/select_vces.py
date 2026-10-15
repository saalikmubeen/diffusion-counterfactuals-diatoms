"""Build a fixed, validated list of (ImageNet val index, [target, target]) pairs for DVCE evaluation.

Follows the "crossover" protocol of Augustin et al. 2022, App. E, as far as it is specified:
  1. split the ImageNet classes into two disjoint sets A and B,
  2. for a source image of a class in one set, use as targets semantically close classes
     ("same WordNet cluster") that lie in the *other* set.
The paper's actual A/B split and WordNet clusters are not published, so this is an approximation:
  - A/B: the classes of the (valid) upstream `some_vces` pairs are 2-coloured so that each upstream
    source/target pair crosses the split; all remaining classes are split by a seeded shuffle (500/500).
  - "same WordNet cluster": Wu-Palmer similarity >= --min_wup between the source and target synsets.

Indices refer to torchvision.datasets.ImageNet(split='val'); this is exactly the dataset that
imagenet_VCEs.py indexes (utils_svces/datasets/imagenet.py: get_ImageNet -> datasets.ImageNet(split='val')),
and the script re-checks that the two sample lists are identical.

Usage:
    python evaluation/select_vces.py --data_folder /path/to/parent_of_imagenet --out evaluation/vce_list_30.json
"""
import argparse
import json
import os
import sys
from collections import deque

import numpy as np
from nltk.corpus import wordnet as wn
from torchvision.datasets import ImageNet

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils_svces.datasets as dl

# upstream `some_vces` from imagenet_VCEs.py
UPSTREAM = {
    14655: [288, 292],
    10452: [207, 208],
    46751: [924, 959],
    48679: [970, 972],
    48539: [970, 980],
    48282: [963, 965],
}

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', required=True, help='parent directory of imagenet/ (as for imagenet_VCEs.py)')
parser.add_argument('--out', required=True)
parser.add_argument('--num_pairs', type=int, default=30, help='total number of image-target pairs (2 per image)')
parser.add_argument('--min_wup', type=float, default=0.85,
                    help='min Wu-Palmer similarity between source and target synsets (stand-in for "same WordNet cluster")')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
assert args.num_pairs > 0 and args.num_pairs % 2 == 0, '--num_pairs must be a positive even number'

ds = ImageNet(os.path.join(args.data_folder, 'imagenet'), split='val')
upstream_ds = dl.get_ImageNet(path=args.data_folder, train=False, augm_type='crop_0.875', size=256).dataset
assert upstream_ds.samples == ds.samples and upstream_ds.wnids == ds.wnids, \
    'imagenet_VCEs.py loader does not index the same images as torchvision ImageNet val'
synsets = [wn.synset_from_pos_and_offset('n', int(w[1:])) for w in ds.wnids]
names = [c[0] for c in ds.classes]
num_classes = len(names)


def wup(c, t):
    return synsets[c].wup_similarity(synsets[t])


# ---------------------------------------------------------------- A/B class partition
# 2-colour the graph of valid upstream source->target pairs so every upstream pair crosses the split
edges = {}
for idx, targets in UPSTREAM.items():
    for t in targets:
        if t != ds.targets[idx]:
            edges.setdefault(ds.targets[idx], set()).add(t)
            edges.setdefault(t, set()).add(ds.targets[idx])
group = {}
for start in sorted(edges):
    if start in group:
        continue
    group[start] = 'A'
    queue = deque([start])
    while queue:
        c = queue.popleft()
        for t in edges[c]:
            if t not in group:
                group[t] = 'B' if group[c] == 'A' else 'A'
                queue.append(t)
            assert group[t] != group[c], 'upstream pairs cannot be split into two crossing sets'

rng = np.random.RandomState(args.seed)
rest = [c for c in rng.permutation(num_classes).tolist() if c not in group]
n_a = num_classes // 2 - sum(g == 'A' for g in group.values())
for i, c in enumerate(rest):
    group[c] = 'A' if i < n_a else 'B'


def neighbours(c, exclude=(), min_wup=None):
    """Classes in the other set with Wu-Palmer similarity >= min_wup to class c, most similar first."""
    min_wup = args.min_wup if min_wup is None else min_wup
    out = [(-wup(c, t), t) for t in range(num_classes)
           if group[t] != group[c] and t not in exclude and wup(c, t) >= min_wup]
    return sorted(out)


def validate_entry(e):
    idx, label, targets = e['idx'], e['label'], e['targets']
    assert 0 <= idx < len(ds) and ds.targets[idx] == label, f'{idx}: bad index/label'
    assert len(targets) == 2 and len(set(targets)) == 2, f'{idx}: need two distinct targets, got {targets}'
    assert label not in targets, f'{idx}: target equals source class'
    for t in targets:
        assert group[t] != group[label], f'{idx}: target {t} not in the opposite set'


# ---------------------------------------------------------------- selection
entries = []
for idx, targets in UPSTREAM.items():
    label = ds.targets[idx]
    valid = [t for t in targets if t != label]
    e = dict(idx=idx, label=label, targets=valid, source='upstream', upstream_targets=targets)
    if len(valid) < len(targets):
        # upstream entry targets its own GT class (not a counterfactual): replace with the closest
        # class of the other set; the original targets stay recorded in `upstream_targets`
        e['targets'] = valid + [t for _, t in neighbours(label, exclude=valid, min_wup=0)][:len(targets) - len(valid)]
        e['source'] = 'upstream-fixed'
    entries.append(e)

used_classes = {e['label'] for e in entries}
for idx in rng.permutation(len(ds)).tolist():
    if 2 * len(entries) == args.num_pairs:
        break
    label = ds.targets[idx]
    if label in used_classes:
        continue
    nb = neighbours(label)
    if len(nb) < 2:
        continue
    entries.append(dict(idx=idx, label=label, targets=[t for _, t in nb[:2]], source='wordnet-random'))
    used_classes.add(label)

n_pairs = sum(len(e['targets']) for e in entries)
assert n_pairs == args.num_pairs, f'only found {n_pairs} of {args.num_pairs} pairs'
for e in entries:
    validate_entry(e)
    e['path'] = '/'.join(ds.samples[e['idx']][0].split('/')[-2:])
    e['label_name'] = names[e['label']]
    e['target_names'] = [names[t] for t in e['targets']]
    e['group'] = group[e['label']]
    e['wup'] = [round(wup(e['label'], t), 3) for t in e['targets']]
    print(f"{e['idx']:>6} {e['path']:<45} {e['group']} {e['label_name']:<26} -> "
          f"{e['target_names']} wup={e['wup']} [{e['source']}]")

with open(args.out, 'w') as f:
    json.dump(dict(seed=args.seed, min_wup=args.min_wup, loader_matches_imagenet_VCEs=True,
                   class_group={c: group[c] for c in range(num_classes)}, entries=entries), f, indent=1)
print(f'{len(entries)} images, {n_pairs} VCEs -> {args.out}')
