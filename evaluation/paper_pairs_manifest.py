"""Build a DVCE manifest from source -> target class pairs shown in the paper (Augustin et al. 2022).

Pairs are taken from Fig. 14 (user-study images, robust MNR-RN50 = Madry+FT, no cone projection) and
Fig. 5 (non-robust classifiers). The paper does not publish which validation images it used, so for
each source class one validation image is drawn with random.seed(SEED) / random.choice, in the order
of PAIRS below.

The output has the same format as evaluation/select_vces.py (entries + class_group), so it can be passed
to imagenet_VCEs.py --vce_list and evaluation/compute_metrics.py. class_group 2-colours the classes so
that every source/target pair crosses the A/B split (paper App. E); only the classes used are assigned.

Usage:
    python evaluation/paper_pairs_manifest.py --data_folder /path/to/parent_of_imagenet \
        --out evaluation/vce_list_paper30.json
"""
import argparse
import json
import os
import random
import sys
from collections import deque

from torchvision.datasets import ImageNet

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils_svces.datasets as dl

SEED = 1

# (source, [target 1, target 2], figure)
PAIRS = [
    ('puffer', ['goldfish', 'lionfish'], 'fig14'),
    ('hartebeest', ['gazelle', 'bison'], 'fig14'),
    ('valley', ['volcano', 'alp'], 'fig14'),
    ('strawberry', ['pineapple', 'pomegranate'], 'fig14'),
    ('loggerhead', ['leatherback', 'box turtle'], 'fig5'),
    ('black and gold garden spider', ['black widow', 'tarantula'], 'fig5'),
    ('sea urchin', ['sea cucumber', 'starfish'], 'fig5'),
    ('lorikeet', ['macaw', 'African grey'], 'fig5'),
    ('red wolf', ['coyote', 'timber wolf'], 'fig5'),
    ('pineapple', ['jackfruit', 'custard apple'], 'fig5'),
    ('harvestman', ['black and gold garden spider', 'tick'], 'fig14'),
    ('rugby ball', ['golf ball', 'baseball'], 'fig14'),
    ('thunder snake', ['night snake', 'king snake'], 'fig14'),
    ('eft', ['tailed frog', 'axolotl'], 'fig14'),
    ('leopard', ['snow leopard', 'cheetah'], 'fig14'),
]

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', required=True, help='parent directory of imagenet/ (as for imagenet_VCEs.py)')
parser.add_argument('--out', required=True)
parser.add_argument('--num_pairs', type=int, default=30, help='total number of image-target pairs (2 per image)')
args = parser.parse_args()
assert args.num_pairs % 2 == 0 and args.num_pairs // 2 <= len(PAIRS), f'at most {2 * len(PAIRS)} pairs available'

dataset = ImageNet(os.path.join(args.data_folder, 'imagenet'), split='val')
upstream_ds = dl.get_ImageNet(path=args.data_folder, train=False, augm_type='crop_0.875', size=256).dataset
assert upstream_ds.samples == dataset.samples, 'imagenet_VCEs.py loader indexes different images'
names = [c[0] for c in dataset.classes]


def find_class_id(class_name):
    matches = [i for i, synonyms in enumerate(dataset.classes)
               if class_name.lower() in [n.lower() for n in synonyms]]
    if len(matches) != 1:
        raise ValueError(f'{class_name!r} matches {len(matches)} classes: {[dataset.classes[i] for i in matches]}')
    return matches[0]


random.seed(SEED)
entries = []
for source_name, target_names, fig in PAIRS[:args.num_pairs // 2]:
    label = find_class_id(source_name)
    targets = [find_class_id(t) for t in target_names]
    idx = random.choice([i for i, (_, y) in enumerate(dataset.samples) if y == label])
    entries.append(dict(idx=idx, label=label, targets=targets, source=f'paper-{fig}'))

# 2-colour the source/target graph into the crossover sets A and B
edges = {}
for e in entries:
    for t in e['targets']:
        edges.setdefault(e['label'], set()).add(t)
        edges.setdefault(t, set()).add(e['label'])
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
            assert group[t] != group[c], f'pairs cannot be split into crossing sets ({names[c]}, {names[t]})'

assert len({e['idx'] for e in entries}) == len(entries), 'duplicate source images'
for e in entries:
    assert len(set(e['targets'])) == 2 and e['label'] not in e['targets']
    e['path'] = '/'.join(dataset.samples[e['idx']][0].split('/')[-2:])
    e['label_name'] = names[e['label']]
    e['target_names'] = [names[t] for t in e['targets']]
    e['group'] = group[e['label']]
    print(f"{e['idx']:>6} {e['path']:<45} {e['group']} {e['label_name']:<28} -> "
          f"{e['targets']} {e['target_names']} [{e['source']}]")

with open(args.out, 'w') as f:
    json.dump(dict(seed=SEED, loader_matches_imagenet_VCEs=True, class_group=group, entries=entries), f, indent=1)
print(f'{len(entries)} images, {2 * len(entries)} VCEs -> {args.out}')
print('\nsome_vces = {')
for e in entries:
    print(f"    {e['idx']}: [{e['targets'][0]}, {e['targets'][1]}],  # {e['label_name']}")
print('}')
