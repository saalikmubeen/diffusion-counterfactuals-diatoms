"""DIATLAS diatom dataset (preprocessed by diatoms/prepare_diatlas.py).

Reads data/diatlas/splits.csv and returns (image, label) with the image as a
3x256x256 float tensor in [0, 1], like the ImageNet loader used by DVCE.

Label granularity is chosen at construction:
  label_key='taxon' (species / infraspecific taxon), 'genus', or 'species_code'
  min_count: keep only classes with at least this many images over all splits
Classes are sorted alphabetically, so indices are stable for a given
(label_key, min_count). Images of classes below min_count are left out.

classes=<list> overrides that: labels are indices into this exact list (e.g. a
diatom classifier's 759 species_code outputs), images whose label is not in it
or is empty (genus-only identifications) are left out. Use root= to pick the
preprocessed copy (data/diatlas = stretch-to-fit, data/diatlas_nativesize_crop224_canvas256
= native size).
"""
import collections
import csv
import os

import torch
from PIL import Image
from torchvision import transforms

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            'data', 'diatlas')


def get_diatlas_labels(root=DEFAULT_ROOT, label_key='taxon', min_count=100):
    """Sorted class names that have >= min_count images (counted over all splits)."""
    with open(os.path.join(root, 'splits.csv')) as f:
        counts = collections.Counter(r[label_key] for r in csv.DictReader(f))
    return sorted(c for c, n in counts.items() if n >= min_count)


class DiatlasDataset(torch.utils.data.Dataset):
    def __init__(self, root=DEFAULT_ROOT, split='test', label_key='taxon', min_count=100, transform=None, classes=None):
        self.root = root
        self.classes = list(classes) if classes is not None else get_diatlas_labels(root, label_key, min_count)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        splits = split.split(',')
        with open(os.path.join(root, 'splits.csv')) as f:
            rows = [r for r in csv.DictReader(f) if r['split'] in splits and r[label_key].strip()
                    and r[label_key] in self.class_to_idx]
        rows.sort(key=lambda r: r['path'])
        self.rows = rows
        self.paths = [r['path'] for r in rows]
        self.targets = [self.class_to_idx[r[label_key]] for r in rows]
        self.path_to_idx = {p: i for i, p in enumerate(self.paths)}
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.paths[idx])).convert('RGB')
        return self.transform(img), self.targets[idx]


def get_diatlas(root=DEFAULT_ROOT, split='test', label_key='taxon', min_count=100, batch_size=32,
                shuffle=None, num_workers=4, transform=None):
    dataset = DiatlasDataset(root, split, label_key, min_count, transform)
    if shuffle is None:
        shuffle = split == 'train'
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
