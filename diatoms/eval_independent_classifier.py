"""Load the independent classifier's diatom ResNet-18 (Lightning checkpoint) and measure its
accuracy on DIATLAS test images. CPU is enough; nothing is trained.

What we know about the checkpoint (DiatlasModels/independent_resnet18/last-v4.ckpt):
  * ResNet-18 with torchvision layout, but skip connections named `shortcut`
    and the head named `linear`; keys are prefixed `model.`. Loads into
    torchvision.models.resnet18 after renaming (max-pool layer included).
  * 759 outputs = species_code values with >= 2 images, connective views
    dropped, in alphabetical order (confirmed: far above chance with this order).
  * Its own validation accuracy was 0.861 (ModelCheckpoint, val/cls/Acc).
  * Preprocessing: the PadCropDiatom transform (diatoms/padcrop_transform.py,
    from his Slack message) + grayscale->3 channels + ImageNet mean/std.
    tgt_scale is not in his message; 0.0826 um/px (dataset median) works best.
    Best so far (2,000 images): --scale_policy none (no physical rescaling)
    top-1 0.750; with the csv scales top-1 0.689. Still below his 0.861.
  * --preproc ours: our 256x256 padded images, Resize 240 + CenterCrop 224,
    ImageNet mean/std (~0.27 top-1).

Examples:
  python diatoms/eval_independent_classifier.py --n 1000                          # his transform, no rescaling
  python diatoms/eval_independent_classifier.py --n 1000 --scale_policy csv       # his transform, csv scales
  python diatoms/eval_independent_classifier.py --n 1000 --preproc ours           # our 256 images
"""
import argparse
import collections
import csv
import os
import random
import sys

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIATLAS = '/mnt/diatoms/datasets/diatlas'
CKPT = os.path.join(REPO, 'DiatlasModels', 'independent_resnet18', 'last-v4.ckpt')
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def independent_classes(diatlas_root=DIATLAS):
    """The 759 class names (species_code), in output order. Built from the raw DIATLAS
    diatoms.csv; if that isn't mounted, read the saved copy next to the checkpoint
    (one name per line; the first class is the empty species code)."""
    if not os.path.exists(os.path.join(diatlas_root, 'diatoms.csv')):
        with open(os.path.join(os.path.dirname(CKPT), 'classes.txt')) as f:
            return f.read().split('\n')[:-1]
    with open(os.path.join(diatlas_root, 'diatoms.csv')) as f:
        rows = [r for r in csv.DictReader(f) if r['connective'] != 'True']
    counts = collections.Counter(r['species_code'] for r in rows)
    return sorted(c for c, n in counts.items() if n >= 2)


def load_independent_resnet18(ckpt=CKPT):
    sd = torch.load(ckpt, map_location='cpu')['state_dict']
    sd = {k[len('model.'):].replace('linear.', 'fc.').replace('shortcut.', 'downsample.'): v for k, v in sd.items()}
    model = torchvision.models.resnet18(num_classes=sd['fc.weight'].shape[0])
    model.load_state_dict(sd)
    return model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000, help='number of random test images')
    p.add_argument('--split', default='test')
    p.add_argument('--preproc', choices=['padcrop', 'ours'], default='padcrop')
    p.add_argument('--tgt_scale', type=float, default=0.0826, help='PadCropDiatom transform: target microns/pixel')
    p.add_argument('--scale_policy', choices=['none', 'csv'], default='none',
                   help="PadCropDiatom transform: 'csv' = rescale with diatoms.csv scale, 'none' = no rescaling")
    p.add_argument('--resize', type=int, default=240)
    p.add_argument('--crop', type=int, default=224)
    p.add_argument('--no_norm', action='store_true', help='skip ImageNet mean/std normalisation')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    classes = independent_classes()
    idx = {c: i for i, c in enumerate(classes)}
    with open(os.path.join(DIATLAS, 'diatoms.csv')) as f:
        meta = {r['path'].lstrip('/'): r for r in csv.DictReader(f)}
    species_of = {k: r['species_code'] for k, r in meta.items()}
    with open(os.path.join(REPO, 'data', 'diatlas', 'splits.csv')) as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == args.split and species_of[r['orig_path']] in idx]
    rows = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    y = torch.tensor([idx[species_of[r['orig_path']]] for r in rows])

    model = load_independent_resnet18()
    norm = [] if args.no_norm else [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if args.preproc == 'ours':
        tf = T.Compose([T.Resize(args.resize), T.CenterCrop(args.crop), T.ToTensor()] + norm)
        load = lambda r: tf(Image.open(os.path.join(REPO, 'data', 'diatlas', r['path'])).convert('RGB'))
    else:
        from padcrop_transform import pad_crop_diatom
        tf = T.Compose([T.Grayscale(3), T.ToTensor()] + norm)

        def load(r):
            src = float(meta[r['orig_path']]['scale']) / 100 if args.scale_policy == 'csv' else args.tgt_scale
            return tf(pad_crop_diatom(Image.open(os.path.join(DIATLAS, r['orig_path'])), src, args.tgt_scale))
    logits = []
    with torch.no_grad():
        for i in range(0, len(rows), 50):
            x = torch.stack([load(r) for r in rows[i:i + 50]])
            logits.append(model(x))
    logits = torch.cat(logits)
    top5 = logits.topk(5, dim=1).indices
    top1_ok = top5[:, 0] == y
    top5_ok = (top5 == y[:, None]).any(1)
    desc = (f'PadCropDiatom transform, tgt_scale {args.tgt_scale}, scale_policy {args.scale_policy}'
            if args.preproc == 'padcrop' else f'our images, resize {args.resize} crop {args.crop}')
    print(f'{len(rows)} {args.split} images, {len(classes)} classes, {desc}, norm {"off" if args.no_norm else "ImageNet"}')
    print(f'top-1 {top1_ok.float().mean():.3f}   top-5 {top5_ok.float().mean():.3f}   (chance {1 / len(classes):.4f})')

    # accuracy by how many images the species has in the whole dataset
    with open(os.path.join(DIATLAS, 'diatoms.csv')) as f:
        n_imgs = collections.Counter(r['species_code'] for r in csv.DictReader(f) if r['connective'] != 'True')
    for lo, hi in ((2, 20), (20, 100), (100, 10 ** 6)):
        sel = torch.tensor([lo <= n_imgs[classes[t]] < hi for t in y.tolist()])
        if sel.any():
            print(f'  species with {lo}-{hi if hi < 10 ** 6 else "max"} images: top-1 {top1_ok[sel].float().mean():.3f} '
                  f'(n={int(sel.sum())})')


if __name__ == '__main__':
    main()
