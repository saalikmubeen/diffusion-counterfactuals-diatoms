"""Fine-tune the independent classifier's diatom ResNet-18 on our preprocessed DIATLAS images,
optionally with adversarial (L2 PGD) training to make it robust for DVCE.

Input pipeline = what DVCE gives the classifier: our 256x256 grayscale images
(data/diatlas/256, as 3 channels in [0, 1]) -> bicubic resize to 224 -> model.
ImageNet mean/std normalisation is built into the saved model, like the robust
ImageNet models DVCE uses. Adversarial perturbations are made at 256x256 in
[0, 1] space, the space in which DVCE edits images.

Classes: the independent classifier's 759 species_code classes (alphabetical). --init
chooses the starting weights: 'independent' (the independent diatom ResNet-18, head reused),
'madry_r50' (robust ImageNet ResNet-50 from the DVCE paper, new head), or
standard ImageNet ResNet-18/50 (new head). Data: our train split for training,
val split for model selection; the test split is never touched here.

  --eps 0      standard fine-tuning (non-robust)
  --eps 1.0    adversarial training, L2 radius 1.0 at 256x256

Outputs in DiatlasModels/<run>/: best.pt (best val accuracy for eps 0, best
val robust accuracy otherwise), last.pt, log.txt (one line per epoch with ETA).

Example (one GPU per run):
  CUDA_VISIBLE_DEVICES=0 python diatoms/train_classifier.py --run resnet18_ft_eps0   --eps 0
  CUDA_VISIBLE_DEVICES=1 python diatoms/train_classifier.py --run resnet18_ft_eps1.0 --eps 1.0
  CUDA_VISIBLE_DEVICES=0 python diatoms/train_classifier.py --run r50madry_native_eps1.0 --init madry_r50 --eps 1.0
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_independent_classifier import (DIATLAS, IMAGENET_MEAN, IMAGENET_STD,  # noqa: E402
                                        load_independent_resnet18, independent_classes)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data', 'diatlas')
DATA_NATIVE = os.path.join(REPO, 'data', 'diatlas_nativesize_crop224_canvas256')


class NormalizedClassifier(nn.Module):
    """[0, 1] RGB input at 224 -> ImageNet normalisation -> ResNet-18."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer('std', torch.tensor(IMAGENET_STD)[None, :, None, None])

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class ResizeToClassifier(nn.Module):
    """How DVCE turns its 256x256 image into the classifier input.
    mode 'resize': transforms.Resize(224), bicubic (ResizeAndMeanWrapper, stretch-to-fit data)
    mode 'centercrop': center 224x224 crop (native-size data, = the independent classifier's input)"""

    def __init__(self, model, size=224, mode='resize'):
        super().__init__()
        self.model = model
        self.resize = (transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
                       if mode == 'resize' else transforms.CenterCrop(size))

    def forward(self, x):
        return self.model(self.resize(x))


MADRY_R50 = os.path.join(REPO, 'ImageNet1000Models', 'Madry_l2_improved', 'checkpoint', 'ep_3.pt')


def random_split_of(path, val=0.1, test=0.1):
    """Random image-level split (80/10/10), fixed per image via a hash of its path.
    Photos from the same atlas page can land in different splits, so test accuracy
    measures recognition of seen pages (like a usual random split), not new pages."""
    import hashlib
    u = int(hashlib.md5(path.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return 'test' if u < test else 'val' if u < test + val else 'train'


def get_label_set(name):
    """Returns (classes, label_key) for --labels:
    independent759   his 759 species_code classes (incl. the empty genus-only class)
    species_pagesN  species_code with >= N distinct (atlas, page) groups in our train split
    genus_trainN    genus_code with >= N train images (genus-only images included)"""
    if name == 'independent759':
        return independent_classes(), 'species_code'
    import collections
    import re
    with open(os.path.join(DATA_NATIVE, 'splits.csv')) as f:
        train = [r for r in csv.DictReader(f) if r['split'] == 'train']
    m = re.fullmatch(r'species_pages(\d+)', name)
    if m:
        pages = collections.defaultdict(set)
        for r in train:
            if r['species_code'].strip():
                pages[r['species_code']].add((r['source'], r['page']))
        return sorted(c for c, pg in pages.items() if len(pg) >= int(m.group(1))), 'species_code'
    m = re.fullmatch(r'genus_train(\d+)', name)
    if m:
        n = collections.Counter(r['genus'] for r in train)
        return sorted(g for g, k in n.items() if k >= int(m.group(1))), 'genus_code'
    raise ValueError(f'unknown label set {name}')


def build_network(init, num_classes):
    """init 'independent': the independent diatom ResNet-18 (759 species, head kept).
    'madry_r50': robust ImageNet ResNet-50 (Madry L2 + FT, the DVCE paper's model), new head.
    'imagenet_r18' / 'imagenet_r50': standard torchvision ImageNet weights, new head."""
    import torchvision
    if init == 'independent':
        net = load_independent_resnet18()
        if num_classes != net.fc.out_features:
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net, 'resnet18'
    if init == 'madry_r50':
        net = torchvision.models.resnet50()
        net.load_state_dict(torch.load(MADRY_R50, map_location='cpu'))
        arch = 'resnet50'
    else:
        arch = {'imagenet_r18': 'resnet18', 'imagenet_r50': 'resnet50'}[init]
        net = getattr(torchvision.models, arch)(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net, arch


class DiatlasSpecies(Dataset):
    """preproc 'ours': data/diatlas/256 images (stretch-to-fit, 256x256).
    preproc 'nativesize': data/diatlas_nativesize_crop224_canvas256 (256x256 canvas,
    native pixel size; classifier gets the center 224 crop).
    preproc 'padcrop': raw DIATLAS image -> the PadCropDiatom (physical size
    kept, no rescaling, tgt 224) -> 224x224.
    keep_empty=False drops images without species_code (genus-only IDs); the
    empty class stays in the 759 outputs but gets no training images.
    label_key: 'species_code' or 'genus_code' (diatoms.csv column used as label);
    images whose label is not in `classes` are left out."""

    def __init__(self, split, classes, augment, preproc='ours', keep_empty=False, tgt_scale=0.0826,
                 label_key='species_code', split_mode='page'):
        idx = {c: i for i, c in enumerate(classes)}
        with open(os.path.join(DIATLAS, 'diatoms.csv')) as f:
            species_of = {r['path'].lstrip('/'): r[label_key] for r in csv.DictReader(f)}
        with open(os.path.join(DATA_NATIVE if preproc == 'nativesize' else DATA, 'splits.csv')) as f:
            rows = list(csv.DictReader(f))
        if split_mode == 'random':
            for r in rows:
                r['split'] = random_split_of(r['orig_path'])
        if True:
            rows = [r for r in rows if r['split'] == split and species_of[r['orig_path']] in idx
                    and (keep_empty or species_of[r['orig_path']].strip())]
        rows.sort(key=lambda r: r['path'])
        if preproc == 'ours':
            self.paths = [os.path.join(DATA, r['path']) for r in rows]
        elif preproc == 'nativesize':
            self.paths = [os.path.join(DATA_NATIVE, r['path']) for r in rows]
        else:
            self.paths = [os.path.join(DIATLAS, r['orig_path']) for r in rows]
        self.targets = [idx[species_of[r['orig_path']]] for r in rows]
        self.augment = augment
        self.preproc = preproc
        self.tgt_scale = tgt_scale

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i])
        if self.preproc == 'padcrop':
            from padcrop_transform import pad_crop_diatom
            img = pad_crop_diatom(img, self.tgt_scale, self.tgt_scale)
        if self.augment == 'strong':
            img = strong_augment(img)
        x = torch.from_numpy(np.asarray(img.convert('RGB')).copy()).permute(2, 0, 1).float() / 255
        if self.augment:
            if torch.rand(1) < 0.5:
                x = x.flip(2)
            if torch.rand(1) < 0.5:
                x = x.flip(1)
        return x, self.targets[i]


def strong_augment(img):
    """Mild geometric + photometric jitter (applied to the 256 canvas, before the
    center crop): rotation +-10 deg, shift +-8 px, scale 0.9-1.1, brightness +-15%,
    contrast +-25%. Uncovered areas are filled with the image's border median, so
    no artificial edges appear. Scale stays small so size information survives."""
    import torchvision.transforms.functional as TF
    g = img.convert('L')
    a = np.asarray(g)
    fill = int(np.median(np.concatenate([a[:2].ravel(), a[-2:].ravel(), a[:, :2].ravel(), a[:, -2:].ravel()])))
    angle = float(np.random.uniform(-10, 10))
    shift = [int(np.random.randint(-8, 9)), int(np.random.randint(-8, 9))]
    scale = float(np.random.uniform(0.9, 1.1))
    g = TF.affine(g, angle=angle, translate=shift, scale=scale, shear=[0.0], fill=fill,
                  interpolation=transforms.InterpolationMode.BILINEAR)
    g = TF.adjust_brightness(g, float(np.random.uniform(0.85, 1.15)))
    g = TF.adjust_contrast(g, float(np.random.uniform(0.75, 1.25)))
    return g


def pgd_l2(model, x, y, eps, steps, step_size=None, amp=False):
    """Untargeted L2 PGD in [0, 1] image space, random start inside the ball."""
    if eps <= 0:
        return x
    step_size = step_size or 2.5 * eps / steps
    delta = torch.randn_like(x)
    delta = delta / delta.flatten(1).norm(dim=1).view(-1, 1, 1, 1) * eps * torch.rand(len(x), 1, 1, 1, device=x.device)
    delta = ((x + delta).clamp(0, 1) - x).detach()
    for _ in range(steps):
        delta.requires_grad_(True)
        # fp16 autocast: scale the loss (like GradScaler) so the small input
        # gradients of a confident model don't underflow to zero; the scale
        # doesn't matter because the gradient is normalised below. Fall back to
        # fp32 if the scaled gradient overflows.
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x + delta)
        loss = F.cross_entropy(logits.float(), y, reduction='sum') * (1024.0 if amp else 1.0)
        grad, = torch.autograd.grad(loss, delta)
        if amp and not torch.isfinite(grad).all():
            loss = F.cross_entropy(model(x + delta).float(), y, reduction='sum')
            grad, = torch.autograd.grad(loss, delta)
        grad = grad / (grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
        delta = delta.detach() + step_size * grad
        norm = delta.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        delta = delta * torch.clamp(eps / (norm + 1e-12), max=1.0)
        delta = ((x + delta).clamp(0, 1) - x).detach()
    return (x + delta).detach()


def evaluate(model, loader, eps, steps, robust_loader=None):
    """Clean accuracy on all of `loader`; robust accuracy (PGD) on `robust_loader`,
    a fixed random subset of val (the val set is sorted by species, so its first
    images would be a biased sample)."""
    model.eval()
    correct = total = rob_correct = rob_total = 0
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast():
            correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    if eps > 0 and robust_loader is not None:
        torch.manual_seed(1234)  # same PGD random start every evaluation
        for x, y in robust_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            x_adv = pgd_l2(model, x, y, eps, steps)
            with torch.no_grad(), torch.cuda.amp.autocast():
                rob_correct += (model(x_adv).argmax(1) == y).sum().item()
            rob_total += len(y)
    return correct / total, (rob_correct / rob_total if rob_total else float('nan'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, help='output folder name under DiatlasModels/')
    p.add_argument('--eps', type=float, default=1.0, help='L2 radius at 256x256 in [0,1]; 0 = standard training')
    p.add_argument('--attack_steps', type=int, default=5, help='PGD steps during training')
    p.add_argument('--eval_eps', type=float, default=None, help='radius for val robust accuracy (default: --eps)')
    p.add_argument('--eval_steps', type=int, default=10)
    p.add_argument('--eval_robust_n', type=int, default=1024, help='val images used for robust accuracy')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--preproc', choices=['nativesize', 'ours', 'padcrop'], default='nativesize',
                   help="'nativesize' = data/diatlas_nativesize_crop224_canvas256, classifier gets the center 224 "
                        "crop (DVCE input); 'ours' = stretch-to-fit 256, resized; 'padcrop' = PadCropDiatom transform on the fly")
    p.add_argument('--keep_empty', action='store_true', help='keep images without species_code (genus-only)')
    p.add_argument('--split_mode', choices=['page', 'random'], default='page',
                   help="'page' = our page-grouped split; 'random' = random image split 80/10/10")
    p.add_argument('--labels', default='independent759',
                   help="independent759 | species_pagesN (e.g. species_pages3) | genus_trainN (e.g. genus_train100)")
    p.add_argument('--augment', choices=['flip', 'strong'], default='flip',
                   help="'flip' = random h/v flips; 'strong' = + rotation, shift, scale, brightness/contrast")
    p.add_argument('--init', choices=['independent', 'madry_r50', 'imagenet_r18', 'imagenet_r50'], default='independent',
                   help='starting weights (see build_network); all use the same 759 species classes')
    args = p.parse_args()
    eval_eps = args.eps if args.eval_eps is None else args.eval_eps
    torch.manual_seed(args.seed)

    out_dir = os.path.join(REPO, 'DiatlasModels', args.run)
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, 'log.txt'), 'a')

    def log(msg):
        line = f'[{datetime.now():%Y-%m-%d %H:%M}] {msg}'
        print(line, flush=True)
        log_f.write(line + '\n')
        log_f.flush()

    classes, label_key = get_label_set(args.labels)
    keep_empty = args.keep_empty or label_key == 'genus_code'
    train_ds = DiatlasSpecies('train', classes, args.augment, args.preproc, keep_empty, label_key=label_key,
                              split_mode=args.split_mode)
    val_ds = DiatlasSpecies('val', classes, False, args.preproc, keep_empty, label_key=label_key,
                            split_mode=args.split_mode)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    rob_idx = torch.randperm(len(val_ds), generator=torch.Generator().manual_seed(0))[:args.eval_robust_n].tolist()
    rob_dl = DataLoader(torch.utils.data.Subset(val_ds, rob_idx), args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    log(f'args {json.dumps(vars(args))}')
    log(f'{len(classes)} classes; train {len(train_ds)} images, val {len(val_ds)} images')

    net, arch = build_network(args.init, len(classes))
    model = ResizeToClassifier(NormalizedClassifier(net),
                               mode='centercrop' if args.preproc == 'nativesize' else 'resize').cuda()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    steps_total = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps_total)))
    scaler = torch.cuda.amp.GradScaler()

    def save(name, epoch, val_acc, val_rob):
        torch.save({'state_dict': model.model.state_dict(),  # NormalizedClassifier: net.* + mean/std
                    'classes': classes, 'label_key': label_key, 'labels': args.labels, 'split_mode': args.split_mode, 'arch': arch, 'init': args.init, 'num_classes': len(classes),
                    'input': ('[0,1] RGB, 224x224 = center crop of the 256 native-size image'
                              if args.preproc == 'nativesize' else '[0,1] RGB, 224x224 (256 resized, bicubic)')
                             + '; ImageNet mean/std inside', 'preproc': args.preproc,
                    'eps_train_l2_256': args.eps, 'epoch': epoch, 'val_acc': val_acc, 'val_robust_acc': val_rob,
                    'eval_eps': eval_eps, 'args': vars(args)}, os.path.join(out_dir, name))

    acc0, rob0 = evaluate(model, val_dl, eval_eps, args.eval_steps, rob_dl)
    log(f'epoch 0 (starting weights "{args.init}", before training): val acc {acc0:.4f}  val robust acc (eps {eval_eps}) {rob0:.4f}')
    best = -1.0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            if args.eps > 0:
                model.eval()  # attack with fixed BN statistics
                x = pgd_l2(model, x, y, args.eps, args.attack_steps)
                model.train()
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        val_acc, val_rob = evaluate(model, val_dl, eval_eps, args.eval_steps, rob_dl)
        score = val_acc if args.eps <= 0 else val_rob
        if score > best:
            best = score
            save('best.pt', epoch, val_acc, val_rob)
        save('last.pt', epoch, val_acc, val_rob)
        elapsed = time.time() - t_start
        eta = elapsed / epoch * (args.epochs - epoch)
        log(f'epoch {epoch}/{args.epochs} | train loss {loss_sum / seen:.4f} | train acc{" (adv)" if args.eps > 0 else ""} '
            f'{correct / seen:.4f} | val acc {val_acc:.4f} | val robust acc (eps {eval_eps}) {val_rob:.4f} | '
            f'lr {sched.get_last_lr()[0]:.2e} | {(time.time() - t_ep) / 60:.1f} min/epoch | '
            f'ETA {eta / 3600:.1f} h (finishes ~{datetime.now() + timedelta(seconds=eta):%a %H:%M})'
            f'{" | new best" if score == best else ""}')
    log(f'training finished; best {"val acc" if args.eps <= 0 else "val robust acc"} {best:.4f} -> {out_dir}/best.pt')


if __name__ == '__main__':
    main()
