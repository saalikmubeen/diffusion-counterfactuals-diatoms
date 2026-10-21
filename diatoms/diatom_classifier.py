"""Load a diatom classifier for DVCE.

The returned module takes what DVCE gives a classifier: a batch of 256x256
native-size images in [0, 1] (data/diatlas_nativesize_crop224_canvas256). It
takes the center 224x224 crop (= the independent classifier's input), applies ImageNet
normalisation and returns logits over the 759 species_code classes.

  ckpt = 'independent'          the original independent ResNet-18 (non-robust)
  ckpt = path to best.pt etc.  a checkpoint saved by diatoms/train_classifier.py
"""
import os
import sys

import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifier import (NormalizedClassifier, ResizeToClassifier,  # noqa: E402
                              load_independent_resnet18, independent_classes)


class _ToModelDevice(nn.Module):
    """Moves the input to the model's device (DVCE scripts sometimes pass CPU tensors)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x.to(next(self.model.parameters()).device))


def load_diatom_classifier(ckpt):
    """Returns (model, class_names, description). The model is in eval mode on CPU."""
    if ckpt == 'independent':
        net = NormalizedClassifier(load_independent_resnet18())
        classes, desc = independent_classes(), 'independent ResNet-18 (original, non-robust)'
        mode = 'centercrop'
    else:
        ck = torch.load(ckpt, map_location='cpu')
        net = NormalizedClassifier(getattr(torchvision.models, ck.get('arch', 'resnet18'))(num_classes=ck['num_classes']))
        net.load_state_dict(ck['state_dict'])
        classes = ck['classes']
        mode = 'centercrop' if ck.get('preproc', 'nativesize') in ('nativesize', 'nativefit') else 'resize'
        desc = (f"{os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}: {ck.get('arch')}, "
                f"init {ck.get('init', 'independent').replace('supervisor', 'independent')}, eps {ck.get('eps_train_l2_256')}, epoch {ck.get('epoch')}")
    model = _ToModelDevice(ResizeToClassifier(net, mode=mode)).eval()
    return model, classes, desc
