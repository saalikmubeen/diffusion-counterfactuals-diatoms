"""DVCEs for DIATLAS diatoms. Copy of imagenet_VCEs.py with only the data side
changed: DIATLAS test split instead of ImageNet val, diatom class names, pairs
from a JSON list, output in DiatlasVCEs/. The DVCE optimisation is unchanged.

Pairs (--vce_list, required) are JSON like
  {"entries": [{"path": "256/test/NPAL/IDF/NPAL_012.png", "targets": ["NCTE", "NCRY"]},
               {"idx": 17, "targets": [3, 5]}]}
where path is relative to data/diatlas (or idx = index into the test set) and
targets are class names or class indices. Use two targets per image, as the
plotting code expects.

Classifier: --diatom_classifier <checkpoint from diatoms/train_classifier.py, or
'independent'> (optional --diatom_classifier_2 for cone projection). Labels are
then its 759 species_code classes and the data is --diatlas_root (default:
data/diatlas_nativesize_crop224_canvas256, which these classifiers need).
Without --diatom_classifier: --diatlas_label_key taxon|genus, --diatlas_min_count N
on the stretch-to-fit data (only useful with a matching classifier).
Diffusion prior: --diffusion_ckpt (default: ImageNet model).
"""
import json
import random

from torchvision.utils import save_image

from blended_diffusion.optimization import DiffusionAttack
from blended_diffusion.optimization.arguments import get_arguments
from configs import get_config
from utils_svces.datasets.diatlas import DiatlasDataset
import utils_svces.datasets as dl
from utils_svces.functions import blockPrint
import torch
import torch.nn as nn
import numpy as np
import os
import pathlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils_svces.load_trained_model import load_model
from tqdm import trange
from time import sleep
from utils_svces.train_types.helpers import create_attack_config, get_adversarial_attack

from utils_svces.Evaluator import Evaluator


hps = get_config(get_arguments())

if not hps.verbose:
    blockPrint()


if len(hps.gpu)==0 or str(hps.gpu[0]).lower() == 'cpu':  # --gpu cpu
    device = torch.device('cpu')
    print('Warning! Computing on CPU')
    hps.device_ids = ['cpu']  # one "device": only its count is used (batch size, no DataParallel)
    num_devices = 1
elif len(hps.gpu)==1:
    hps.device_ids = [int(hps.gpu[0])]
    device = torch.device('cuda:' + str(hps.gpu[0]))
    num_devices = 1
else:
    hps.device_ids = [int(i) for i in hps.gpu]
    device = torch.device('cuda:' + str(min(hps.device_ids)))
    num_devices = len(hps.device_ids)
hps.device = device


img_size = 256
num_imgs = hps.num_imgs
pixel_d_min = 5
conf_threshold = 0.01
out_dir = 'DiatlasVCEs'
dataset = 'diatlas'
imagenet_mode = (f'clf_{os.path.basename(os.path.dirname(hps.diatom_classifier)) or hps.diatom_classifier}'
                 f'__diff_{os.path.basename(os.path.dirname(hps.diffusion_ckpt))}'
                 if hps.diatom_classifier else f'{hps.diatlas_label_key}_min{hps.diatlas_min_count}')
bs = hps.batch_size * len(hps.device_ids)

torch.manual_seed(hps.seed)
random.seed(hps.seed)
np.random.seed(hps.seed)

if hps.diatom_classifier:
    # labels = the classifier's own classes (species_code), data = native-size copy
    from diatoms.diatom_classifier import load_diatom_classifier
    _, clf_classes, _ = load_diatom_classifier(hps.diatom_classifier)
    in_dataset = DiatlasDataset(root=hps.diatlas_root, split='test', label_key='species_code', classes=clf_classes)
    label_desc = f'species_code, classes of {hps.diatom_classifier}'
else:
    in_dataset = DiatlasDataset(split='test', label_key=hps.diatlas_label_key, min_count=hps.diatlas_min_count)
    label_desc = f'{hps.diatlas_label_key}, min {hps.diatlas_min_count}'
in_labels = in_dataset.classes
print(f'DIATLAS test set ({in_dataset.root}): {len(in_dataset)} images, {len(in_labels)} classes ({label_desc})')

if not hps.vce_list:
    raise SystemExit('diatlas_VCEs.py needs --vce_list <pairs.json> (see the docstring for the format)')


def _class_idx(t):
    if isinstance(t, str):
        if t not in in_dataset.class_to_idx:
            raise SystemExit(f'target class {t!r} is not in the label set ({label_desc})')
        return in_dataset.class_to_idx[t]
    return int(t)


some_vces = {}
with open(hps.vce_list) as f:
    for e in json.load(f)['entries']:
        idx = in_dataset.path_to_idx[e['path']] if 'path' in e else int(e['idx'])
        some_vces[idx] = [_class_idx(t) for t in e['targets']]


def _plot_counterfactuals(dir, original_imgs, orig_labels, segmentations, targets,
                          perturbed_imgs, perturbed_probabilities, original_probabilities, radii, class_labels, filenames=None, img_idcs=None, num_plot_imgs=hps.num_imgs):
    num_imgs = num_plot_imgs
    num_radii = len(radii)
    scale_factor = 4.0
    target_idx = 0


    if img_idcs is None:
        img_idcs = torch.arange(num_imgs, dtype=torch.long)

    pathlib.Path(dir+'/single_images').mkdir(parents=True, exist_ok=True)

    # Two VCEs per starting image - we fix them to 2
    num_VCEs_per_image = 2
    for lin_idx in trange(int(len(img_idcs)/num_VCEs_per_image), desc=f'Image write'):

        # we fix only one radius
        radius_idx = 0
        lin_idx *= num_VCEs_per_image

        img_idx = img_idcs[lin_idx]




        in_probabilities = original_probabilities[img_idx, target_idx, radius_idx]


        pred_original = in_probabilities.argmax()
        pred_value = in_probabilities.max()

        num_rows = 1
        num_cols = num_VCEs_per_image + 1
        fig, ax = plt.subplots(num_rows, num_cols,
                               figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))
        img_label = orig_labels[img_idx]
        title = f'GT: {class_labels[img_label]}' #, predicted: {class_labels[pred_original]},{pred_value:.2f}'

        img_segmentation = segmentations[img_idx]
        bin_segmentation = torch.sum(img_segmentation, dim=0) > 0.0
        img_segmentation[:, bin_segmentation] = 0.5
        mask_color = torch.zeros_like(img_segmentation)
        mask_color[1, :, :] = 1.0

        # plot original:
        ax[0].axis('off')
        ax[0].set_title(title)
        img_original = original_imgs[img_idx, :].permute(1, 2, 0).cpu().detach()
        ax[0].imshow(img_original, interpolation='lanczos')

        save_image(original_imgs[img_idx, :].clip(0, 1),
                   os.path.join(dir, 'single_images', f'{img_idx}_original.png'))
        for i in range(num_VCEs_per_image):

            img = torch.clamp(perturbed_imgs[img_idx+i, target_idx, radius_idx].permute(1, 2, 0), min=0.0,
                              max=1.0)
            img_probabilities = perturbed_probabilities[img_idx+i, target_idx, radius_idx]

            img_target = targets[img_idx+i]
            target_original = in_probabilities[img_target]

            target_conf = img_probabilities[img_target]

            ax[i+1].axis('off')
            ax[i+1].imshow(img, interpolation='lanczos')

            title = f'{class_labels[img_target]}: {target_conf:.2f}, i:{target_original:.2f}'
            ax[i+1].set_title(title)


            save_image(perturbed_imgs[img_idx+i, target_idx, radius_idx].clip(0, 1), os.path.join(dir, 'single_images', f'{img_idx}_{class_labels[img_target]}.png'))

        plt.tight_layout()
        if filenames is not None:
            fig.savefig(os.path.join(dir, f'{filenames[lin_idx]}.png'))
            fig.savefig(os.path.join(dir, f'{filenames[lin_idx]}.pdf'))
        else:
            fig.savefig(os.path.join(dir, f'{lin_idx}.png'))
            fig.savefig(os.path.join(dir, f'{lin_idx}.pdf'))

        plt.close(fig)

plot = False
plot_top_imgs = True

imgs = torch.zeros((num_imgs, 3, img_size, img_size))
segmentations = torch.zeros((num_imgs, 3, img_size, img_size))
targets_tensor = torch.zeros(num_imgs, dtype=torch.long)
labels_tensor = torch.zeros(num_imgs, dtype=torch.long)
dataset_idcs_tensor = torch.zeros(num_imgs, dtype=torch.long)
filenames = []

image_idx = 0
kernel = np.ones((5, 5), np.uint8)

selected_vces = list(some_vces.items())


if hps.world_size > 1:
    print('Splitting relevant classes')
    print(f'{hps.world_id} out of {hps.world_size}')
    splits = np.array_split(np.arange(len(selected_vces)), hps.world_size)
    print(f'Using clusters {splits[hps.world_id]} out of {len(targets_tensor)}')

for i, (img_idx, target_classes) in enumerate(selected_vces):
    if hps.world_size > 1 and i not in splits[hps.world_id]:
        pass
    else:
        in_image, label = in_dataset[img_idx]
        for i in range(len(target_classes)):
            targets_tensor[image_idx+i] = target_classes[i]
            labels_tensor[image_idx+i] = label
            dataset_idcs_tensor[image_idx+i] = img_idx
            imgs[image_idx+i] = in_image
        image_idx += len(target_classes)
        if image_idx >= num_imgs:
            break

imgs = imgs[:image_idx]
segmentations = segmentations[:image_idx]
targets_tensor = targets_tensor[:image_idx]

use_diffusion = False
for method in [hps.method]:
    if method.lower() == 'svces':
        radii = np.array([150.])
        attack_type = 'afw'
        norm = 'L1.5'
        stepsize = None
        steps = 75
    elif method.lower() == 'apgd':
        radii = np.array([12.])
        attack_type = 'apgd'
        norm = 'L2'
        stepsize = None
        steps = 75
    elif method.lower() == 'dvces':
        attack_type = 'diffusion'
        radii = np.array([0.])
        norm = 'L2'
        steps = 150
        stepsize = None
        use_diffusion=True
    else:
        raise NotImplementedError()


    attack_config = create_attack_config(eps=radii[0], steps=steps, stepsize=stepsize, norm=norm, momentum=0.9,
                                         pgd=attack_type)

    num_classes = len(in_labels)
    if attack_type == 'diffusion':
        img_dimensions = (3, 256, 256)
    else:
        img_dimensions = imgs.shape[1:]
    num_targets = 1
    num_radii = len(radii)
    num_imgs = len(imgs)

    with torch.no_grad():

        model_bs = bs
        dir = f'{out_dir}/{imagenet_mode}/{norm}_{hps.l2_sim_lambda}_l1_{hps.l1_sim_lambda}_l{hps.lp_custom}_{hps.lp_custom_value}_classifier_{hps.classifier_type}_{hps.second_classifier_type}_{hps.third_classifier_type}_{hps.classifier_lambda}_reg_lpips_{hps.lpips_sim_lambda}_example_{hps.timestep_respacing}_steps_skip_{hps.skip_timesteps}_start_{hps.gen_type}_deg_{str(hps.deg_cone_projection)}_s_{str(hps.seed)}{"_bl" if hps.use_blended else ""}_wid_{hps.world_id}_{hps.world_size}_{hps.method}/'
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

        out_imgs = torch.zeros((num_imgs, num_targets, num_radii) + img_dimensions)
        out_probabilities = torch.zeros((num_imgs, num_targets, num_radii, num_classes))
        in_probabilities = torch.zeros((num_imgs, num_targets, num_radii, num_classes))
        model_original_probabilities = torch.zeros((num_imgs, num_classes))

        n_batches = int(np.ceil(num_imgs / model_bs))


        if use_diffusion or attack_config['pgd'] in ['afw', 'apgd']:
            if use_diffusion:
                att = DiffusionAttack(hps)
            else:
                loss = 'log_conf' if attack_config['pgd'] == 'afw' else 'ce-targeted-cfts'
                print('using loss', loss)
                model = None
                att = get_adversarial_attack(attack_config, model, loss, num_classes,
                                             args=hps, Evaluator=Evaluator)
            if att.args.second_classifier_type != -1:
                print('setting model to second classifier')
                model = att.second_classifier
            else:
                model = att.classifier
        else:
            model = None

        n_out = model(imgs[:1].to(device)).shape[1]
        if n_out != num_classes:
            raise SystemExit(f'classifier has {n_out} outputs but the diatom label set has {num_classes} classes: '
                             'load a diatom classifier (DIATOM_PLAN.md section 5)')

        for batch_idx in trange(n_batches, desc=f'Batches progress'):
            sleep(0.1)
            batch_start_idx = batch_idx * model_bs
            batch_end_idx = min(num_imgs, (batch_idx + 1) * model_bs)

            batch_data = imgs[batch_start_idx:batch_end_idx, :]
            batch_targets = targets_tensor[batch_start_idx:batch_end_idx]
            print('batch segmentations before', segmentations.shape)
            batch_segmentations = segmentations[batch_start_idx:batch_end_idx, :]
            print('batch segmentations after', batch_segmentations.shape)
            target_idx = 0

            orig_out = model(batch_data)
            with torch.no_grad():
                orig_confidences = torch.softmax(orig_out, dim=1)
                model_original_probabilities[batch_start_idx:batch_end_idx, :] = orig_confidences.detach().cpu()

            for radius_idx in range(len(radii)):
                batch_data = batch_data.to(device)
                batch_targets = batch_targets.to(device)


                if not use_diffusion:
                    att.eps = radii[radius_idx]

                if use_diffusion:
                    batch_adv_samples_i = att.perturb(batch_data,
                                                            batch_targets, dir)[
                        0].detach()
                else:
                    if attack_config['pgd'] in ['afw']:

                        batch_adv_samples_i = att.perturb(batch_data,
                                                                batch_targets,
                                                                targeted=True).detach()



                    else:
                        batch_adv_samples_i = att.perturb(batch_data,
                                                                batch_targets,
                                                                best_loss=True)[0].detach()
                out_imgs[batch_start_idx:batch_end_idx, target_idx, radius_idx,
                :] = batch_adv_samples_i.cpu().detach()

                batch_model_out_i = model(batch_adv_samples_i)
                batch_model_in_i = model(batch_data)
                batch_probs_i = torch.softmax(batch_model_out_i, dim=1)
                batch_probs_in_i = torch.softmax(batch_model_in_i, dim=1)

                out_probabilities[batch_start_idx:batch_end_idx, target_idx, radius_idx,
                :] = batch_probs_i.cpu().detach()
                in_probabilities[batch_start_idx:batch_end_idx, target_idx, radius_idx,
                :] = batch_probs_in_i.cpu().detach()

            if (batch_idx + 1) % hps.plot_freq == 0 or batch_idx == n_batches-1:
                data_dict = {}

                data_dict['gt_imgs'] = imgs[:batch_end_idx]
                data_dict['gt_labels'] = labels_tensor[:batch_end_idx]
                data_dict['dataset_idcs'] = dataset_idcs_tensor[:batch_end_idx]
                data_dict['segmentations'] = segmentations[:batch_end_idx]
                data_dict['targets'] = targets_tensor[:batch_end_idx]
                data_dict['counterfactuals'] = out_imgs[:batch_end_idx]
                data_dict['out_probabilities'] = out_probabilities[:batch_end_idx]
                data_dict['in_probabilities'] = in_probabilities[:batch_end_idx]
                data_dict['radii'] = radii
                torch.save(data_dict, os.path.join(dir, f'{num_imgs}.pth'))
                _plot_counterfactuals(dir, imgs[:batch_end_idx], labels_tensor, segmentations[:batch_end_idx],
                                      targets_tensor[:batch_end_idx],
                                      out_imgs[:batch_end_idx], out_probabilities[:batch_end_idx], in_probabilities[:batch_end_idx], radii, in_labels, filenames=None, num_plot_imgs=len(imgs[:batch_end_idx]))
