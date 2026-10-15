# Reproducible DVCE setup and run guide

This is a practical, end-to-end guide for running the authors' official code
for **Diffusion Visual Counterfactual Explanations (DVCE)** on ImageNet.  It
includes the issues found while bringing the repository up on an RTX 3090.

The goal is a faithful **ImageNet smoke run first**.  Do not start adapting the
method to another dataset until that run completes and its outputs have been
saved.

> **Scope and honesty.** The commands below have been verified through (1)
> loading ImageNet metadata and an actual validation image and (2) executing a
> CUDA tensor on an RTX 3090 with PyTorch 1.9.0+cu111.  A full DVCE generation
> is only a successful reproduction once its command finishes and produces
> output images.  Runtime, output quality, and numerical equality are not
> guaranteed by an installation guide.

## 1. What this project runs

DVCE starts from a real ImageNet image and uses a pretrained diffusion model
plus classifier gradients to generate an image that a classifier assigns to a
chosen target class.  The provided script uses six fixed ImageNet examples and
two targets per source image unless the source code is changed.

For the default DVCE configuration, the required assets are:

| Asset | Required location relative to the DVCE checkout | Why it is needed |
| --- | --- | --- |
| Official source | current directory | Sampling and guidance code |
| OpenAI 256×256 unconditional diffusion checkpoint | `checkpoints/256x256_diffusion_uncond.pt` | The diffusion prior |
| Madry + fine-tuned robust ImageNet classifier | `ImageNet1000Models/Madry_l2_improved/checkpoint/ep_3.pt` | Default classifier/guidance model |
| ImageNet ILSVRC-2012 train and validation data | external data root, described below | Source images and labels |
| ImageNet metadata | `<imagenet-root>/meta.bin` | Maps ImageNet synsets to the 1,000 labels |

The code is not a downloader for ImageNet.  You must have legitimate ImageNet
access and obey its license.

## 2. Prerequisites

Use a **Linux x86_64** machine with:

- Conda (or Miniconda) and a shell.
- An NVIDIA GPU and a working NVIDIA driver.
- At least one accessible GPU. An RTX 3090 (24 GB) works with the compatibility
  environment below.
- Enough disk space for ImageNet, the 2.1 GB diffusion checkpoint, and output.
- Authorized access to ImageNet 2012 and to the Madry + FT checkpoint linked
  by the authors.

The authors pinned Python 3.8, PyTorch 1.9.0, torchvision 0.10.0, and CUDA
10.2. Those original CUDA 10.2 wheels **do not contain support for RTX 3090
(sm_86)**.  On an RTX 30-series GPU, create the original environment and then
replace only the PyTorch binary trio with their matching CUDA 11.1 wheels as
shown in section 5. Keep this environment separate from the original one.

Do not use macOS/Apple Silicon for this exact GPU workflow.  CPU execution is
implemented nominally in the script but is not a practical or validated way to
run gradient-guided diffusion sampling.

## 3. Obtain a known source revision

Work in a fresh clone so that the authors' source remains identifiable.  The
audited revision is `2de46691856841e27a1db96dadbe8195a9d34690`.

```bash
git clone https://github.com/valentyn1boreiko/DVCEs.git
cd DVCEs
git checkout 2de46691856841e27a1db96dadbe8195a9d34690
git status --short
git rev-parse HEAD
```

`git status --short` should initially print nothing, and the final command
should print the commit above.  Save that hash in every experiment log.

Create a place for logs now:

```bash
mkdir -p logs checkpoints
```

## 4. Download and verify the model assets

### 4.1 Diffusion checkpoint

From the DVCE checkout:

```bash
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt \
  -O checkpoints/256x256_diffusion_uncond.pt
ls -lh checkpoints/256x256_diffusion_uncond.pt
sha256sum checkpoints/256x256_diffusion_uncond.pt
```

The file should be roughly 2.1 GB.  Record the displayed SHA-256 in your log;
do not silently replace this checkpoint with a newer diffusion model.

### 4.2 Robust ImageNet classifier

Download `MadryFT.zip` from the Google Drive link in the
[official README](https://github.com/valentyn1boreiko/DVCEs#setup), place the
archive in the DVCE checkout, and unpack it:

```bash
unzip MadryFT.zip
test -f ImageNet1000Models/Madry_l2_improved/checkpoint/ep_3.pt \
  && echo 'Madry + FT checkpoint: PASS' \
  || echo 'Madry + FT checkpoint: MISSING OR IN THE WRONG DIRECTORY'
ls -lh ImageNet1000Models/Madry_l2_improved/checkpoint/ep_3.pt
```

If the test prints the failure message, stop and inspect the archive layout.
Do not move a guessed `.pt` file into place: the default code expects the
exact path above.

## 5. Create the Python environment

### 5.1 Preserve the authors' original environment

From the DVCE checkout:

```bash
conda env create -f python_38_dvces.yml
conda activate python_38_dvces
python -m pip install git+https://github.com/RobustBench/robustbench.git
```

This produces the author-pinned reference environment.  On an RTX 3090 it is
useful as a dependency reference but its original CUDA 10.2 PyTorch wheel will
fail at GPU execution with `no kernel image is available for execution on the
device`.

### 5.2 RTX 3090 / Ampere compatibility environment

Clone the reference environment.  This preserves a clean comparison point:

```bash
conda create -n python_38_dvces_cu111 --clone python_38_dvces
conda activate python_38_dvces_cu111
```

Install the three matching **official PyTorch 1.9.0 CUDA 11.1 wheels**.  The
`--no-deps` flag intentionally avoids a broad, unrecorded dependency upgrade:

```bash
python -m pip install --force-reinstall --no-deps \
  --index-url https://download.pytorch.org/whl/cu111 \
  torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0
```

Do **not** instead run a generic `conda install cudatoolkit=11.3 ...` command
unless its dry-run proves that it keeps CUDA-enabled PyTorch.  A common solver
result installs `pytorch ... cpu_0`, which defeats the purpose.

Verify the actual binary, not only package names:

```bash
python - <<'PY'
import torch

print('PyTorch:', torch.__version__)
print('Built with CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'
print('GPU:', torch.cuda.get_device_name(0))
print('Architectures:', torch.cuda.get_arch_list())
assert 'sm_86' in torch.cuda.get_arch_list(), 'RTX 3090 support (sm_86) is missing'
x = torch.randn(1024, 1024, device='cuda')
y = torch.relu(x)
torch.cuda.synchronize()
print('CUDA tensor test: PASS', tuple(y.shape))
PY
```

For an RTX 3090, the key expected facts are `1.9.0+cu111`, CUDA `11.1`, and
`sm_86` in the architecture list.  If any assertion fails, do not run DVCE.

### 5.3 RobustBench warning

Run this informational check once:

```bash
python -m pip check
```

The current moving RobustBench package can report version conflicts with the
authors' old pins (for example `timm`, `pandas`, `requests`, or `tqdm`).  Do
**not** “fix all” these warnings automatically.  In particular, DVCE pins
`timm==0.5.4`; changing it changes the dependency stack.  Record the warnings,
then proceed only if the smoke run below reaches model loading.  If a traceback
names one of these packages, diagnose that specific failure and record any
minimal change.

## 6. Prepare ImageNet exactly as torchvision expects

Choose a parent directory.  In this example, `DATA_PARENT` is
`/home/GTL/smubeen/imagenet_data`; therefore the required ImageNet root is
`/home/GTL/smubeen/imagenet_data/imagenet`.

```bash
export DATA_PARENT=/home/GTL/smubeen/imagenet_data
export IMAGENET_ROOT="$DATA_PARENT/imagenet"
```

The final directory layout must be:

```text
$IMAGENET_ROOT/
├── meta.bin
├── train/
│   ├── n01440764/
│   └── ...                             # 1,000 synset directories
└── val/
    ├── n01440764/
    └── ...                             # 1,000 synset directories
```

`meta.bin` is not passed on the command line.  `torchvision.datasets.ImageNet`
automatically reads it when `root=$IMAGENET_ROOT`.  It is essential because it
contains the ImageNet class/synset metadata.

### 6.1 Build `meta.bin` if it is absent

`torchvision==0.10.0` expects the official ImageNet 2012 devkit archive
`ILSVRC2012_devkit_t12.tar.gz` to be available in `$IMAGENET_ROOT` when it has
to construct metadata.  Download that archive from the ImageNet site using
your authorized account, place it there, then run:

```bash
python - <<'PY'
from pathlib import Path
from torchvision.datasets import ImageNet

root = Path('/home/GTL/smubeen/imagenet_data/imagenet')
ImageNet(root=str(root), split='val')
print('meta.bin exists:', (root / 'meta.bin').is_file())
PY
```

Use your own absolute path if it differs.  If torchvision reports that the
devkit archive is missing or corrupted, it is telling you precisely what to
obtain.  Do not create a fake `meta.bin`.

### 6.2 Use shared storage without copying ImageNet (optional)

If a lab already exposes fully organized ImageNet directories, symlinks are
safe and save disk space.  The shared **validation directory itself must
already contain class folders**.  Example:

```bash
export SHARED_TRAIN=/mnt/PublicDatasets/Imagenet/ILSVRC/Data/CLS-LOC/train
export SHARED_VAL=/mnt/PublicDatasets/Imagenet/ILSVRC/Data/CLS-LOC/val

test -d "$SHARED_TRAIN" && test -d "$SHARED_VAL" || exit 1
mkdir -p "$IMAGENET_ROOT"
ln -s "$SHARED_TRAIN" "$IMAGENET_ROOT/train"
ln -s "$SHARED_VAL" "$IMAGENET_ROOT/val"
```

If `train` or `val` already exists, **do not run `ln -s` over it**.  First
inspect it with `ls -ld "$IMAGENET_ROOT/train" "$IMAGENET_ROOT/val"`.

#### Repairing the specific broken-validation-link layout

One common mistake is to create a local `val/nXXXXXXXX/` tree whose JPEG
symlinks point to a *flat* shared validation directory, while the real shared
images live in `val/nXXXXXXXX/`.  The symptom is:

```text
FileNotFoundError: .../val/n01440764/ILSVRC2012_val_00000293.JPEG
```

If your shared validation directory is already class-organized, replace only
the bad local `val` directory with a direct link.  This is recoverable because
the old directory is renamed, not deleted:

```bash
mv "$IMAGENET_ROOT/val" "$IMAGENET_ROOT/val.broken-links-YYYYMMDD"
ln -s "$SHARED_VAL" "$IMAGENET_ROOT/val"
```

Replace `YYYYMMDD` with the date.  Do this only after confirming
`test -f "$SHARED_VAL/n01440764/ILSVRC2012_val_00000293.JPEG"` succeeds.

### 6.3 Dataset verification: this is mandatory

Run this before DVCE:

```bash
python - <<'PY'
from torchvision.datasets import ImageNet

root = '/home/GTL/smubeen/imagenet_data/imagenet'
dataset = ImageNet(root=root, split='val')
image, label = dataset[0]
print('Validation images:', len(dataset))
print('Classes:', len(dataset.classes))
print('First sample:', image.size, 'label:', label)
assert len(dataset) == 50000
assert len(dataset.classes) == 1000
PY
```

Expected output includes `Validation images: 50000`, `Classes: 1000`, and a
valid first image size.  A successful dataset length alone is not enough:
`dataset[0]` must also load without `FileNotFoundError`.

## 7. Make the `--data_folder` argument actually work

The audited upstream revision has a path bug.  The generation script correctly
passes `--data_folder`, but `get_imagenet_path(data_folder)` ignores that
argument and always returns `/scratch/datasets/imagenet` for label lookup.
This causes a false request for a devkit under `/scratch` even when a valid
data directory was supplied.

Apply this narrow, backward-compatible patch in
`utils_svces/datasets/paths.py`:

```python
def get_imagenet_path(data_folder=None):
    if data_folder:
        return os.path.join(data_folder, 'imagenet')
    return os.path.join(get_base_data_dir(), 'imagenet')
```

It preserves the authors' `/scratch/datasets/imagenet` default when no path is
given, but makes `--data_folder /path/to/parent` work as documented.  This is
an infrastructure patch, not a DVCE algorithm modification.  Record it:

```bash
git diff -- utils_svces/datasets/paths.py
git diff --check
```

Keep this patch as a separate commit or a separately saved diff.  Do not alter
diffusion, classifier, or guidance code for the smoke run.

## 8. Preflight checklist

From the DVCE checkout, with `python_38_dvces_cu111` activated:

```bash
test -f checkpoints/256x256_diffusion_uncond.pt && echo 'diffusion: PASS'
test -f ImageNet1000Models/Madry_l2_improved/checkpoint/ep_3.pt && echo 'robust classifier: PASS'
test -f "$IMAGENET_ROOT/meta.bin" && echo 'ImageNet metadata: PASS'
test -d "$IMAGENET_ROOT/train" && echo 'ImageNet train: PASS'
test -d "$IMAGENET_ROOT/val" && echo 'ImageNet validation: PASS'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Every asset line should say `PASS`, and the final line should end in `True`.

## 9. First DVCE run: smallest practical smoke test

Use this exact command first.  It reduces only the **workload** to two selected
examples and a one-image GPU batch; it does not change the default DVCE
guidance configuration.

```bash
mkdir -p logs
PYTHONUNBUFFERED=1 python -u imagenet_VCEs.py \
  --data_folder /home/GTL/smubeen/imagenet_data \
  --num_imgs 2 \
  --batch_size 1 \
  --denoise_dist_input \
  --verbose \
  2>&1 | tee logs/dvce_smoke_2images_cu111.log
```

Replace only the `--data_folder` value if your parent directory differs.  It
must be the **parent** of `imagenet`, not the `imagenet` directory itself.

Why these options:

- `--data_folder …`: points to `$DATA_PARENT`, so the loader reaches
  `$DATA_PARENT/imagenet`.
- `--num_imgs 2`: minimal workload for debugging.
- `--batch_size 1`: safest starting point for GPU memory.
- `--denoise_dist_input`: used by the official default DVCE command.
- `--verbose`: prevents the script from suppressing its console output.
- `tee`: saves the complete terminal log while still showing it live.

The default configuration selected by the script is `configs/imagenet1000.yml`.
Its important values include 200 respaced diffusion steps, late start at 100,
classifier guidance weight 0.1, Lp distance regularization with p=1.0 and
weight 0.15, and the Madry + FT classifier.  The printed `final config` is the
authoritative record for a particular run; retain it in the log.

Expected output location:

```bash
find ImageNetVCEs -maxdepth 3 -type f | sort
```

The official README says generated images are written below
`ImageNetVCEs/examples/`, including combined figures and `single_images/`.
Do not claim the method reproduced until the Python process exits successfully
and these files exist.

## 10. Official comparison commands (only after the smoke test works)

Run each command separately and keep a separate log.  Start from the same
environment, checkpoint files, source commit, and ImageNet root.

### Default DVCE / no cone projection

```bash
PYTHONUNBUFFERED=1 python -u imagenet_VCEs.py \
  --data_folder /home/GTL/smubeen/imagenet_data \
  --num_imgs 12 --denoise_dist_input --verbose \
  2>&1 | tee logs/dvce_default_12.log
```

### Cone-projected DVCE

The authors show ConvNeXt (`31`) and Swin-T (`30`) as the second classifier:

```bash
for second_classifier_t in 31 30; do
  PYTHONUNBUFFERED=1 python -u imagenet_VCEs.py \
    --data_folder /home/GTL/smubeen/imagenet_data \
    --deg_cone_projection 30 \
    --second_classifier_type "$second_classifier_t" \
    --num_imgs 12 --denoise_dist_input --aug_num 16 --verbose \
    2>&1 | tee "logs/dvce_cone_second_${second_classifier_t}.log"
done
```

### SVCE baseline

```bash
PYTHONUNBUFFERED=1 python -u imagenet_VCEs.py \
  --data_folder /home/GTL/smubeen/imagenet_data \
  --num_imgs 12 --config svce.yml --verbose \
  2>&1 | tee logs/svce_12.log
```

### Blended diffusion baseline

```bash
PYTHONUNBUFFERED=1 python -u imagenet_VCEs.py \
  --data_folder /home/GTL/smubeen/imagenet_data \
  --num_imgs 12 --config blended.yml --use_blended \
  --background_preservation_loss --verbose \
  2>&1 | tee logs/blended_12.log
```

Do not compare these methods by reusing an output directory blindly.  Archive
or rename each completed `ImageNetVCEs/` directory before the next run, and
record the exact command, seed, source commit, package versions, checkpoint
hash, GPU model, runtime, and peak GPU memory.

## 11. How to tell whether it is working

In another terminal, while the command is running:

```bash
ps -fu "$USER" | grep '[i]imagenet_VCEs.py'
nvidia-smi
```

The DVCE process may spend time loading data/models before sustained GPU use.
For an active sampling phase, `nvidia-smi` should show the Python process and
GPU memory use.  A GPU utilization value of `0%` at one instant is not alone a
failure; use the saved log and process state as well.

If you accidentally start two runs, identify their PIDs with the first command
and stop only the unwanted PID using `kill -TERM <PID>`.  Never use broad
commands that kill every Python process on a shared machine.

## 12. Troubleshooting table

| Symptom | Meaning | Correct action |
| --- | --- | --- |
| `ILSVRC2012_devkit_t12.tar.gz is not present in /scratch/datasets/imagenet` despite `--data_folder` | Upstream path bug in `get_imagenet_path` | Apply the tiny patch in section 7; then ensure `meta.bin` exists in your actual ImageNet root. |
| `Permission denied` when creating `/scratch` | `/scratch` is a system-owned directory, and you should not need it | Do not use `sudo` or create `/scratch`. Use a directory you own and pass its parent through `--data_folder`. |
| `FileNotFoundError` for a JPEG under `val/nXXXXXXXX/...` | The validation image symlink layout is wrong | Verify the target file, then use the recoverable direct-`val` symlink repair in section 6.2. |
| Dataset reports 50,000 images but `dataset[0]` fails | Directory metadata was found, but actual JPEG links are broken | Treat this as a failed dataset setup; run the full section 6.3 check. |
| `CUDA error: no kernel image is available for execution on the device` on RTX 3090 | PyTorch was built without sm_86 kernels, usually the CUDA 10.2 author pin | Use the separate `python_38_dvces_cu111` environment and validate it as in section 5.2. |
| Conda proposes `pytorch ... cpu_0` | It selected a CPU build | Cancel; do not install it. Use the matching CUDA 11.1 wheels instead. |
| `No module named robust_finetuning` or `No module named clip` appears before a run | Optional imports are attempted by the repository | Note it in the log. If execution continues to model loading, do not install unrelated packages preemptively. Diagnose only a traceback that stops the run. |
| No output appears in the terminal | The repository hides output unless verbose mode is enabled | Use `--verbose`, `python -u`, and `PYTHONUNBUFFERED=1` exactly as in section 9. |
| Out-of-memory error | The per-GPU batch is too large | Retry with `--batch_size 1`. Record the change; it changes throughput, not the sampling method. |

## 13. What to save for every experiment

Create one log directory per run and save:

```text
experiment-name/
├── command.txt
├── stdout-stderr.log
├── git-commit.txt
├── git-diff.patch                 # includes the data-folder infrastructure patch
├── packages.txt                   # python -m pip freeze
├── conda-explicit.txt             # conda list --explicit
├── gpu.txt                        # nvidia-smi output
├── checkpoints.sha256
├── final-config.txt
└── outputs/                       # copied/renamed ImageNetVCEs result directory
```

At minimum, capture the environment immediately before a run:

```bash
git rev-parse HEAD > logs/git-commit.txt
git diff > logs/git-diff.patch
python --version > logs/python-version.txt
python -m pip freeze > logs/pip-freeze.txt
conda list --explicit > logs/conda-explicit.txt
nvidia-smi > logs/nvidia-smi.txt
sha256sum checkpoints/256x256_diffusion_uncond.pt \
  ImageNet1000Models/Madry_l2_improved/checkpoint/ep_3.pt \
  > logs/checkpoints.sha256
```

Use a new, timestamped log directory for each real run rather than overwriting
these files.

## 14. Boundaries before diatom adaptation

This guide establishes only that the official ImageNet pipeline is installed
and runnable.  It does **not** validate DVCE on diatoms, prove biological
meaning, or make ImageNet checkpoints suitable for microscopy.

Before changing data, classifiers, diffusion models, or losses:

1. Complete and archive an official ImageNet smoke run.
2. Complete a small published-style ImageNet result and inspect it.
3. Record any divergence from the paper/code.
4. Add regression tests around the confirmed baseline.
5. Only then design the diatom experiment and its baselines.

For the diatom experiments built on this setup, see [`README.md`](README.md)
and [`diatoms/COMMANDS.md`](diatoms/COMMANDS.md).

