# How we made the diatom classifiers robust (adversarial fine-tuning)

## 1. The problem: normal classifiers are easy to fool

A normally trained classifier can be fooled by an **adversarial perturbation**:
a tiny change to the pixels, invisible to a human, computed so that the
classifier changes its answer. The independent classifier (a standard ResNet-18) is right on 83% of
clean test images, but after such invisible changes on only ~2% (eps 1.0).

For counterfactual explanations (DVCE) that's fatal. DVCE changes the image
until the classifier says "species B". With a non-robust classifier, an
invisible change already does that, and the "explanation" shows nothing.

## 2. The idea: train on attacked images

**Adversarial training:** in every training step we first **attack** the
training images ourselves (find the small change that fools the current model
most), then train the model to give the **right answer on the attacked
images**. Over time, small changes stop working, and only real, visible
features (outline, striae, size) can change the model's decision.

Analogy: a student who only practises with clean exam questions fails when a
question is worded slightly trickily. If every practice question is
deliberately reworded in the trickiest small way, the student learns the
actual content instead of surface patterns.

## 3. One training step, concretely

For each batch of 128 training images:

1. **Attack (PGD, 5 steps).** Start from a small random change. Five times:
   - compute in which direction the pixels must change to make the model
     *more wrong* (the gradient of the loss with respect to the image)
   - take a step in that direction
   - shrink the change back if its total size exceeds **eps**, and keep
     pixels between 0 and 1
2. **Learn.** Feed the attacked images to the model, compute the error
   against the true species, and update the weights to reduce it (SGD).

**eps** limits the size of the change (L2 norm over all pixels, values 0–1):
- eps 0.5 ≈ 0.3 grey levels per pixel on average, invisible
- eps 1.0 is twice that, still invisible

A larger eps gives a more robust model but costs some clean accuracy.

## 4. What we did for the two models

| | Independent classifier, robust fine-tuned | Our robust ResNet-50 |
|---|---|---|
| Starting weights | his trained diatom ResNet-18 (759 species), **not robust** | ResNet-50 already made **robust on ImageNet** (Madry et al., used in the DVCE paper), with a new 759-species output layer |
| Data split | page split (whole atlas pages per split) | random image split 80/10/10 |
| Training images | 23,494 | 26,921 |
| eps (two versions) | 0.5 and 1.0 | 0.5 and 1.0 |
| Attack during training | PGD, 5 steps | PGD, 5 steps |
| Epochs | 20 | 20 |
| Optimiser | SGD, lr 0.01, momentum 0.9, cosine decay, weight decay 5e-4 | same |
| Augmentation | random flips | flips + rotation ±10°, shift ±8 px, scale ±10%, brightness ±15%, contrast ±25% |
| Time | ~40 min per version (one GPU) | ~2 h per version (one GPU) |
| Model kept | epoch with best **validation accuracy under attack** | same |

Both use the same input as DVCE: native-size 256×256 image → center 224×224
crop → ImageNet normalisation (built into the saved model).

Why start the ResNet-50 from a robust ImageNet model **and** train
adversarially? We tested it: fine-tuning that robust model **without**
attacks loses almost all robustness (10.5% correct under an eps-0.5 attack).
The robust start helps, but adversarial training is what keeps the model
robust.

## 5. How we measured robustness

On the **test** images (never used in training):
- **Clean accuracy:** % correct on the normal images.
- **Accuracy under attack:** attack every test image with a **stronger** PGD
  (10 steps) at eps 0.25 / 0.5 / 1.0 / 2.0, then count % still correct.
  - attacks run in full precision (fp32); fp16 made them too weak
  - same random start for all models, so the comparison is fair

Results: `ROBUSTNESS.md`.

## 6. Code

- training: `diatoms/train_classifier.py` (`--eps`, `--init`, `--split_mode`,
  `--augment`); the attack is `pgd_l2()` in the same file
- evaluation: `diatoms/eval_classifier.py`
- exact commands: `diatoms/COMMANDS.md`, sections 10 and 14

Example (our ResNet-50, eps 1.0):

```bash
CUDA_VISIBLE_DEVICES=1 python diatoms/train_classifier.py --run r50madry_random_eps1.0 \
    --split_mode random --init madry_r50 --eps 1.0 --augment strong --epochs 20
```
