"""The independent classifier's diatom preprocessing (PadCropDiatom), as he shared it, adapted
to our environment. Used to feed the independent ResNet-18 the images it was trained on.

Steps (from his code):
  1. scale matching: resize by src_scale / tgt_scale (microns per pixel), INTER_AREA
  2. long axis vertical, via np.transpose (a transpose, not a rotation)
  3. center-crop to tgt_size if larger (no downscaling)
  4. "seamless" square padding: canvas filled with the image median, the image
     flipped upside down (np.flip axis 0) and Poisson-cloned onto the centre
  5. grayscale (to_gray=True)

Differences from his version:
  * our OpenCV (4.5.5) has no cv2.NORMAL_CLONE_WIDE (newer OpenCV), so we use
    cv2.NORMAL_CLONE with the outermost mask pixels cleared, which old
    seamlessClone needs when the image touches the canvas border.
  * src_scale per image comes from DIATLAS diatoms.csv: `scale` is the length
    in microns of 100 pixels, so src_scale = scale / 100.
  * tgt_scale is not in his message; it is a parameter here.
"""
import cv2
import numpy as np
from PIL import Image


def pad_crop_diatom(img, src_scale, tgt_scale, tgt_size=224, to_gray=True, crop_size=None):
    """crop_size: center-crop limit (default tgt_size). With crop_size=224, tgt_size=256 the
    diatom content is identical to his 224 version, on a larger 256 canvas."""
    crop_size = crop_size or tgt_size
    img_cv = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)

    resize_factor = src_scale / tgt_scale
    if resize_factor != 1.0:
        new_size = (max(1, int(round(img_cv.shape[1] * resize_factor))),
                    max(1, int(round(img_cv.shape[0] * resize_factor))))
        img_cv = cv2.resize(img_cv, new_size, interpolation=cv2.INTER_AREA)

    if img_cv.shape[0] < img_cv.shape[1]:
        img_cv = np.ascontiguousarray(np.transpose(img_cv, (1, 0, 2)))

    h, w = img_cv.shape[:2]
    if h > crop_size or w > crop_size:
        sy, sx = max(0, (h - crop_size) // 2), max(0, (w - crop_size) // 2)
        img_cv = img_cv[sy:sy + crop_size, sx:sx + crop_size]

    processed = _convert_to_square(img_cv, tgt_size)

    if to_gray:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(processed)


def _convert_to_square(image, size):
    final_image = np.zeros((size, size, 3), dtype=np.uint8)
    final_image[:, :] = np.median(image.reshape(-1, 3), axis=0)
    image_flipped = np.ascontiguousarray(np.flip(image, axis=0))
    mask = np.full_like(image_flipped, 255, dtype=np.uint8)
    if min(mask.shape[:2]) > 2:
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    center = (size // 2, size // 2)
    flags = getattr(cv2, 'NORMAL_CLONE_WIDE', cv2.NORMAL_CLONE)
    try:
        return cv2.seamlessClone(image_flipped, final_image, mask, center, flags)
    except cv2.error:
        # very small crops: plain paste
        h, w = image_flipped.shape[:2]
        y, x = (size - h) // 2, (size - w) // 2
        final_image[y:y + h, x:x + w] = image_flipped
        return final_image
