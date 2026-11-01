"""Camera-fingerprint removal: static-pixel masks, bottom blanking, colour-balance and JPEG jitter.

Images from one contributor share the car bonnet, dashboard, windscreen reflections, timestamp
overlays and compression settings. The deterministic parts (static mask, bottom band) are applied
to train and evaluation images alike; the random parts (colour balance, JPEG quality) only to
training images.

Static mask: pixels whose standard deviation across a contributor's photos (taken from several
different sequences) stays below a threshold. Pixels are filled with the ImageNet mean colour,
which is ~0 after normalisation. Masks are estimated from the contributor's own images and never
use labels, so they are computed for test contributors too.

Caveat (measured on the 32-country subset, 128 x 170): only 226 of 3,674 contributors with enough
images get a non-empty mask. The largest masks cover repetitive scene content (road surface, snow,
uniform sky) rather than camera artefacts, because bonnets reflect the sky and contributors change
cars and mounts. Masks are therefore capped at 10 % of the frame and are opt-in (--static-mask);
the bottom band does the main work against bonnets and dashboards.
"""
import numpy as np
from scipy import ndimage

FILL_RGB = np.array([124, 116, 104], dtype=np.uint8)  # ImageNet mean


def static_mask(stack, seq_ids, std_thresh=10.0, min_images=8, min_sequences=4, per_sequence=3,
                min_component=12, dilate=1, max_area=0.1):
    """stack: (n, H, W, 3) uint8 images of one contributor; seq_ids: their sequence ids.

    Returns (mask or None, info). None when there are too few images or sequences (frames of one
    trip look alike, so they would make the whole road look static), or when the mask covers more
    than max_area of the frame (a sign that the contributor's photos are too similar to judge).
    """
    seq_ids = np.asarray(seq_ids)
    keep = np.zeros(len(seq_ids), dtype=bool)
    for s in np.unique(seq_ids):
        keep[np.flatnonzero(seq_ids == s)[:per_sequence]] = True
    stack, seq_ids = stack[keep], seq_ids[keep]
    info = {"n_images": int(len(stack)), "n_sequences": int(len(np.unique(seq_ids)))}
    if info["n_images"] < min_images or info["n_sequences"] < min_sequences:
        info["status"] = "too_few"
        return None, info
    sd = stack.astype(np.float32).std(axis=0).mean(axis=-1)
    m = sd < std_thresh
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
        m = np.concatenate([[False], sizes >= min_component])[lab]
    if dilate:
        m = ndimage.binary_dilation(m, iterations=dilate)
    info["area"] = float(m.mean())
    if info["area"] > max_area:
        info["status"] = "too_large"
        return None, info
    info["status"] = "ok"
    return m, info


def apply_blanking(arr, mask=None, bottom_frac=0.0):
    """Fill masked pixels and the bottom band of an (H, W, 3) uint8 array in place."""
    if mask is not None:
        arr[mask] = FILL_RGB
    if bottom_frac > 0:
        arr[int(round(arr.shape[0] * (1 - bottom_frac))):] = FILL_RGB
    return arr


def pack_masks(masks):
    return np.packbits(np.stack(masks).reshape(len(masks), -1), axis=1)


def unpack_mask(bits, shape):
    return np.unpackbits(bits, count=shape[0] * shape[1]).reshape(shape).astype(bool)
