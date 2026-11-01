"""Dataset, augmentation and data loading for the packed 128 x 170 OSV-5M subset.

Standard augmentation (both conditions): random resized crop back to 128 x 170. No horizontal
flip: mirroring swaps the driving side and mirrors text, both genuine country cues.

Fingerprint removal (--fingerprint): blank the bottom band (bonnet, dashboard) and, with
--static-mask, the contributor's static-pixel mask on every image; on training images also re-encode as JPEG at a
random quality and jitter white balance, brightness, contrast and saturation. Hue is not jittered
because the colours of road markings and signs (e.g. yellow vs white centre lines) are country cues.
Blanking keeps the input geometry identical in both conditions, so the comparison changes only the
information shown, not the image scale.

Work split: DataLoader workers decode, blank, crop and JPEG-recompress, and hand over uint8
tensors; `Prep` then does the colour jitter and normalisation on the whole batch on the GPU, which
keeps the few CPU cores of a Kaggle machine from becoming the bottleneck.
"""
import io
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as T

from .fingerprint import apply_blanking, unpack_mask
from .images import STORE_H, STORE_W, PackedReader

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@dataclass
class AugConfig:
    crop_scale: tuple = (0.6, 1.0)
    fingerprint: bool = False
    bottom_frac: float = 0.2
    static_mask: bool = False     # opt-in: see the note in geocnn/fingerprint.py
    white_balance: float = 0.15   # per-channel gain in [1 - wb, 1 + wb]
    color_jitter: float = 0.25    # brightness / contrast / saturation strength
    jpeg_prob: float = 0.8
    jpeg_quality: tuple = (30, 95)


class Masks:
    """Per-contributor static-pixel masks stored by scripts/build_masks.py."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.shape = tuple(z["shape"])
        self.bits = z["bits"]
        self.row = {c: i for i, c in enumerate(z["creators"].astype(str))}

    def get(self, creator):
        i = self.row.get(creator)
        return None if i is None else unpack_mask(self.bits[i], self.shape)


class OSVDataset(Dataset):
    """Yields (uint8 tensor 3 x 128 x 170, label, row index)."""

    def __init__(self, rows, reader, train, aug, masks=None):
        self.ids = rows["id"].tolist()
        self.labels = rows["label"].to_numpy(np.int64)
        self.creators = rows["creator_id"].tolist()
        self.reader, self.train, self.aug, self.masks = reader, train, aug, masks
        self.crop = T.RandomResizedCrop((STORE_H, STORE_W), scale=aug.crop_scale,
                                        ratio=(STORE_W / STORE_H * 0.9, STORE_W / STORE_H * 1.1),
                                        antialias=True)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = self.reader.image(self.ids[i])
        aug = self.aug
        if aug.fingerprint:
            mask = self.masks.get(self.creators[i]) if (aug.static_mask and self.masks) else None
            img = Image.fromarray(apply_blanking(np.array(img), mask, aug.bottom_frac))
        if self.train:
            img = self.crop(img)
            if aug.fingerprint and random.random() < aug.jpeg_prob:
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=random.randint(*aug.jpeg_quality))
                img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        return torch.from_numpy(np.array(img)).permute(2, 0, 1), self.labels[i], i


class Prep(nn.Module):
    """uint8 batch -> normalised float batch, with per-image colour jitter on training batches
    when fingerprint removal is on. Runs on the model's device."""

    def __init__(self, aug, train):
        super().__init__()
        self.jitter = train and aug.fingerprint
        self.wb, self.c = aug.white_balance, aug.color_jitter
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.register_buffer("luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def _u(self, n, k, strength, x):
        return 1 + (torch.rand(n, k, 1, 1, device=x.device) * 2 - 1) * strength

    @torch.no_grad()
    def forward(self, x):
        x = x.float().div_(255)
        if self.jitter:
            n = x.shape[0]
            x = x * self._u(n, 3, self.wb, x)                                    # white balance
            x = x * self._u(n, 1, self.c, x)                                     # brightness
            gray = (x * self.luma).sum(1, keepdim=True)
            m = gray.mean((2, 3), keepdim=True)
            x = (x - m) * self._u(n, 1, self.c, x) + m                           # contrast
            gray = (x * self.luma).sum(1, keepdim=True)
            x = ((x - gray) * self._u(n, 1, self.c, x) + gray).clamp_(0, 1)      # saturation
        return (x - self.mean) / self.std


def load_split_tables(data_dir):
    man = pd.read_csv(os.path.join(data_dir, "manifest.csv"),
                      dtype={"id": str, "creator_id": str, "sequence": str}, keep_default_na=False)
    with open(os.path.join(data_dir, "classes.json")) as f:
        classes = json.load(f)
    return man, classes


def open_data(data_dir, images="images.bin"):
    """Manifest rows that have a packed image, the class list, the reader and the masks (or None)."""
    man, classes = load_split_tables(data_dir)
    reader = PackedReader(os.path.join(data_dir, images))
    have = man["id"].astype(object).isin(reader.row.keys())  # pandas 3 str isin is slow
    if not have.all():
        print(f"note: {(~have).sum():,} of {len(man):,} manifest images are not packed yet; using the rest")
    mask_path = os.path.join(data_dir, "masks.npz")
    masks = Masks(mask_path) if os.path.exists(mask_path) else None
    return man[have].reset_index(drop=True), classes, reader, masks


class CoverageSampler(Sampler):
    """Epochs of `epoch_size` images that walk through one random permutation of the dataset, so
    every image is drawn once before any is drawn again. Call set_epoch(e) before each epoch."""

    def __init__(self, n, epoch_size, seed):
        self.n, self.epoch_size, self.seed, self.epoch = n, epoch_size, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        out, pos = [], self.epoch * self.epoch_size
        while len(out) < self.epoch_size:
            cycle, off = divmod(pos + len(out), self.n)
            perm = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed * 1_000_003 + cycle))
            out += perm[off: off + self.epoch_size - len(out)].tolist()
        return iter(out)


def make_loader(rows, reader, train, aug, masks, batch_size, workers, seed=0, epoch_size=0):
    """Workers are restarted every epoch (not persistent) so that re-seeding `loader.generator`
    before an epoch fixes both the shuffle order and the workers' augmentation seeds.
    epoch_size > 0 makes each training epoch the next `epoch_size` images of a random permutation
    (CoverageSampler; call loader.sampler.set_epoch)."""
    ds = OSVDataset(rows, reader, train, aug, masks)
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = None
    if train and 0 < epoch_size < len(ds):
        sampler = CoverageSampler(len(ds), epoch_size, seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=train and sampler is None, sampler=sampler,
                      num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=train,
                      prefetch_factor=4 if workers > 0 else None, generator=g)
