"""Storage format of the downsampled images and the packed file that holds them.

Every image is centre-cropped to 4:3 and resized to STORE_H x STORE_W (128 x 170). A fixed size
lets the per-contributor static-pixel masks be defined in one coordinate frame, and the crop also
removes the 4:3 vs 16:9 aspect ratio, which is itself a camera property.

The JPEGs are concatenated into one binary file; <name>_index.csv holds id, offset, length and the
original width and height.
"""
import io
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

STORE_H, STORE_W = 128, 170


def downsample(jpeg_bytes, h=STORE_H, w=STORE_W, quality=90):
    """Original JPEG bytes -> (stored JPEG bytes, original (width, height))."""
    im = Image.open(io.BytesIO(jpeg_bytes))
    orig = im.size
    im.draft("RGB", (2 * w, 2 * h))  # fast DCT-domain downscale, still >= 2x the target size
    im = ImageOps.exif_transpose(im).convert("RGB")
    iw, ih = im.size
    if iw * h > ih * w:  # wider than 4:3
        cw = round(ih * w / h)
        box = ((iw - cw) // 2, 0, (iw - cw) // 2 + cw, ih)
    else:
        ch = round(iw * h / w)
        box = (0, (ih - ch) // 2, iw, (ih - ch) // 2 + ch)
    im = im.resize((w, h), Image.BICUBIC, box=box)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=quality)
    return out.getvalue(), orig


class PackedWriter:
    """Append-only writer; resumes after a crash by trusting only rows already in the index."""

    def __init__(self, bin_path):
        self.bin_path = bin_path
        self.idx_path = os.path.splitext(bin_path)[0] + "_index.csv"
        _drop_partial_last_line(self.idx_path)
        done = read_index(self.idx_path) if os.path.exists(self.idx_path) else None
        self.done = set() if done is None else set(done["id"])
        end = 0 if done is None or done.empty else int((done["offset"] + done["length"]).max())
        with open(bin_path, "ab") as f:
            f.truncate(end)
        self.f = open(bin_path, "ab")
        self.pos = end
        new = not os.path.exists(self.idx_path)
        self.idx = open(self.idx_path, "a", buffering=1)  # line-buffered: rows reach disk whole
        if new:
            self.idx.write("id,offset,length,orig_w,orig_h\n")

    def add(self, image_id, data, orig_size):
        self.f.write(data)
        self.f.flush()
        self.idx.write(f"{image_id},{self.pos},{len(data)},{orig_size[0]},{orig_size[1]}\n")
        self.pos += len(data)
        self.done.add(image_id)

    def flush(self):
        self.f.flush()
        self.idx.flush()
        os.fsync(self.f.fileno())
        os.fsync(self.idx.fileno())

    def close(self):
        self.flush()
        self.f.close()
        self.idx.close()


def _drop_partial_last_line(path):
    """After a hard kill the index may end in a half-written row; cut it off."""
    if not os.path.exists(path):
        return
    with open(path, "rb+") as f:
        data = f.read()
        if data and not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)


def read_index(path):
    """Index rows up to the last complete line: a fetch may be appending while a trainer reads."""
    with open(path, "rb") as f:
        data = f.read()
    return pd.read_csv(io.BytesIO(data[: data.rfind(b"\n") + 1]), dtype={"id": str}, keep_default_na=False)


class PackedReader:
    """Random access to the stored JPEGs. Safe in DataLoader workers (pread, file opened lazily)."""

    def __init__(self, bin_path, index=None):
        self.bin_path = bin_path
        index = read_index(os.path.splitext(bin_path)[0] + "_index.csv") if index is None else index
        self.row = {k: i for i, k in enumerate(index["id"])}
        self.offset = index["offset"].to_numpy(np.int64)
        self.length = index["length"].to_numpy(np.int64)
        self._fd = None
        self._pid = None

    def __contains__(self, image_id):
        return image_id in self.row

    def read(self, image_id):
        if self._fd is None or self._pid != os.getpid():
            self._fd, self._pid = os.open(self.bin_path, os.O_RDONLY), os.getpid()
        i = self.row[image_id]
        return os.pread(self._fd, int(self.length[i]), int(self.offset[i]))

    def image(self, image_id):
        return Image.open(io.BytesIO(self.read(image_id))).convert("RGB")
