"""Read single members of the OSV-5M image zips on Hugging Face with HTTP range requests.

Each train shard (images/train/00.zip .. 97.zip) is ~2.5 GB with ~50k JPEGs stored
uncompressed. Reading the central directory (~4.6 MB at the end of the file) gives every
member's offset, so a selected image can be fetched with one range request instead of
downloading the whole shard.
"""
import glob
import os
import struct
import threading
import time
import zlib

import pandas as pd
import requests

HF_BASE = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main"

_tls = threading.local()  # one keep-alive session per thread, shared by all shards (same CDN host)


def _session():
    s = getattr(_tls, "s", None)
    if s is None:
        s = _tls.s = requests.Session()
    return s

_EOCD = b"PK\x05\x06"
_EOCD64_LOC = b"PK\x06\x07"
_CDIR = b"PK\x01\x02"
_LOCAL = b"PK\x03\x04"


def shard_url(split, shard):
    return f"{HF_BASE}/images/{split}/{shard:02d}.zip"


class RemoteFile:
    """Range reads on one remote file. The HF redirect to the signed CDN URL is resolved
    once and refreshed when the CDN starts rejecting it (expired signature)."""

    def __init__(self, url, timeout=60, retries=8):
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self._lock = threading.Lock()
        self._cdn = None
        self._size = None

    def _resolve(self, force=False):
        with self._lock:
            if self._cdn is None or force:
                r = _session().head(self.url, allow_redirects=False, timeout=self.timeout)
                if r.status_code in (301, 302, 303, 307, 308):
                    self._cdn = r.headers["location"]
                    self._size = int(r.headers.get("x-linked-size") or 0) or None
                else:
                    r.raise_for_status()
                    self._cdn = self.url
                    self._size = int(r.headers["content-length"])
            return self._cdn

    @property
    def size(self):
        if self._size is None:
            self._resolve()
        if self._size is None:
            r = _session().head(self._resolve(), timeout=self.timeout)
            r.raise_for_status()
            self._size = int(r.headers["content-length"])
        return self._size

    def read(self, start, length):
        """Return bytes [start, start + length)."""
        end = start + length - 1
        err = None
        for attempt in range(self.retries):
            url = self._resolve(force=isinstance(err, PermissionError))
            try:
                r = _session().get(url, headers={"Range": f"bytes={start}-{end}"},
                                        timeout=self.timeout)
                if r.status_code in (401, 403):
                    err = PermissionError(f"HTTP {r.status_code}")
                    continue
                if r.status_code != 206:
                    raise IOError(f"expected 206, got HTTP {r.status_code}")
                if len(r.content) != length:
                    raise IOError(f"short read {len(r.content)} of {length}")
                return r.content
            except (requests.RequestException, IOError) as e:
                err = e
                time.sleep(min(30, 2 ** attempt))
        raise IOError(f"range read failed for {self.url} [{start}, {end}]: {err}")


def read_central_directory(rf):
    """List members of a remote zip: dicts with name, offset, csize, usize, method."""
    tail_len = min(rf.size, 1 << 16)
    tail = rf.read(rf.size - tail_len, tail_len)
    i = tail.rfind(_EOCD)
    if i < 0:
        raise ValueError("end of central directory not found")
    _, _, _, _, n_total, cd_size, cd_off, _ = struct.unpack("<4s4H2LH", tail[i:i + 22])
    if 0xFFFFFFFF in (cd_size, cd_off) or n_total == 0xFFFF:
        j = tail.rfind(_EOCD64_LOC, 0, i)
        if j < 0:
            raise ValueError("zip64 locator not found")
        eocd64_off = struct.unpack("<Q", tail[j + 8:j + 16])[0]
        rec = rf.read(eocd64_off, 56)
        n_total, cd_size, cd_off = struct.unpack("<3Q", rec[32:56])
    cd = rf.read(cd_off, cd_size)
    out, p = [], 0
    while p < len(cd):
        if cd[p:p + 4] != _CDIR:
            raise ValueError(f"bad central directory entry at {p}")
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, usize, n_name, n_extra, n_comment = struct.unpack("<2L3H", cd[p + 20:p + 34])
        off, = struct.unpack("<L", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + n_name].decode("utf-8")
        if 0xFFFFFFFF in (csize, usize, off):
            csize, usize, off = _zip64_extra(cd[p + 46 + n_name:p + 46 + n_name + n_extra],
                                             csize, usize, off)
        out.append({"name": name, "offset": off, "csize": csize, "usize": usize, "method": method})
        p += 46 + n_name + n_extra + n_comment
    if len(out) != n_total:
        raise ValueError(f"central directory has {len(out)} entries, EOCD says {n_total}")
    return out


def _zip64_extra(extra, csize, usize, off):
    q = 0
    while q + 4 <= len(extra):
        hid, hlen = struct.unpack("<2H", extra[q:q + 4])
        if hid == 1:
            vals = list(struct.unpack(f"<{hlen // 8}Q", extra[q + 4:q + 4 + hlen - hlen % 8]))
            if usize == 0xFFFFFFFF:
                usize = vals.pop(0)
            if csize == 0xFFFFFFFF:
                csize = vals.pop(0)
            if off == 0xFFFFFFFF:
                off = vals.pop(0)
            break
        q += 4 + hlen
    return csize, usize, off


def read_member(rf, offset, csize, method, slack=512):
    """Fetch one member's bytes given its local-header offset (one request in the common case)."""
    buf = rf.read(offset, min(30 + slack + csize, rf.size - offset))
    if buf[:4] != _LOCAL:
        raise ValueError(f"no local file header at {offset}")
    n_name, n_extra = struct.unpack("<2H", buf[26:30])
    start = 30 + n_name + n_extra
    if start + csize > len(buf):
        buf = buf + rf.read(offset + len(buf), start + csize - len(buf))
    data = buf[start:start + csize]
    if method == 0:
        return data
    if method == 8:
        return zlib.decompress(data, -15)
    raise ValueError(f"unsupported compression method {method}")


def load_locations(index_dir, split, ids):
    """Rows (id, shard, offset, csize, method) of the zip index (scripts/index_zips.py) for `ids`."""
    parts = []
    for p in sorted(glob.glob(os.path.join(index_dir, f"{split}_*.csv"))):
        d = pd.read_csv(p, dtype={"id": str}, keep_default_na=False)
        parts.append(d[d["id"].astype(object).isin(ids)])  # object isin: pandas 3 str isin is ~30x slower
    if not parts:
        raise SystemExit(f"no zip index files in {index_dir}; run scripts/index_zips.py first")
    return pd.concat(parts, ignore_index=True)
