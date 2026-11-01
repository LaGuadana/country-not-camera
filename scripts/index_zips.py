"""Index the OSV-5M image zips on Hugging Face without downloading them.

Reads each shard's central directory with HTTP range requests (~4.6 MB per 2.5 GB train shard)
and writes <out-dir>/<split>_<shard>.csv with columns id, shard, offset, csize, method.
Shards already indexed are skipped, so the script can be re-run after an interruption.

Usage:
  python -I scripts/index_zips.py --out-dir data/zip_index --split train
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geocnn.remote_zip import RemoteFile, read_central_directory, shard_url  # noqa: E402

N_SHARDS = {"train": 98, "test": 5}


def index_one(split, shard, out_dir):
    out = os.path.join(out_dir, f"{split}_{shard:02d}.csv")
    if os.path.exists(out):
        return out, None
    entries = read_central_directory(RemoteFile(shard_url(split, shard)))
    df = pd.DataFrame([e for e in entries if e["name"].endswith(".jpg")])
    df["id"] = df["name"].str.rsplit("/", n=1).str[-1].str.removesuffix(".jpg")
    df["shard"] = shard
    tmp = out + ".tmp"
    df[["id", "shard", "offset", "csize", "method"]].to_csv(tmp, index=False)
    os.replace(tmp, out)
    return out, len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", choices=list(N_SHARDS), default="train")
    ap.add_argument("--shards", default="", help="comma-separated shard numbers (default: all)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    shards = [int(s) for s in a.shards.split(",")] if a.shards else range(N_SHARDS[a.split])
    failed = []
    with ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(index_one, a.split, s, a.out_dir): s for s in shards}
        for f in as_completed(futs):
            try:
                out, n = f.result()
                print(f"{out}: {'exists' if n is None else f'{n:,} images'}", flush=True)
            except Exception as e:  # keep going; re-run to retry
                failed.append(futs[f])
                print(f"shard {futs[f]:02d} FAILED: {e}", flush=True)
    if failed:
        sys.exit(f"failed shards (re-run to retry): {sorted(failed)}")


if __name__ == "__main__":
    main()
