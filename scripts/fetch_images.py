"""Fetch the manifest's images from the OSV-5M zips on Hugging Face, downsample and pack them.

One HTTP range request per image (offsets from scripts/index_zips.py). Each image is
centre-cropped to 4:3, resized to 128 x 170 and re-encoded as JPEG (geocnn/images.py), then
appended to <out>.bin with its row in <out>_index.csv. Re-running resumes where it stopped.

Throughput is bounded by the Hugging Face CDN (~15 requests/s measured from a home connection),
so ~140k images take ~2.5 h locally; a Kaggle CPU notebook is usually faster.

Usage:
  python -I scripts/fetch_images.py --manifest data/c32/manifest.csv --index-dir data/zip_index \
      --out data/c32/images.bin
  # smoke test: a few images per split
  python -I scripts/fetch_images.py ... --out data/smoke/images.bin --limit-per-split 300
"""
import argparse
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geocnn.images import PackedWriter, downsample  # noqa: E402
from geocnn.remote_zip import RemoteFile, load_locations, read_member, shard_url  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--out", required=True, help="packed .bin path; <out>_index.csv is written next to it")
    ap.add_argument("--osv-split", default="train", help="OSV-5M split the images come from")
    ap.add_argument("--splits", default="", help="comma-separated manifest splits (default: all)")
    ap.add_argument("--limit-per-split", type=int, default=0, help="random sample per split (smoke tests)")
    ap.add_argument("--max-chunk", type=int, default=0,
                    help="with train_extra.csv (scripts/extend_train.py): only chunks 1..N")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    man = pd.read_csv(a.manifest, dtype={"id": str, "creator_id": str}, keep_default_na=False)
    if a.splits:
        man = man[man["split"].isin(a.splits.split(","))]
    if a.max_chunk:
        man = man[man["chunk"] <= a.max_chunk]
    if a.limit_per_split:
        man = man.sample(frac=1, random_state=a.seed)
        man = man[man.groupby("split").cumcount() < a.limit_per_split]
    loc = load_locations(a.index_dir, a.osv_split, set(man["id"]))
    missing = sorted(set(man["id"]) - set(loc["id"]))
    if missing:
        print(f"WARNING: {len(missing)} manifest ids not found in the zip index (first: {missing[:5]})")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    writer = PackedWriter(a.out)
    todo = loc[~loc["id"].astype(object).isin(writer.done)].sort_values(["shard", "offset"])
    print(f"{len(man):,} requested, {len(writer.done):,} already packed, {len(todo):,} to fetch", flush=True)

    remotes = {}

    def job(row):
        rf = remotes.get(row.shard)
        if rf is None:
            rf = remotes.setdefault(row.shard, RemoteFile(shard_url(a.osv_split, row.shard)))
        data = read_member(rf, row.offset, row.csize, row.method)
        return row.id, downsample(data, quality=a.quality)

    failures, n_done, t0 = [], 0, time.time()
    rows = iter(todo.itertuples(index=False))
    with ThreadPoolExecutor(a.threads) as ex:
        pending = set()
        while True:
            while len(pending) < 4 * a.threads:
                row = next(rows, None)
                if row is None:
                    break
                fut = ex.submit(job, row)
                fut.row = row
                pending.add(fut)
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                try:
                    image_id, (data, orig) = fut.result()
                    writer.add(image_id, data, orig)
                    n_done += 1
                except Exception as e:
                    failures.append({"id": fut.row.id, "shard": fut.row.shard, "error": repr(e)[:300]})
            if n_done and n_done % 500 < len(finished):
                writer.flush()
                rate = n_done / (time.time() - t0)
                eta = (len(todo) - n_done - len(failures)) / max(rate, 1e-9)
                print(f"  {n_done:,}/{len(todo):,}  {rate:.1f} img/s  ETA {eta / 60:.0f} min  "
                      f"failures {len(failures)}", flush=True)
    writer.close()
    print(f"done: {n_done:,} fetched in {(time.time() - t0) / 60:.1f} min, {len(failures)} failures")
    if failures:
        fail_path = os.path.splitext(a.out)[0] + "_failures.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        sys.exit(f"{len(failures)} images failed (listed in {fail_path}); re-run to retry them")


if __name__ == "__main__":
    main()
