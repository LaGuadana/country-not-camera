"""Build Dashcam Duel, the phone game: you vs. a model, guessing the country of a street photo.

Photos come from the contributor-disjoint `test` split (photographers the model never trained on),
at most one per contributor, the same number per country. They are dealt into fixed games of
--game-size photos. The page holds the countries and the model's top-3 guesses. Each game's
photos go to g/<game>.json next to it (~0.8 MB), and the page fetches one when that game starts.
Per photo, the file holds:
  hi    the original OSV-5M image (512 px high), fetched once from Hugging Face and cached;
        EXIF stripped, since it can hold the GPS position
  lo    the 128 x 170 centre crop the model sees
  crop  where that crop sits in the original (fractions x, y, w, h)
  cam   Grad-CAM of the model's top guess (geocnn/explain.py), upsampled to CAM_H x CAM_W, uint8
The page remembers which games the phone has played, so photos repeat only once all of them
have been seen. The photo selection depends only on the data and --seed, so re-running with a
better checkpoint (e.g. the Kaggle fine-tune) keeps the same games and changes only the opponent.

Usage:
  python -I scripts/export_game.py --ckpt runs/probe/best.pt --out game/dashcam_duel.html \
      --model-label "ResNet-18, ImageNet features + linear probe"
  then publish the page together with game/g/*.json
"""
import argparse
import base64
import hashlib
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geocnn.data import AugConfig, Prep, make_loader, open_data  # noqa: E402
from geocnn.explain import grad_cam  # noqa: E402
from geocnn.images import STORE_H, STORE_W, PackedReader, PackedWriter  # noqa: E402
from geocnn.model import CountryNet  # noqa: E402
from geocnn.remote_zip import RemoteFile, load_locations, read_member, shard_url  # noqa: E402

CAM_H, CAM_W = 24, 32
COUNTRIES = {  # code: (short name, picker group)
    "AT": ("Austria", "Europe"), "BE": ("Belgium", "Europe"), "CH": ("Switzerland", "Europe"),
    "CZ": ("Czechia", "Europe"), "DE": ("Germany", "Europe"), "DK": ("Denmark", "Europe"),
    "ES": ("Spain", "Europe"), "FR": ("France", "Europe"), "GB": ("UK", "Europe"),
    "GR": ("Greece", "Europe"), "HR": ("Croatia", "Europe"), "HU": ("Hungary", "Europe"),
    "IT": ("Italy", "Europe"), "NL": ("Netherlands", "Europe"), "PL": ("Poland", "Europe"),
    "PT": ("Portugal", "Europe"), "RU": ("Russia", "Europe"), "SE": ("Sweden", "Europe"),
    "TR": ("Türkiye", "Europe"), "ID": ("Indonesia", "Asia"), "IN": ("India", "Asia"),
    "JP": ("Japan", "Asia"), "MY": ("Malaysia", "Asia"), "TH": ("Thailand", "Asia"),
    "TW": ("Taiwan", "Asia"), "BR": ("Brazil", "Americas"), "CA": ("Canada", "Americas"),
    "CO": ("Colombia", "Americas"), "MX": ("Mexico", "Americas"), "US": ("USA", "Americas"),
    "AU": ("Australia", "Africa & Oceania"), "ET": ("Ethiopia", "Africa & Oceania"),
}


def usable(img_bytes):
    """Skip black, blown-out or flat frames."""
    a = np.asarray(Image.open(io.BytesIO(img_bytes)).convert("L"), dtype=np.float32)
    return 25 < a.mean() < 235 and a.std() > 20


def default_note(ckpt_path):
    d = os.path.dirname(os.path.abspath(ckpt_path))
    for path, key in ((os.path.join(d, "eval", "metrics.json"), "test"), (os.path.join(d, "probe.json"), None)):
        if os.path.exists(path):
            m = json.load(open(path))
            bacc = m["splits"][key]["balanced_acc"] if key else m["by_l2"][str(m["chosen_l2"])]["val_balanced_acc"]
            return f"right {bacc:.0%} of the time on unseen photographers, chance is 3%"
    return ""


def originals(ids, index_dir, cache, threads=16):
    """Original JPEGs of `ids` from the OSV-5M train zips, cached in a packed file."""
    writer = PackedWriter(cache)
    todo = set(ids) - writer.done
    if todo:
        loc = load_locations(index_dir, "train", todo)
        if len(loc) != len(todo):
            raise SystemExit(f"{len(todo) - len(loc)} ids not in the zip index {index_dir}")
        remotes = {}

        def job(row):
            rf = remotes.get(row.shard) or remotes.setdefault(row.shard, RemoteFile(shard_url("train", row.shard)))
            return row.id, read_member(rf, row.offset, row.csize, row.method)

        with ThreadPoolExecutor(threads) as ex:
            for image_id, data in ex.map(job, loc.itertuples(index=False)):
                writer.add(image_id, data, Image.open(io.BytesIO(data)).size)
        print(f"fetched {len(todo)} originals into {cache}")
    writer.close()
    return PackedReader(cache)


def game_photo(orig_bytes, max_h, quality):
    """Upright original, at most max_h high, re-encoded without metadata; plus the model's crop."""
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(orig_bytes))).convert("RGB")
    if im.height > max_h:
        im = im.resize((round(im.width * max_h / im.height), max_h), Image.LANCZOS)
    w, h = im.size
    if w * STORE_H > h * STORE_W:  # same 4:3 centre crop as geocnn.images.downsample
        cw = h * STORE_W / STORE_H
        crop = ((w - cw) / 2 / w, 0, cw / w, 1)
    else:
        ch = w * STORE_H / STORE_W
        crop = (0, (h - ch) / 2 / h, 1, ch / h)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=quality, optimize=True)
    return out.getvalue(), (w, h), [round(v, 4) for v in crop]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default=None, help="default: the checkpoint's training data")
    ap.add_argument("--index-dir", default=str(ROOT / "data" / "zip_index"), help="for the original photos")
    ap.add_argument("--orig-cache", default=None, help="default: <data-dir>/game_originals.bin")
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-country", type=int, default=20)
    ap.add_argument("--game-size", type=int, default=10)
    ap.add_argument("--max-height", type=int, default=512)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--template", default=str(ROOT / "game" / "template.html"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-label", default="ResNet-18")
    ap.add_argument("--model-note", default=None, help="default: accuracy read from the run's eval/probe files")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    data_dir = a.data_dir or cfg["args"]["data_dir"]
    man, classes, reader, masks = open_data(data_dir)
    if classes != cfg["classes"]:
        raise SystemExit("class list of the data differs from the checkpoint's")
    rows = man[man["split"] == a.split].sample(frac=1, random_state=a.seed).drop_duplicates("creator_id")
    keep = []
    for _, g in rows.groupby("country"):
        picked = [i for i in g.index if usable(reader.read(g.at[i, "id"]))][: a.per_country]
        keep.extend(picked)
    rows = rows.loc[keep].sample(frac=1, random_state=a.seed).reset_index(drop=True)
    n = len(rows) // a.game_size * a.game_size  # round i belongs to game i // game_size
    rows = rows.iloc[:n]

    aug = AugConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["aug"].items()})
    model = CountryNet(len(classes), init="scratch", dropout=cfg["args"].get("dropout", 0.2))
    model.load_state_dict(ck["model"])
    model.eval()
    prep = Prep(aug, train=False)
    logits = torch.zeros(n, len(classes))
    cams = torch.zeros(n, CAM_H, CAM_W)
    for x, _, idx in make_loader(rows, reader, False, aug, masks, 64, 0):
        maps, lg = grad_cam(model, prep(x))
        logits[idx] = lg
        cams[idx] = F.interpolate(maps[:, None], size=(CAM_H, CAM_W), mode="bicubic", align_corners=False)[:, 0]
    prob = torch.softmax(logits, 1).numpy()
    top3 = np.argsort(-prob, axis=1)[:, :3]
    cams = (cams.clamp(0, 1) * 255).round().to(torch.uint8).numpy()

    orig = originals(rows["id"], a.index_dir, a.orig_cache or os.path.join(data_dir, "game_originals.bin"))
    rounds, photos = [], []
    for i, row in rows.iterrows():
        hi, (w, h), crop = game_photo(orig.read(row["id"]), a.max_height, a.quality)
        rounds.append({"id": str(row["id"]), "c": row["country"], "w": w, "h": h,
                       "ai": [[classes[k], round(float(prob[i, k]), 3)] for k in top3[i]]})
        photos.append({"hi": base64.b64encode(hi).decode(), "lo": base64.b64encode(reader.read(row["id"])).decode(),
                       "crop": crop, "cam": base64.b64encode(cams[i].tobytes()).decode()})
    ai_right = np.mean([rd["ai"][0][0] == rd["c"] for rd in rounds])
    data = {
        "version": hashlib.sha1((a.model_label + "|" + ",".join(rd["id"] for rd in rounds)).encode()).hexdigest()[:10],
        "model": {"label": a.model_label, "note": default_note(a.ckpt) if a.model_note is None else a.model_note},
        "classes": [{"code": c, "name": COUNTRIES[c][0], "region": COUNTRIES[c][1]} for c in classes],
        "gameSize": a.game_size,
        "camSize": [CAM_H, CAM_W],
        "rounds": rounds,
    }
    game_dir = Path(a.out).resolve().parent / "g"
    game_dir.mkdir(parents=True, exist_ok=True)
    for old in game_dir.glob("*.json"):
        old.unlink()
    for g in range(n // a.game_size):
        sl = slice(g * a.game_size, (g + 1) * a.game_size)
        (game_dir / f"{g}.json").write_text(json.dumps({rd["id"]: p for rd, p in zip(rounds[sl], photos[sl])}))
    html = Path(a.template).read_text().replace("__GAME_DATA__", json.dumps(data).replace("</", "<\\/"))
    Path(a.out).write_text(html)
    sizes = [p.stat().st_size for p in game_dir.glob("*.json")]
    print(f"{n} photos ({rows['country'].nunique()} countries) in {n // a.game_size} games, AI top-1 on them "
          f"{ai_right:.0%}; wrote {a.out} ({os.path.getsize(a.out) / 1e3:.0f} KB) and {game_dir}/*.json "
          f"(~{np.mean(sizes) / 1e6:.2f} MB each, {sum(sizes) / 1e6:.0f} MB in all)")


if __name__ == "__main__":
    main()
