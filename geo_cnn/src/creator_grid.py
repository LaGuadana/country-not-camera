"""Rows of images by the same Mapillary creator, to show the camera 'fingerprint'."""
import sys, zipfile, io
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

idx_csv, zip_path, out = sys.argv[1:4]
df = pd.read_csv(idx_csv, dtype=str)
# creators with many images spread over several sequences, one per country
g = df.groupby(["creator_id", "country"]).agg(n=("id", "size"), seqs=("sequence", "nunique")).reset_index()
g = g[(g.n >= 8) & (g.seqs >= 4)].sort_values("n", ascending=False)
picked, seen = [], set()
for _, r in g.iterrows():
    if r.country in seen:
        continue
    picked.append(r); seen.add(r.country)
    if len(picked) == 3:
        break
fig, axes = plt.subplots(len(picked), 4, figsize=(10, 2.1 * len(picked)))
with zipfile.ZipFile(zip_path) as zf:
    for i, r in enumerate(picked):
        sub = df[(df.creator_id == r.creator_id)].drop_duplicates("sequence").head(4)
        for j, (_, row) in enumerate(sub.iterrows()):
            im = Image.open(io.BytesIO(zf.read(row.member))).convert("RGB")
            ax = axes[i, j]
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
        axes[i, 0].set_ylabel(f"photographer {i+1}\n({r.country})", fontsize=9, family="serif")
fig.suptitle("Same photographer, different streets (4 different trips each)", x=0.01, ha="left",
             fontsize=10, family="serif", weight="bold")
fig.tight_layout()
fig.savefig(out, dpi=130)
print([(r.creator_id, r.country, r.n, r.seqs) for r in picked])
