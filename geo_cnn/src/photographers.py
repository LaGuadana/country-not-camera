"""Photographers (Mapillary creators) per country in OSV-5M train/test metadata."""
import sys
import numpy as np
import pandas as pd

train_csv, test_csv, out_csv = sys.argv[1:4]
cols = ["country", "creator_id", "sequence"]

def counts(path):
    parts = []
    for ch in pd.read_csv(path, usecols=cols, dtype=str, chunksize=500_000):
        ch["creator_id"] = ch["creator_id"].fillna("NA")
        parts.append(ch.groupby(["country", "creator_id"]).size())
    s = pd.concat(parts).groupby(level=[0, 1]).sum()
    return s.rename("n").reset_index()

tr, te = counts(train_csv), counts(test_csv)
for name, d in [("train", tr), ("test", te)]:
    print(f"{name}: images {d.n.sum():,}  photographers {d.creator_id.nunique():,}  "
          f"missing-creator images {d.loc[d.creator_id=='NA','n'].sum():,}")
    per = d.groupby("creator_id").n.sum().sort_values(ascending=False)
    print(f"  images/photographer median {per.median():.0f}, top1 {per.iloc[0]:,} "
          f"({per.iloc[0]/per.sum():.1%}), top10 share {per.iloc[:10].sum()/per.sum():.1%}, "
          f"top100 share {per.iloc[:100].sum()/per.sum():.1%}")
trc, tec = set(tr.creator_id), set(te.creator_id)
shared = trc & tec
te_img_shared = te.loc[te.creator_id.isin(shared), "n"].sum()
print(f"photographers in both train and test: {len(shared):,} of {len(tec):,} test photographers; "
      f"test images by a train photographer: {te_img_shared:,} / {te.n.sum():,} ({te_img_shared/te.n.sum():.1%})")

def summary(d, tag):
    g = d.groupby("country")
    out = pd.DataFrame({
        f"{tag}_images": g.n.sum(),
        f"{tag}_photographers": g.creator_id.nunique(),
        f"{tag}_top1_share": g.n.max() / g.n.sum(),
        # effective number of photographers: 1 / sum(p_i^2) (inverse Simpson)
        f"{tag}_eff_photographers": g.n.apply(lambda x: 1.0 / np.sum((x / x.sum()) ** 2)),
    })
    return out

S = summary(tr, "train").join(summary(te, "test"), how="outer")
te_new = te[~te.creator_id.isin(trc)].groupby("country").n.sum()
S["test_images_new_photographer"] = te_new
S = S.sort_values("train_images", ascending=False)
S.to_csv(out_csv)
pd.set_option("display.width", 200)
fmt = S[["train_images", "train_photographers", "train_eff_photographers", "train_top1_share",
         "test_images", "test_photographers", "test_images_new_photographer"]].copy()
print(fmt.head(60).round(2).to_string())
for k in [50, 100, 200, 500]:
    sel = S[(S.train_photographers >= k)]
    print(f"countries with >= {k} train photographers: {len(sel)}")
for k in [20, 50, 100]:
    sel = S[(S.train_eff_photographers >= k)]
    print(f"countries with effective photographers >= {k}: {len(sel)}")
