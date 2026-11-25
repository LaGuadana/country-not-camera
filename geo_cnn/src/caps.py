"""How many images per country survive a per-photographer cap (OSV-5M train)."""
import sys
import pandas as pd

train_csv, out = sys.argv[1:3]
parts = []
for ch in pd.read_csv(train_csv, usecols=["country", "creator_id", "drive_side"], dtype=str, chunksize=500_000):
    parts.append(ch.groupby(["country", "creator_id"]).size())
d = pd.concat(parts).groupby(level=[0, 1]).sum().rename("n").reset_index()
d.to_csv(out, index=False)
for cap in [20, 50, 100, 200]:
    c = d.assign(n=d.n.clip(upper=cap)).groupby("country").agg(img=("n", "sum"), ph=("creator_id", "nunique"))
    line = [f"cap {cap:>3}/photographer:"]
    for need in [2000, 5000, 10000]:
        line.append(f">= {need:,} imgs & >=100 photographers: {((c.img >= need) & (c.ph >= 100)).sum():2d} countries")
    print("  ".join(line))
c = d.assign(n=d.n.clip(upper=100)).groupby("country").agg(img=("n", "sum"), ph=("creator_id", "nunique"))
sel = c[(c.ph >= 200)].sort_values("img", ascending=False)
print(f"\n{len(sel)} countries with >= 200 photographers; images after cap 100:")
print(sel.img.astype(int).to_string())
