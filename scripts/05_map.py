"""Mapa luster pokolorowana odsetkiem alarmów:
python scripts/05_map.py <katalog wyników> [wariant=max] [liczba podpisanych luster=15]
Wynik: mirror_map.png w katalogu wyników."""
import sys
import cv2, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from magic_mirrors import geometry, store

out = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else "max"
n_label = int(sys.argv[3]) if len(sys.argv) > 3 else 15

s = pd.read_parquet(f"{out}/scores.parquet")
s = s[s[f"score_{variant}"].notna()]
frac = s.groupby("idx")[f"alarm_{variant}"].mean().reindex(range(249)).fillna(0).values

_, _, meta, _ = store.load_features(out)
img = cv2.cvtColor(cv2.imread(meta["path"].iloc[-1]), cv2.COLOR_BGR2RGB)   # tło: ostatni kadr
polys = geometry.polygons()

fig, ax = plt.subplots(1, 2, figsize=(15, 8))
for a in ax:
    a.imshow(img); a.set_xlim(300, 585); a.set_ylim(358, 45); a.axis("off")
ax[0].set_title("kadr")
pc = PolyCollection(polys, array=frac, cmap="magma", edgecolors="white", linewidths=0.3, alpha=0.9)
pc.set_clim(0, max(frac.max(), 1e-6))
ax[1].add_collection(pc)
ax[1].set_title(f"odsetek kadrów z alarmem ({variant})")
fig.colorbar(pc, ax=ax[1], fraction=0.046)
for i in np.argsort(-frac)[:n_label]:
    if frac[i] > 0:
        x, y = polys[i].mean(0)
        ax[1].text(x, y, str(i), ha="center", va="center", fontsize=7, color="cyan", weight="bold")
fig.tight_layout()
fig.savefig(f"{out}/mirror_map.png", dpi=130)
print("zapisano", f"{out}/mirror_map.png")
print(pd.Series(frac).sort_values(ascending=False).head(n_label).round(3).to_string())
