import cv2
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd

from matplotlib.patches import Polygon, Patch
from magic_mirrors import geometry, store


i = 1205
CROP = (300, 585, 358, 45)          
POLYS = geometry.polygons()

def vis():
    img = cv2.cvtColor(cv2.imread(meta["path"][i]), cv2.COLOR_BGR2RGB)
        
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_xlim(CROP[0], CROP[1]); ax.set_ylim(CROP[2], CROP[3]); ax.axis("off")
    st = ("#2ecc71", "-") 
    st = ("#ff9f1a", "-")  
    for idx in f[f['alarm_max']].index.tolist():
        q = POLYS[idx]
        ax.add_patch(Polygon(q, closed=True, fill=False, edgecolor=st[0], linestyle=st[1], linewidth=2.2))
    plt.show()

out = "../../output/wyniki/"
s = pd.read_parquet(f"{out}/scores.parquet")
_, _, meta, _ = store.load_features(out)
s["plik"] = s["frame"].map(lambda i: os.path.basename(meta["path"][i]))
t = s[s["split"] == "test"]

file_names_list = list(t['plik'].unique())
for name in file_names_list[25:35]:
    hit = meta.index[meta["path"].str.contains(name)]
    i = hit[0]
    f = s[s["frame"] == i].set_index("idx").reindex(range(249))
    n_alarm = len(f[f["alarm_max"]])
    print(f"i = {i}, {name} -> {n_alarm}")
    if n_alarm > 0:
        print(f[f['alarm_max']].index.tolist())