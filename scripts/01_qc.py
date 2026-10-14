"""Selekcja kadrów: python scripts/01_qc.py <plik lub katalog z kadrami> <katalog wyników> [config.yaml]"""
import sys
import cv2, pandas as pd
from magic_mirrors import load_config, frames, qc, store

src, out = sys.argv[1], sys.argv[2]
cfg_path = sys.argv[3] if len(sys.argv) > 3 else "config.yaml"
cfg = load_config(cfg_path)
store.prepare(out, cfg_path)
ref = cv2.cvtColor(cv2.imread(cfg["frame"]["reference_frame"]), cv2.COLOR_BGR2GRAY)
meta = frames.list_frames(src, cfg)
rows, prev = [], None
for path, alt in zip(meta["path"], meta["sun_alt"]):
    img = frames.load(path, cfg)
    rows.append(qc.qc_frame(img, alt, cfg, ref, prev))
    prev = img
df = pd.concat([meta, pd.DataFrame(rows).drop(columns="sun_alt")], axis=1)
store.save_qc(df, out)
print(f"{len(df)} kadrów, dobrych: {(df['reject'] == '').sum()}")
bad = df.loc[df["reject"] != "", "reject"].str.split(",").explode()
if len(bad):
    print(bad.value_counts().to_string())
