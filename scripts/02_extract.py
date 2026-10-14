"""Cechy luster dla dobrych kadrów: python scripts/02_extract.py <katalog wyników> [config.yaml]
Wynik: X.npz z X [N_kadrów, 249, N_cech] i sat [N_kadrów, 249]."""
import sys
import numpy as np
from joblib import Parallel, delayed
from magic_mirrors import load_config, frames, geometry, features, store

out = sys.argv[1]
cfg = load_config(sys.argv[2] if len(sys.argv) > 2 else f"{out}/config_used.yaml")
qc = store.load_qc(out)
meta = qc[qc["reject"] == ""].reset_index(drop=True)
names = cfg["features"]
facets = geometry.Facets((cfg["frame"]["height"], cfg["frame"]["width"]), cfg["geometry"]["inset"])
res = Parallel(n_jobs=-1)(delayed(lambda p: features.extract(frames.load(p, cfg), facets, names))(p)
                          for p in meta["path"])       # równolegle po kadrach, nie po lustrach
X = np.stack([r[0] for r in res]) if res else np.empty((0, len(facets), len(names)), np.float32)
sat = np.stack([r[1] for r in res]) if res else np.empty((0, len(facets)), np.float32)
store.save_features(out, X, sat, meta, names)
print("X:", X.shape)
