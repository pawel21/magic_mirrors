"""Benchmark na anomaliach syntetycznych: python scripts/04_benchmark.py <katalog wyników> [config.yaml]

Podział chronologiczny fit / kalibracja / test. W każdy kadr testowy wstrzykiwane są anomalie
(typ x lambda), cechy liczone od nowa z obrazu, wynik porównywany z czystymi lustrami tego kadru.
Wynik: benchmark.csv (wariant x typ x lambda: AUPRC, czułość przy progu z kalibracji, FP na kadr).
"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from magic_mirrors import load_config, frames, geometry, features, baseline, scoring, inject, store

out = sys.argv[1]
cfg = load_config(sys.argv[2] if len(sys.argv) > 2 else f"{out}/config_used.yaml")
b = cfg["benchmark"]
rng = np.random.default_rng(b["seed"])
X, sat, meta, names = store.load_features(out)
R, _ = baseline.residuals(X, meta, cfg, names)
valid = np.isfinite(R).any(axis=(1, 2))
fit, cal, test = store.make_split(meta, valid, cfg)
if min(fit.sum(), cal.sum(), test.sum()) == 0:
    sys.exit("Za mało kadrów z referencjami na podział fit / kalibracja / test.")
facets = geometry.Facets((cfg["frame"]["height"], cfg["frame"]["width"]), cfg["geometry"]["inset"])
polys = geometry.polygons(0.0)

# reszty kadrów testowych z anomaliami: lista (kind, lam, frame, labels[249], R[249, F])
cases = []
for i in np.nonzero(test)[0]:
    img = frames.load(meta["path"][i], cfg)
    for kind in inject.KINDS:
        for lam in b["lambdas"]:
            idx = rng.choice(len(polys), b["n_inject_per_frame"], replace=False)
            im = img
            for k in idx:
                im = inject.inject(im, int(k), kind, lam, rng, polys)
            Xi, _ = features.extract(im, facets, names)
            Ri, _ = baseline.residual_one(i, X, meta, cfg, names, Xi=Xi)
            y = np.zeros(len(polys), bool); y[idx] = True
            cases.append((kind, lam, i, y, Ri))

rows = []
for name, fn in scoring.VARIANTS.items():
    S_all = fn(R, fit, names)
    med = np.nanmedian(S_all[cal], axis=0)
    med = np.where(np.isfinite(med) & (med > 0), med, np.nanmedian(med)) + 1e-9
    thr = scoring.threshold(S_all / med, cal, cfg["scoring"]["threshold_percentile"])
    fp_per_frame = float(((S_all[test] / med) > thr).sum(axis=1).mean())
    Rc = np.stack([c[4] for c in cases])
    Sc = fn(np.concatenate([R[fit], Rc]), np.r_[np.ones(fit.sum(), bool), np.zeros(len(Rc), bool)], names)[fit.sum():] / med
    for kind in inject.KINDS:
        for lam in b["lambdas"]:
            sel = [j for j, c in enumerate(cases) if c[0] == kind and c[1] == lam]
            y = np.concatenate([cases[j][3] for j in sel]); s = np.concatenate([Sc[j] for j in sel])
            ok = np.isfinite(s)
            rows.append({"variant": name, "kind": kind, "lambda": lam, "n_frames": len(sel),
                         "auprc": average_precision_score(y[ok], s[ok]), "auprc_chance": y[ok].mean(),
                         "recall_at_thr": float((s[ok][y[ok]] > thr).mean()), "fp_per_clean_frame": fp_per_frame})
res = pd.DataFrame(rows)
res.to_csv(f"{out}/benchmark.csv", index=False)
print(res.pivot_table(index=["variant", "kind"], columns="lambda", values="auprc").round(3).to_string())
print("\nFP na czysty kadr testowy:", res.groupby("variant")["fp_per_clean_frame"].first().round(2).to_dict())
