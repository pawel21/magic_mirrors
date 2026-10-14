"""Wynik anomalii: python scripts/03_score.py <katalog wyników> [config.yaml]
Wynik: scores.parquet (kadr x lustro: wynik każdego wariantu, próg, alarm)."""
import sys
import numpy as np, pandas as pd
from magic_mirrors import load_config, baseline, scoring, store

out = sys.argv[1]
cfg = load_config(sys.argv[2] if len(sys.argv) > 2 else f"{out}/config_used.yaml")
X, sat, meta, names = store.load_features(out)
R, n_refs = baseline.residuals(X, meta, cfg, names)
valid = np.isfinite(R).any(axis=(1, 2))
fit, cal, test = store.make_split(meta, valid, cfg)
print(f"kadry: {len(X)}, z wystarczającą liczbą referencji: {valid.sum()} (fit {fit.sum()}, kalibracja {cal.sum()}, test {test.sum()})")
if cfg["baseline"].get("ref_until"):
    print("referencje zamrożone do", cfg["baseline"]["ref_until"])
if fit.sum() == 0 or cal.sum() == 0:
    sys.exit("Za mało kadrów z referencjami - zwiększ zbiór albo poluzuj baseline w configu.")
N, M = X.shape[:2]
df = pd.DataFrame({"frame": np.repeat(np.arange(N), M), "time": np.repeat(meta["time"].values, M),
                   "idx": np.tile(np.arange(M), N), "n_refs": np.repeat(n_refs, M), "sat_frac": sat.ravel()})
split = np.where(test, "test", np.where(cal, "calibration", np.where(fit, "fit", "")))
df["split"] = np.repeat(split, M)
for name, fn in scoring.VARIANTS.items():
    S = scoring.per_mirror_scale(fn(R, fit, names), cal)
    thr = scoring.threshold(S, cal, cfg["scoring"]["threshold_percentile"])
    df[f"score_{name}"] = S.ravel()
    df[f"alarm_{name}"] = scoring.alarms(S, thr, sat, cfg, meta["time"].values).ravel()
    df[f"thr_{name}"] = thr
    print(f"{name}: próg {thr:.2f}, alarmów {int(df[f'alarm_{name}'].sum())}")
np.savez_compressed(f"{out}/R.npz", R=R, n_refs=n_refs)
df.to_parquet(f"{out}/scores.parquet", index=False)
