"""Ocena na prawdziwych etykietach:
python scripts/06_evaluate.py <katalog wyników> labels.csv reviewed.csv [split=test]

Bierze tylko kadry przejrzane (reviewed.csv) z danego splitu. Kategorie etykiet:
  missing, defocus, changed  -> anomalia (detektor powinien ją znaleźć)
  occluded, uncertain        -> pomijane w ocenie (zasłonięcie to nie awaria lustra; niepewne nie ma prawdy)
  brak etykiety              -> lustro zdrowe (alarm na nim = fałszywy alarm)
Wynik: tabela per wariant + evaluation_per_mirror.csv (wykrycie i opóźnienie dla każdego zdarzenia).
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from magic_mirrors import store

out, lab_path, rev_path = sys.argv[1:4]
split = sys.argv[4] if len(sys.argv) > 4 else "test"
s = pd.read_parquet(f"{out}/scores.parquet")
_, _, meta, _ = store.load_features(out)
s["frame_name"] = s["frame"].map(lambda i: os.path.basename(meta["path"][i]))
lab = pd.read_csv(lab_path)
rev = set(pd.read_csv(rev_path)["frame"])

e = s[(s["split"] == split) & s["frame_name"].isin(rev)]
e = e.merge(lab[["frame", "idx", "label"]].rename(columns={"frame": "frame_name"}), on=["frame_name", "idx"], how="left")
POSITIVE = {"missing", "defocus", "changed"}
IGNORED = {"occluded", "uncertain"}
n_ign = int(e["label"].isin(IGNORED).sum())
e = e[~e["label"].isin(IGNORED)]
e["anomaly"] = e["label"].isin(POSITIVE)
scored = e[e["score_max"].notna()]
print(f"split '{split}': przejrzanych kadrów {e['frame_name'].nunique()}, ocenionych {scored['frame_name'].nunique()}, "
      f"etykiet anomalii {int(e['anomaly'].sum())} (ocenionych {int(scored['anomaly'].sum())}), "
      f"pominiętych (occluded/uncertain) {n_ign}")
if scored.empty or not scored["anomaly"].any():
    sys.exit("Brak ocenionych kadrów z etykietami w tym splicie - sprawdź daty, reviewed.csv i n_refs.")

rows = []
for v in [c[6:] for c in s.columns if c.startswith("score_")]:
    a = scored
    rows.append({
        "wariant": v,
        "AUPRC": average_precision_score(a["anomaly"], a[f"score_{v}"]),
        "AUPRC_losowo": a["anomaly"].mean(),
        "czułość (alarm na lustrze z etykietą)": a.loc[a["anomaly"], f"alarm_{v}"].mean(),
        "fałszywe alarmy na kadr": a[~a["anomaly"]].groupby("frame_name")[f"alarm_{v}"].sum().mean(),
    })
print(pd.DataFrame(rows).round(3).to_string(index=False))

# czułość osobno dla każdej kategorii (ile luster z daną etykietą dostało alarm)
pos = scored[scored["anomaly"]]
per_kind = pd.DataFrame({v: pos.groupby("label")[f"alarm_{v}"].mean() for v in [c[6:] for c in s.columns if c.startswith("score_")]})
per_kind.insert(0, "n", pos.groupby("label").size())
print("\nczułość według kategorii:")
print(per_kind.round(3).to_string())

# zdarzenie = kolejne kadry z etykietą dla tego samego lustra; opóźnienie = kadry do pierwszego alarmu
ev = []
for (idx, label), g in scored[scored["anomaly"]].sort_values("time").groupby(["idx", "label"]):
    for v in ["max", "mahalanobis"]:
        hit = g[f"alarm_{v}"].values
        ev.append({"idx": idx, "label": label, "wariant": v, "od": g["time"].min(), "do": g["time"].max(),
                   "kadrow": len(g), "z_alarmem": int(hit.sum()),
                   "opoznienie_kadrow": int(np.argmax(hit)) if hit.any() else None})
pd.DataFrame(ev).to_csv(f"{out}/evaluation_per_mirror.csv", index=False)
print(f"\nzdarzenia per lustro -> {out}/evaluation_per_mirror.csv")
