"""Ile referencji mają kadry testowe, dzień po dniu: python scripts/check_refs.py <katalog wyników>"""
import sys
import pandas as pd
import yaml

out = sys.argv[1]
cfg = yaml.safe_load(open(f"{out}/config_used.yaml"))
s = pd.read_parquet(f"{out}/scores.parquet")
f = s.groupby("frame").agg(time=("time", "first"), split=("split", "first"), n_refs=("n_refs", "first"))
f["oceniony"] = f["n_refs"] >= cfg["baseline"]["min_refs"]
t = f[f["split"] == "test"]
if t.empty:
    sys.exit("Brak kadrów w splicie 'test' - sprawdź daty w config.yaml.")
day = t.groupby(t["time"].dt.date).agg(kadrow=("n_refs", "size"), ocenionych=("oceniony", "sum"),
                                        mediana_refs=("n_refs", "median"))
print(day.to_string())
print(f"\nocenionych kadrów testowych: {int(t['oceniony'].sum())} z {len(t)}")