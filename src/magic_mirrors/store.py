"""Zapis i odczyt wyników pośrednich. Etapy rozmawiają przez pliki, nie przez pamięć."""
import os
import shutil
import sqlite3
import numpy as np
import pandas as pd


def prepare(out_dir, config_path):
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "config_used.yaml")
    if os.path.abspath(config_path) != os.path.abspath(dst):
        shutil.copyfile(config_path, dst)


def save_qc(df, out_dir):
    d = df.copy()
    d["time"] = d["time"].astype(str)
    with sqlite3.connect(os.path.join(out_dir, "qc.sqlite")) as con:
        d.to_sql("qc", con, if_exists="replace", index=False)


def load_qc(out_dir):
    with sqlite3.connect(os.path.join(out_dir, "qc.sqlite")) as con:
        d = pd.read_sql("select * from qc", con)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    return d


def save_features(out_dir, X, sat, meta, names):
    np.savez_compressed(os.path.join(out_dir, "X.npz"), X=X, sat=sat, names=np.array(names, dtype=str),
                        path=np.array(list(meta["path"]), dtype=str), time=np.array([str(t) for t in meta["time"]], dtype=str),
                        sun_alt=meta["sun_alt"].values, sun_az=meta["sun_az"].values,
                        sky_class=np.array(list(meta["sky_class"]), dtype=str))


def load_features(out_dir):
    d = np.load(os.path.join(out_dir, "X.npz"))
    meta = pd.DataFrame({"path": d["path"], "time": pd.to_datetime(d["time"], utc=True),
                         "sun_alt": d["sun_alt"], "sun_az": d["sun_az"], "sky_class": d["sky_class"]})
    return d["X"], d["sat"], meta, list(d["names"])


def chrono_split(valid, fractions):
    """Maski fit / cal / test po kolei w czasie, tylko wśród kadrów z policzoną resztą."""
    idx = np.nonzero(valid)[0]
    a = int(round(len(idx) * fractions[0]))
    b = int(round(len(idx) * (fractions[0] + fractions[1])))
    out = []
    for part in (idx[:a], idx[a:b], idx[b:]):
        m = np.zeros(len(valid), bool)
        m[part] = True
        out.append(m)
    return out


def date_mask(meta, rng):
    """Maska kadrów z zakresu dat [od, do] (włącznie, po dacie UTC)."""
    d = meta["time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    return ((d >= pd.Timestamp(rng[0])) & (d <= pd.Timestamp(rng[1]))).values


def make_split(meta, valid, cfg):
    """fit / cal / test: po datach z cfg['split'], jeśli podane, inaczej chronologicznie w proporcjach."""
    sp = cfg.get("split") or {}
    if sp.get("calibration") and sp.get("test"):
        cal = date_mask(meta, sp["calibration"]) & valid
        test = date_mask(meta, sp["test"]) & valid
        if sp.get("fit"):
            fit = date_mask(meta, sp["fit"]) & valid
        else:
            fit = valid & (meta["time"].values < meta["time"].values[cal].min()) if cal.any() else np.zeros_like(valid)
        return fit, cal, test
    return chrono_split(valid, cfg["benchmark"]["split"])
