"""Dobór referencji po położeniu Słońca i podwójna normalizacja.

1) w kadrze: robust z każdej cechy w obrębie strefy (lustra odbijające niebo / grunt),
   co usuwa wspólną zmianę oświetlenia;
2) w czasie: odjęcie typowej wartości danego lustra z kadrów referencyjnych i podzielenie
   przez jej typowy rozrzut, co usuwa stały charakter lustra.
"""
import numpy as np
import pandas as pd
from .frames import angular_sep

MAD = 1.4826


def select_refs(i, meta, cfg):
    """Indeksy kadrów referencyjnych dla kadru i: wcześniejsze, podobne Słońce, ta sama klasa nieba."""
    b = cfg["baseline"]
    t = meta["time"].values
    age = (t[i] - t) / np.timedelta64(1, "D")
    ok = (age >= b["gap_days"]) & (age <= b["lookback_days"])
    ok &= angular_sep(meta["sun_alt"].values[i], meta["sun_az"].values[i],
                      meta["sun_alt"].values, meta["sun_az"].values) <= b["max_sep_deg"]
    if b.get("ref_until"):
        ok &= meta["time"].values < np.datetime64(pd.Timestamp(b["ref_until"]) + pd.Timedelta(days=1), "ns")
    if b["same_sky_class"] and "sky_class" in meta:
        ok &= meta["sky_class"].values == meta["sky_class"].values[i]
    return np.nonzero(ok)[0]


def zones(X_ref, br_col, thr):
    """Strefa lustra z referencji: 1 = odbija niebo, 0 = grunt/konstrukcję. Z referencji, nie
    z kadru testowego, żeby brakujące lustro nie zmieniło sobie grupy odniesienia."""
    return (np.nanmedian(X_ref[:, :, br_col], axis=0) > thr).astype(int)


def robust_z(X, zone):
    """X [..., 249, F] -> z-score po lustrach, osobno w każdej strefie."""
    Z = np.full(X.shape, np.nan, np.float32)
    for z in np.unique(zone):
        s = zone == z
        if s.sum() < 5:
            continue
        med = np.nanmedian(X[..., s, :], axis=-2, keepdims=True)
        mad = MAD * np.nanmedian(np.abs(X[..., s, :] - med), axis=-2, keepdims=True)
        Z[..., s, :] = (X[..., s, :] - med) / (mad + 1e-6)
    return Z


def residual_one(i, X, meta, cfg, names, Xi=None):
    """Reszta [249, F] kadru i względem jego referencji; None, gdy referencji jest za mało.
    Xi pozwala podstawić zmienione cechy kadru (benchmark z wstrzykniętą anomalią).

    Skala lustra ma podłogę = mediana skali po lustrach, bo przy kilkunastu referencjach
    MAD pojedynczego lustra bywa przypadkowo bliski zeru.
    """
    b = cfg["baseline"]
    refs = select_refs(i, meta, cfg)
    if len(refs) < b["min_refs"]:
        return None, len(refs)
    zone = zones(X[refs], names.index("br_ratio"), b["zone_br_threshold"])
    Zr, Zi = robust_z(X[refs], zone), robust_z(X[i] if Xi is None else Xi, zone)
    mu = np.nanmedian(Zr, axis=0)
    sd = MAD * np.nanmedian(np.abs(Zr - mu), axis=0)
    sd = np.maximum(sd, np.nanmedian(sd, axis=0, keepdims=True))
    return (Zi - mu) / (sd + 1e-6), len(refs)


def residuals(X, meta, cfg, names):
    """Reszty wszystkich kadrów: R [N, 249, F] (NaN, gdy za mało referencji) i liczba referencji."""
    R = np.full(X.shape, np.nan, np.float32)
    n_refs = np.zeros(len(X), int)
    for i in range(len(X)):
        r, n_refs[i] = residual_one(i, X, meta, cfg, names)
        if r is not None:
            R[i] = r
    return R, n_refs
