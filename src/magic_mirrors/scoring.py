"""Wynik anomalii z reszt R [N, 249, F]. Warianty mają ten sam podpis, żeby benchmark
mógł po nich iterować: score(R, fit, names) -> S [N, 249], gdzie fit = maska kadrów uczących.
"""
import numpy as np
from sklearn.covariance import LedoitWolf


def score_cov(R, fit, names):
    """Tylko jedna cecha: |reszta cov|."""
    return np.abs(R[..., names.index("cov")])


def score_manhattan(R, fit, names):
    return np.abs(R).mean(axis=-1)


def score_max(R, fit, names):
    """Największa pojedyncza reszta; nie rozmywa anomalii widocznej tylko w jednej cesze."""
    return np.abs(R).max(axis=-1)


def score_mahalanobis(R, fit, names):
    """Kowariancja Ledoit-Wolf na resztach kadrów uczących (wspólna dla luster), potem d^2."""
    A = R[fit].reshape(-1, R.shape[-1])
    A = A[np.isfinite(A).all(axis=1)]
    A = np.clip(A, -10, 10)                       # pojedyncze skrajne reszty nie ustawiają kowariancji
    P = LedoitWolf().fit(A).precision_
    return np.einsum("...i,ij,...j->...", R, P, R)


VARIANTS = {"cov": score_cov, "max": score_max, "manhattan": score_manhattan, "mahalanobis": score_mahalanobis}


def per_mirror_scale(S, cal):
    """Dzieli wynik przez jego medianę dla danego lustra na kadrach kalibracyjnych."""
    med = np.nanmedian(S[cal], axis=0)
    return S / (np.where(np.isfinite(med) & (med > 0), med, np.nanmedian(med)) + 1e-9)


def threshold(S, cal, percentile):
    """Próg empiryczny z kadrów kalibracyjnych (nie tych, na których liczono baseline)."""
    return float(np.nanpercentile(S[cal], percentile))


def alarms(S, thr, sat, cfg, times=None):
    """Alarm = wynik powyżej progu w k z n kolejnych kadrów; kadry prześwietlone dla lustra pomijane.
    Jeśli podano times, okno trwałości nie przechodzi przez przerwę dłuższą niż persistence_max_gap_min
    (np. noc albo odrzucone kadry)."""
    c = cfg["scoring"]
    hit = (S > thr) & (sat <= c["sat_frac_veto"])
    k, n = c["persistence_k"], c["persistence_n"]
    if k <= 1:
        return hit
    seg = np.zeros(len(hit), int)
    if times is not None:
        gap = np.diff(np.asarray(times, "datetime64[ns]")) > np.timedelta64(int(c.get("persistence_max_gap_min", 40)), "m")
        seg[1:] = np.cumsum(gap)
    out = np.zeros_like(hit)
    for i in range(len(hit)):
        j0 = max(0, i - n + 1)
        j0 = max(j0, int(np.searchsorted(seg, seg[i])))       # początek bieżącego odcinka bez przerw
        out[i] = hit[j0:i + 1].sum(axis=0) >= min(k, i - j0 + 1)
    return out & hit
