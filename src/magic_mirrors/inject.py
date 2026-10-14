"""Anomalie syntetyczne wstrzykiwane w obraz (nie w cechy), z regulowaną siłą lam w [0, 1]."""
import numpy as np
import cv2
from .geometry import polygons

KINDS = ("defocus", "missing", "swap")


def _mask(shape, poly):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(poly * 16).astype(np.int32)], 1, shift=4)
    return m.astype(bool)


def inject(img, idx, kind, lam, rng, polys=None):
    """Zwraca kopię kadru z anomalią w lustrze idx.

    defocus: piksele mieszane ze swoją średnią (utrata blokady -> jednolite odbicie)
    missing: wklejony fragment tła spoza czaszy (zbocze)
    swap:    wklejony obraz innego lustra (zmiana ustawienia)
    """
    polys = polys or polygons(0.0)
    out = img.astype(np.float32).copy()
    m = _mask(img.shape, polys[idx])
    ys, xs = np.nonzero(m)
    if kind == "defocus":
        src = np.broadcast_to(out[m].mean(axis=0), out[m].shape)
    elif kind == "missing":
        dy, dx = 215 + rng.integers(0, 60) - ys.min(), 625 + rng.integers(0, 150) - xs.min()
        src = out[ys + dy, xs + dx]
    elif kind == "swap":
        j = int(rng.choice([k for k in range(len(polys)) if k != idx]))
        d = np.round(polys[j].mean(0) - polys[idx].mean(0)).astype(int)
        src = out[np.clip(ys + d[1], 0, img.shape[0] - 1), np.clip(xs + d[0], 0, img.shape[1] - 1)]
    else:
        raise ValueError(kind)
    out[m] = (1 - lam) * out[m] + lam * src
    return np.clip(out, 0, 255).astype(np.uint8)
