"""Cechy per lustro. Każda funkcja dostaje FacetPatch (piksele tylko spod maski) i zwraca liczbę.

Zasady: tło spoza maski nigdy nie wchodzi do statystyk; przed operatorami sąsiedztwa
(Laplasjan, Canny) tło jest wypełniane najbliższym pikselem maski, a wynik czytany
z zerodowanego rdzenia. sat_frac jest liczone zawsze, ale jako veto, nie jako cecha.
"""
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


@dataclass
class FacetPatch:
    bgr: np.ndarray        # wycinek bbox, uint8
    mask: np.ndarray       # bool
    core: np.ndarray       # bool, maska po erozji

    def __post_init__(self):
        self.gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.px = self.gray[self.mask]
        # tło -> najbliższy piksel maski, żeby krawędź maski nie udawała krawędzi w obrazie
        idx = distance_transform_edt(~self.mask, return_distances=False, return_indices=True)
        self.filled = self.gray[idx[0], idx[1]]


def f_median(p):
    return np.median(p.px)


def f_cov(p):
    """std/mean; ok. 0.3 dla zogniskowanego lustra, ok. 0.02 przy utracie blokady."""
    return p.px.std() / (p.px.mean() + 1e-6)


def f_iqr_norm(p):
    q1, q2, q3 = np.percentile(p.px, [25, 50, 75])
    return (q3 - q1) / (q2 + 1e-6)


def f_laplacian_std(p):
    return cv2.Laplacian(p.filled, cv2.CV_32F)[p.core].std()


def f_edge_density(p):
    """Udział pikseli krawędzi (Canny) w rdzeniu; progi względne: hi = P90 gradientu, lo = 0.4*hi."""
    g = p.filled
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    hi = max(np.percentile((np.abs(gx) + np.abs(gy))[p.core], 90), 10.0)
    edges = cv2.Canny(gx.astype(np.int16), gy.astype(np.int16), 0.4 * hi, hi)
    return (edges[p.core] > 0).mean()


def f_br_ratio(p):
    """Mediana B/R: niebo > 1, grunt i konstrukcja < 1. Mało zależy od jasności."""
    b = p.bgr[..., 0][p.mask].astype(np.float32)
    r = p.bgr[..., 2][p.mask].astype(np.float32)
    return np.median((b + 1) / (r + 1))


def f_sat_frac(p):
    return (p.bgr[p.mask].max(axis=1) >= 250).mean()


FEATURES = {
    "median": f_median,
    "cov": f_cov,
    "iqr_norm": f_iqr_norm,
    "laplacian_std": f_laplacian_std,
    "edge_density": f_edge_density,
    "br_ratio": f_br_ratio,
}


def extract(img, facets, names):
    """Kadr BGR -> (X [249, len(names)], sat_frac [249])."""
    X = np.empty((len(facets), len(names)), np.float32)
    sat = np.empty(len(facets), np.float32)
    for i in range(len(facets)):
        y0, y1, x0, x1 = facets.bbox[i]
        p = FacetPatch(img[y0:y1, x0:x1], facets.mask[i], facets.core[i])
        X[i] = [FEATURES[n](p) for n in names]
        sat[i] = f_sat_frac(p)
    return X, sat
