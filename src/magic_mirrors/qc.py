"""Selekcja kadrów. Miary liczone poza czaszą; czasza nie decyduje o odrzuceniu kadru.

qc_frame zwraca słownik miar + 'reject' (pusty string = kadr dobry) + 'sky_class'.
Nic nie jest kasowane: wynik trafia do tabeli, żeby dało się policzyć, ile kadrów odpadło i dlaczego.
"""
import numpy as np
import cv2
from .geometry import project, row_span


def _roi(img, r):
    x0, y0, x1, y1 = r
    return img[y0:y1, x0:x1]


def ridge_sharpness(gray, rois):
    """Wariancja Laplasjanu na kopułach i grzbiecie; spada we mgle i przy kroplach na szybie."""
    return float(np.mean([cv2.Laplacian(_roi(gray, r), cv2.CV_32F).var() for r in rois]))


def dark_channel(img, roi, win=7):
    """Mediana dark channel (min z RGB w oknie) na zboczu; rośnie we mgle."""
    dc = cv2.erode(_roi(img, roi).min(axis=2), np.ones((win, win), np.uint8))
    return float(np.median(dc))


def sky_stats(img, rois, white_thr=1.3):
    """Średnia B/R w ROI nieba i udział pikseli białawych (B/R < white_thr) = przybliżony
    udział chmur. Samo std B/R się nie nadaje: czyste niebo ma duży gradient ku horyzontowi."""
    v = np.concatenate([((_roi(img, r)[..., 0].astype(np.float32) + 1) /
                         (_roi(img, r)[..., 2].astype(np.float32) + 1)).ravel() for r in rois])
    return float(v.mean()), float((v < white_thr).mean())


def camera_shift(gray, ref_gray, rois):
    """Przesunięcie kamery [px] względem kadru wzorcowego, korelacja fazowa na tle statycznym."""
    d = []
    for r in rois:
        a, b = _roi(ref_gray, r).astype(np.float32), _roi(gray, r).astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
        d.append(np.hypot(dx, dy))
    return float(np.median(d))


def grid_contrast(gray, rows=range(11, 16), n=40):
    """Czy czasza stoi tam, gdzie zakłada model: odpowiedź filtra ciemnych linii na szczelinach
    między lustrami podzielona przez odpowiedź pół lustra obok. ~1 = siatki nie ma w tym miejscu.
    Liczone w dolnej (odbijającej niebo) części czaszy, kolumny 1..7, gdzie nic nie zasłania.
    """
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((5, 5), np.uint8)).astype(np.float32)

    def sample(u, v):
        p = project(u, v)
        x = np.clip(np.round(p[..., 0]).astype(int), 0, gray.shape[1] - 1)
        y = np.clip(np.round(p[..., 1]).astype(int), 0, gray.shape[0] - 1)
        return bh[y, x]

    on, off = [], []
    t = np.linspace(0.15, 0.85, n)
    for r in rows:
        a = max(row_span(r)[0], row_span(r - 1)[0]) + 1
        for c in range(a, 7):
            on.append(sample(c + t, np.full(n, float(r))))        # szczelina pozioma
            off.append(sample(c + t, np.full(n, r + 0.5)))
            on.append(sample(np.full(n, float(c)), r + t))        # szczelina pionowa
            off.append(sample(np.full(n, c + 0.5), r + t))
    return float(np.mean(on) / (np.mean(off) + 1e-6))


def qc_frame(img, sun_alt, cfg, ref_gray=None, prev_img=None):
    q, c = cfg["qc"], cfg["qc"]["rois"]
    out = {"sun_alt": sun_alt}
    if img is None:
        return {**out, "reject": "unreadable", "sky_class": ""}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out["frozen"] = bool(prev_img is not None and np.array_equal(img, prev_img))
    out["ridge_sharpness"] = ridge_sharpness(gray, c["ridge"])
    out["dark_channel"] = dark_channel(img, c["terrain"])
    out["sky_br_mean"], out["cloud_frac"] = sky_stats(img, c["sky"])
    out["shift_px"] = camera_shift(gray, ref_gray, c["static"]) if ref_gray is not None else np.nan
    out["grid_contrast"] = grid_contrast(gray)
    out["sky_class"] = "cloudy" if out["cloud_frac"] > q["max_clear_cloud_frac"] else "clear"
    reasons = []
    if out["frozen"]:
        reasons.append("frozen")
    if sun_alt < q["min_sun_alt_deg"]:
        reasons.append("sun_low")
    if out["ridge_sharpness"] < q["min_ridge_sharpness"]:
        reasons.append("blurred")
    if out["dark_channel"] > q["max_dark_channel"]:
        reasons.append("haze")
    if out["shift_px"] > q["max_shift_px"]:
        reasons.append("camera_shift")
    if out["grid_contrast"] < q["min_grid_contrast"]:
        reasons.append("no_grid")
    out["reject"] = ",".join(reasons)
    return out
