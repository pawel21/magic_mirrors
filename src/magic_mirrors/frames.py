"""Wczytanie kadru, czas z nazwy pliku i położenie Słońca (bez astropy)."""
import os
import re
import glob
from datetime import datetime, timezone
import numpy as np
import cv2
import pandas as pd


def parse_time(path, regex=r"(\d{8})_(\d{4})"):
    """Czas UTC z nazwy pliku, np. 20240401_0940_la.jpg. Zwraca None, gdy brak wzorca."""
    m = re.search(regex, os.path.basename(path))
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def sun_altaz(t, lat, lon):
    """Wysokość i azymut Słońca [deg] (azymut od północy przez wschód).

    Wzory NOAA; dokładność rzędu 0.1 deg, bez refrakcji - wystarcza do doboru referencji
    z tolerancją kilku stopni.
    """
    jd = t.timestamp() / 86400.0 + 2440587.5
    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + T * (36000.76983 + 0.0003032 * T)) % 360
    M = np.radians(357.52911 + T * (35999.05029 - 0.0001537 * T))
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    C = ((1.914602 - T * (0.004817 + 0.000014 * T)) * np.sin(M)
         + (0.019993 - 0.000101 * T) * np.sin(2 * M) + 0.000289 * np.sin(3 * M))
    om = np.radians(125.04 - 1934.136 * T)
    lam = np.radians(L0 + C - 0.00569 - 0.00478 * np.sin(om))
    eps = np.radians(23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60
                     + 0.00256 * np.cos(om))
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    y = np.tan(eps / 2) ** 2
    L0r = np.radians(L0)
    eot = 4 * np.degrees(y * np.sin(2 * L0r) - 2 * e * np.sin(M) + 4 * e * y * np.sin(M) * np.cos(2 * L0r)
                         - 0.5 * y * y * np.sin(4 * L0r) - 1.25 * e * e * np.sin(2 * M))   # minuty
    minutes = t.hour * 60 + t.minute + t.second / 60
    ha = np.radians(((minutes + eot + 4 * lon) % 1440) / 4 - 180)
    la = np.radians(lat)
    alt = np.arcsin(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha))
    az = np.arctan2(np.sin(ha), np.cos(ha) * np.sin(la) - np.tan(dec) * np.cos(la)) + np.pi
    return float(np.degrees(alt)), float(np.degrees(az) % 360)


def angular_sep(alt1, az1, alt2, az2):
    """Separacja kątowa [deg] między dwoma położeniami na niebie (wektorowo na tablicach)."""
    a1, z1, a2, z2 = map(np.radians, (alt1, az1, alt2, az2))
    c = np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(z1 - z2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def list_frames(path, cfg):
    """Ścieżka do pliku albo katalogu -> DataFrame: path, time, sun_alt, sun_az (chronologicznie)."""
    files = [path] if os.path.isfile(path) else sorted(
        glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True))
    rows = []
    for f in files:
        t = parse_time(f, cfg["frame"]["filename_regex"])
        if t is None:
            continue
        alt, az = sun_altaz(t, cfg["site"]["lat"], cfg["site"]["lon"])
        rows.append((f, t, alt, az))
    df = pd.DataFrame(rows, columns=["path", "time", "sun_alt", "sun_az"])
    return df.sort_values("time").reset_index(drop=True)


def load(path, cfg=None):
    """Kadr BGR uint8 albo None, gdy plik jest uszkodzony lub ma zły rozmiar."""
    im = cv2.imread(path)
    if im is None:
        return None
    if cfg and im.shape[:2] != (cfg["frame"]["height"], cfg["frame"]["width"]):
        return None
    return im
