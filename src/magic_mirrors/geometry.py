"""Jedyne źródło geometrii: 249 luster jako poligony, maski i rejestr idx/row/col.

Model: siatka 17x17 bez 4 narożników po 10 luster, na paraboloidzie f = 17 m, rzutowana
kamerą otworkową. 7 parametrów dopasowanych do kadrów z 1-2.04.2024 (pozycja parkingowa,
kadr 830x480). Po przestawieniu kamery PARAMS trzeba dopasować ponownie.
"""
import numpy as np
import cv2
import pandas as pd

F_M = 17.0
CX, CY = 415.0, 240.0
# rvec(3), tvec(3) [m], ogniskowa [px]
PARAMS = [3.01664, -0.19735, -0.84556, 1.78994, -2.015, 55.30507, 851.79333]
N_MIRRORS = 249


def row_span(row):
    """Pierwsza i ostatnia kolumna (włącznie) w wierszu 0..16."""
    k = min(row, 16 - row)
    a = max(0, 4 - k)
    return a, 16 - a


def project(u, v, p=PARAMS):
    """Współrzędne siatki (u kolumna, v wiersz; 0..17 = narożniki luster) -> piksele."""
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    X, Y = u - 8.5, 8.5 - v
    P = np.stack([X, Y, (X * X + Y * Y) / (4 * F_M)], -1).reshape(-1, 3)
    R, _ = cv2.Rodrigues(np.asarray(p[:3], float))
    Pc = P @ R.T + np.asarray(p[3:6])
    xy = np.stack([p[6] * Pc[:, 0] / Pc[:, 2] + CX, p[6] * Pc[:, 1] / Pc[:, 2] + CY], -1)
    return xy.reshape(u.shape + (2,))


def registry():
    """DataFrame: idx (oś tensora cech), row, col. Kolejność wierszami od lewego górnego rogu."""
    rows = [(r, c) for r in range(17) for c in range(row_span(r)[0], row_span(r)[1] + 1)]
    df = pd.DataFrame(rows, columns=["row", "col"])
    df.insert(0, "idx", np.arange(len(df)))
    return df


def polygons(inset=0.0):
    """Lista 249 poligonów (4x2, piksele). inset = margines w ułamku boku lustra."""
    out = []
    for r, c in registry()[["row", "col"]].itertuples(index=False):
        u = np.array([c + inset, c + 1 - inset, c + 1 - inset, c + inset])
        v = np.array([r + inset, r + inset, r + 1 - inset, r + 1 - inset])
        out.append(project(u, v))
    return out


class Facets:
    """Maski liczone raz i używane dla wszystkich kadrów.

    Dla każdego lustra: bbox (y0, y1, x0, x1), maska bool w bboxie i jej zerodowany rdzeń
    (do operatorów sąsiedztwa: Laplasjan, Canny).
    """

    def __init__(self, shape=(480, 830), inset=0.15):
        self.shape, self.inset = shape, inset
        self.polys = polygons(inset)
        self.bbox, self.mask, self.core = [], [], []
        k = np.ones((3, 3), np.uint8)
        for q in self.polys:
            m = np.zeros(shape, np.uint8)
            cv2.fillPoly(m, [np.round(q * 16).astype(np.int32)], 1, shift=4)
            ys, xs = np.nonzero(m)
            y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 2, shape[0])
            x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 2, shape[1])
            sub = m[y0:y1, x0:x1]
            self.bbox.append((y0, y1, x0, x1))
            self.mask.append(sub.astype(bool))
            self.core.append(cv2.erode(sub, k).astype(bool))

    def __len__(self):
        return len(self.polys)

    def label_image(self):
        """Obraz etykiet: 0 = tło, idx+1 = lustro (pełne poligony, bez marginesu)."""
        lab = np.zeros(self.shape, np.uint8)
        for i, q in enumerate(polygons(0.0)):
            cv2.fillPoly(lab, [np.round(q * 16).astype(np.int32)], i + 1, shift=4)
        return lab


def draw_overlay(img, highlight=None, scale=5, crop=(300, 45, 585, 358)):
    """Kadr z siatką luster i numerami idx; highlight = lista idx zaznaczonych na żółto."""
    highlight = set(highlight or [])
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for i, q in enumerate(polygons(0.0)):
        hot = i in highlight
        cv2.polylines(big, [np.round(q * scale * 16).astype(np.int32)], True,
                      (0, 255, 255) if hot else (0, 0, 255), 3 if hot else 1, cv2.LINE_AA, shift=4)
        cx, cy = q.mean(0) * scale
        cv2.putText(big, str(i), (int(cx) - 12, int(cy) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(big, str(i), (int(cx) - 12, int(cy) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    x0, y0, x1, y1 = crop
    return big[y0 * scale:y1 * scale, x0 * scale:x1 * scale]
