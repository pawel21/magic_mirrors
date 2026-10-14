"""pytest tests/  - szybkie testy, które mają łapać zepsucie geometrii, masek i cech."""
import numpy as np, cv2
from datetime import datetime, timezone
from magic_mirrors import geometry, features, frames, inject

IMG = cv2.imread("data/reference_frame.jpg")
NAMES = list(features.FEATURES)


def test_geometry():
    assert len(geometry.registry()) == 249 == geometry.N_MIRRORS
    lab = geometry.Facets().label_image()
    assert len(np.unique(lab)) == 250
    # przypięcie geometrii: środek lustra 152 (wiersz 10, kol. 2)
    assert np.allclose(geometry.polygons()[152].mean(0), [365.5, 250.6], atol=0.5)


def test_sun():
    alt, az = frames.sun_altaz(datetime(2024, 6, 21, 13, 15, tzinfo=timezone.utc), 28.7616, -17.8908)
    assert abs(alt - 84.7) < 0.3            # górowanie w przesilenie: 90 - 28.76 + 23.44


def test_features_finite_and_mask_only():
    F = geometry.Facets()
    X, sat = features.extract(IMG, F, NAMES)
    assert X.shape == (249, len(NAMES)) and np.isfinite(X).all()
    # zmiana pikseli poza maskami nie może zmienić cech
    inside = np.zeros(IMG.shape[:2], bool)
    for (y0, y1, x0, x1), m in zip(F.bbox, F.mask):
        inside[y0:y1, x0:x1] |= m
    img2 = IMG.copy(); img2[~inside] = 0
    X2, _ = features.extract(img2, F, NAMES)
    assert np.allclose(X, X2)


def test_inject_defocus_lowers_cov():
    F = geometry.Facets(); rng = np.random.default_rng(0)
    X0, _ = features.extract(IMG, F, NAMES)
    X1, _ = features.extract(inject.inject(IMG, 30, "defocus", 1.0, rng), F, NAMES)
    c = NAMES.index("cov")
    assert X1[30, c] < 0.2 * X0[30, c]
    assert np.allclose(np.delete(X0, 30, 0), np.delete(X1, 30, 0))
