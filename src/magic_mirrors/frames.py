"""Frame I/O and solar geometry utilities.

Loads image frames, parses UTC timestamps from file names, and computes the
Sun's altitude and azimuth using NOAA formulas (no astropy dependency).
"""

import os
import re
import glob
from datetime import datetime, timezone
import numpy as np
import cv2
import pandas as pd


def parse_time(path, regex=r"(\d{8})_(\d{4})"):
    """Extract a UTC timestamp from an image file name.

    The timestamp is expected in the base name of the file as
    ``YYYYMMDD_HHMM`` (e.g. ``20240401_0940_la.jpg``). Directory components
    are ignored; the first match is used and seconds are set to zero.

    Args:
        path: Path to the file (or just its name) carrying the timestamp.
        regex: Pattern with two capture groups: date (``YYYYMMDD``) and
            time (``HHMM``).

    Returns:
        Timezone-aware ``datetime`` in UTC, or ``None`` if the file name
        does not match ``regex``.

    Raises:
        ValueError: If the matched digits are not a valid date/time.
    """
    m = re.search(regex, os.path.basename(path))
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def sun_altaz(t, lat, lon):
    """Compute the Sun's altitude and azimuth for a given time and location.

    Uses the NOAA solar position formulas (low-precision algorithm based on
    Meeus). Accuracy is of the order of 0.1 deg. Atmospheric refraction is not
    included, which is sufficient for selecting reference images with a
    tolerance of a few degrees.

    Args:
        t: Observation time as a timezone-aware ``datetime`` in UTC (e.g. the
            output of ``parse_time``). The UTC hour, minute and second are
            used directly, so a non-UTC or naive ``datetime`` gives wrong results.
        lat: Geodetic latitude of the observer in degrees, positive north.
        lon: Longitude of the observer in degrees, positive east.

    Returns:
        A tuple ``(alt, az)`` of floats in degrees:

        * ``alt``: geometric altitude above the horizon, in [-90, 90];
          negative when the Sun is below the horizon.
        * ``az``: azimuth measured from north through east, in [0, 360).

    Example:
        >>> alt, az = sun_altaz(parse_time("20240401_0940_la.jpg"), 28.76, -17.89)
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
    """Compute the angular separation between two positions on the sky.

    Uses the spherical law of cosines on horizontal coordinates. The input
    is vectorised: scalars and NumPy arrays (with broadcastable shapes) are
    both accepted. The cosine is clipped to [-1, 1] to guard against
    round-off errors that would otherwise make ``arccos`` return NaN for
    (nearly) identical or antipodal positions.

    Args:
        alt1: Altitude of the first position, in degrees.
        az1: Azimuth of the first position, in degrees.
        alt2: Altitude of the second position, in degrees.
        az2: Azimuth of the second position, in degrees. Azimuths must use
            the same convention for both positions (e.g. from north through
            east, as returned by ``sun_altaz``).

    Returns:
        Angular separation in degrees, in [0, 180]. A float for scalar
        input, otherwise a ``numpy.ndarray`` with the broadcast shape of
        the inputs.

    Example:
        >>> angular_sep(45.0, 90.0, 45.0, 90.0)
        0.0
    """
    a1, z1, a2, z2 = map(np.radians, (alt1, az1, alt2, az2))
    c = np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(z1 - z2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def list_frames(path, cfg):
    """List image frames with their timestamps and Sun positions.

    Accepts either a single image file or a directory. For a directory, all
    ``*.jpg`` files are collected recursively. Frames whose file name does not
    match the timestamp pattern are silently skipped. For each remaining frame
    the Sun's altitude and azimuth at the observation site are computed.

    Args:
        path: Path to a single image file or to a directory searched
            recursively for ``*.jpg`` files.
        cfg: Configuration mapping with the following entries:

            * ``cfg["frame"]["filename_regex"]``: regex passed to
              ``parse_time`` (two capture groups: date and time).
            * ``cfg["site"]["lat"]``: site latitude in degrees, positive north.
            * ``cfg["site"]["lon"]``: site longitude in degrees, positive east.

    Returns:
        ``pandas.DataFrame`` sorted chronologically, with a fresh
        ``RangeIndex`` and the columns:

        * ``path``: path to the image file.
        * ``time``: timezone-aware UTC ``datetime`` parsed from the file name.
        * ``sun_alt``: Sun altitude in degrees (negative below the horizon).
        * ``sun_az``: Sun azimuth in degrees, from north through east.

        The DataFrame is empty (but keeps these columns) if no file matches.

    Raises:
        ValueError: If a file name matches the regex but encodes an invalid
            date/time (propagated from ``parse_time``).
    """
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
    """Load an image frame from disk, optionally validating its size.

    Reads the file with OpenCV, so the result is in BGR channel order. If
    ``cfg`` is given, the image dimensions must match the expected frame size;
    frames with a different size are rejected. Unreadable or corrupted files
    are also rejected, without raising an exception.

    Args:
        path: Path to the image file.
        cfg: Optional configuration mapping. If provided, the image shape
            must equal ``(cfg["frame"]["height"], cfg["frame"]["width"])``.
            If ``None`` (default), no size check is performed.

    Returns:
        The image as a ``numpy.ndarray`` of shape ``(H, W, 3)`` and dtype
        ``uint8`` in BGR order, or ``None`` if the file cannot be read
        (missing, corrupted, unsupported format) or its size does not match
        the configuration.
    """
    im = cv2.imread(path)
    if im is None:
        return None
    if cfg and im.shape[:2] != (cfg["frame"]["height"], cfg["frame"]["width"]):
        return None
    return im
