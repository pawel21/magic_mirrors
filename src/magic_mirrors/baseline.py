"""Reference selection by Sun position and double normalization.

1) Within a frame: robust z-score of each feature within a zone (mirrors
   reflecting the sky vs. ground), which removes the common illumination
   change.
2) Over time: subtraction of the per-mirror median over the reference
   frames and division by its MAD-based scatter, which removes the constant
   character of each mirror.
"""
import numpy as np
import pandas as pd
from .frames import angular_sep

MAD = 1.4826


def select_refs(i, meta, cfg):
    """Select reference frames for a given frame.

    A frame ``j`` is a valid reference for frame ``i`` if all of the
    following hold:

    * it is older than frame ``i`` by at least ``gap_days`` and at most
      ``lookback_days``;
    * the Sun was within ``max_sep_deg`` of its position in frame ``i``
      (angular separation on the sky);
    * optionally, it was taken before the end of day ``ref_until``;
    * optionally, it has the same sky class as frame ``i``.

    Args:
        i: Positional index (row number) of the target frame in ``meta``.
        meta: ``pandas.DataFrame`` of frames, e.g. the output of
            ``list_frames``. Must contain the columns ``time`` (UTC
            datetimes), ``sun_alt`` and ``sun_az`` (degrees). The column
            ``sky_class`` is required only if ``cfg["baseline"]["same_sky_class"]``
            is true.
        cfg: Configuration mapping. The ``cfg["baseline"]`` section uses:

            * ``gap_days``: minimum age of a reference relative to frame
              ``i``, in days.
            * ``lookback_days``: maximum age of a reference, in days.
            * ``max_sep_deg``: maximum angular separation of the Sun, in
              degrees.
            * ``ref_until``: optional date (anything accepted by
              ``pandas.Timestamp``); references from later than the end of
              this day are excluded. Skipped if missing, ``None`` or empty.
            * ``same_sky_class``: if true, only frames with the same
              ``sky_class`` as frame ``i`` are kept.

    Returns:
        ``numpy.ndarray`` of integer positional indices into ``meta`` of
        the selected reference frames (empty if none qualify). Frame ``i``
        itself is never included as long as ``gap_days > 0``.
    """
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
    """Classify mirror zones from reference frames.

    Marks each position (mirror) as sky-reflecting or not, based on its
    median brightness across the reference frames. The classification is
    derived from the references rather than from the test frame, so that a
    missing or lost mirror in the test frame cannot change its own
    comparison group.

    Args:
        X_ref: Feature array of the reference frames with shape
            ``(n_refs, n_mirrors, n_features)``. NaNs are ignored in the
            median.
        br_col: Index of the brightness feature along the last axis of
            ``X_ref``.
        thr: Brightness threshold. Mirrors with a median brightness strictly
            above ``thr`` are assigned to the sky-reflecting zone.

    Returns:
        ``numpy.ndarray`` of ints with shape ``(n_mirrors,)``: ``1`` for
        mirrors reflecting the sky (median brightness > ``thr``), ``0`` for
        mirrors reflecting the ground or structure. Mirrors with NaN in all
        references get ``0``, because NaN compares as false.
    """
    return (np.nanmedian(X_ref[:, :, br_col], axis=0) > thr).astype(int)


def robust_z(X, zone):
    """Compute robust z-scores across mirrors, separately within each zone.

    For every zone, the median and the scaled median absolute deviation
    (MAD) are computed across the mirrors of that zone, independently for
    each feature and each leading index (e.g. each frame). Each value is
    then standardised as ``(x - median) / (MAD * scale + 1e-6)``. Using the
    median and MAD makes the score insensitive to outliers such as a few
    lost or misaligned mirrors, which is the point of using it for anomaly
    detection.

    Args:
        X: Feature array of shape ``(..., n_mirrors, n_features)``, e.g.
            ``(n_frames, 249, F)`` or ``(249, F)``. NaNs are ignored when
            computing the statistics.
        zone: Integer array of shape ``(n_mirrors,)`` assigning each mirror
            to a zone, e.g. the output of ``zones``. Statistics are
            computed separately for each unique value.

    Returns:
        ``numpy.ndarray`` of dtype ``float32`` with the same shape as ``X``.
        Entries are robust z-scores. Mirrors belonging to a zone with fewer
        than 5 mirrors are left as ``NaN``, as are entries that were
        ``NaN`` in ``X``.

    Note:
        The module-level constant ``MAD`` is the factor that scales the raw
        median absolute deviation (typically ``1.4826`` to make it
        consistent with the standard deviation for normally distributed
        data). A small constant ``1e-6`` is added to the denominator to
        avoid division by zero for constant features.
    """
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
    """Compute the residual of a frame relative to its reference frames.

    Selects reference frames for frame ``i`` (see ``select_refs``), splits
    the mirrors into zones based on the references (see ``zones``), and
    converts both the references and the target frame to robust z-scores
    within each zone (see ``robust_z``). The residual is the deviation of
    the target frame's z-scores from the per-mirror median z-score of the
    references, expressed in units of the per-mirror robust scatter of the
    references.

    The per-mirror scatter is floored at the median scatter across mirrors
    (computed separately for each feature). With only a dozen or so
    references, the MAD of a single mirror can be close to zero by chance,
    which would inflate the residual of that mirror.

    Args:
        i: Positional index of the target frame in ``X`` and ``meta``.
        X: Feature array of shape ``(n_frames, n_mirrors, n_features)``.
        meta: ``pandas.DataFrame`` with one row per frame, aligned with
            ``X`` (columns as required by ``select_refs``).
        cfg: Configuration mapping. Uses ``cfg["baseline"]["min_refs"]``
            (minimum number of references required) and
            ``cfg["baseline"]["zone_br_threshold"]`` (brightness threshold
            for ``zones``), plus the entries used by ``select_refs``.
        names: Sequence of feature names along the last axis of ``X``. Must
            contain ``"br_ratio"``, which is the brightness feature used to
            define zones.
        Xi: Optional replacement features of the target frame, with shape
            ``(n_mirrors, n_features)``. If given, it is used instead of
            ``X[i]``, while the references are still selected for frame
            ``i`` (e.g. for benchmarks with an injected anomaly).

    Returns:
        A tuple ``(residual, n_refs)``:

        * ``residual``: ``numpy.ndarray`` of shape ``(n_mirrors,
          n_features)`` with the residual of the target frame, or ``None``
          if fewer than ``min_refs`` references were found.
        * ``n_refs``: number of selected references (also returned when
          ``residual`` is ``None``).
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
    """Compute residuals for all frames relative to their reference frames.

    Calls ``residual_one`` for every frame in ``X`` and collects the results
    into a single array. Frames for which too few reference frames are
    available (fewer than ``cfg["baseline"]["min_refs"]``) get ``NaN``
    residuals.

    Args:
        X: Feature array of shape ``(n_frames, n_mirrors, n_features)``.
        meta: ``pandas.DataFrame`` with one row per frame, aligned with
            ``X`` (see ``select_refs`` for the required columns).
        cfg: Configuration mapping, as used by ``residual_one``.
        names: Sequence of feature names along the last axis of ``X``; must
            contain ``"br_ratio"``.

    Returns:
        A tuple ``(R, n_refs)``:

        * ``R``: ``numpy.ndarray`` of dtype ``float32`` with the same shape
          as ``X``, holding the residual of each frame. Frames without
          enough references are filled with ``NaN``.
        * ``n_refs``: integer ``numpy.ndarray`` of shape ``(n_frames,)``
          with the number of references selected for each frame.
    """
    R = np.full(X.shape, np.nan, np.float32)
    n_refs = np.zeros(len(X), int)
    for i in range(len(X)):
        r, n_refs[i] = residual_one(i, X, meta, cfg, names)
        if r is not None:
            R[i] = r
    return R, n_refs
