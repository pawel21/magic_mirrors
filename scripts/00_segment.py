"""Segment mirrors in a single frame.

Writes the mirror outlines, a label image and a visual overlay.

Usage:
    python scripts/00_segment.py frame.jpg [output_dir]

Outputs:
    mirrors.geojson: Mirror outlines as GeoJSON.
    mirrors_labels.png: Label image (0 = background, idx + 1 = mirror).
    mirrors_overlay.png: Overlay of the segmentation on the frame.
"""
import sys, os, json
import cv2, numpy as np
from magic_mirrors import geometry

img_path, out = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(out, exist_ok=True)
img = cv2.imread(img_path)
assert img is not None and img.shape[:2] == (480, 830), "oczekiwany kadr 830x480"
reg = geometry.registry()
feats = [{"type": "Feature", "properties": {"idx": int(i), "row": int(r), "col": int(c)},
          "geometry": {"type": "Polygon", "coordinates": [np.round(np.vstack([q, q[:1]]), 2).tolist()]}}
         for (i, r, c), q in zip(reg.itertuples(index=False), geometry.polygons())]
json.dump({"type": "FeatureCollection", "features": feats}, open(os.path.join(out, "mirrors.geojson"), "w"))
cv2.imwrite(os.path.join(out, "mirrors_labels.png"), geometry.Facets().label_image())
cv2.imwrite(os.path.join(out, "mirrors_overlay.png"), geometry.draw_overlay(img))
print("zapisano 249 luster do", out)
