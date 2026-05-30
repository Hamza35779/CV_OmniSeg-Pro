"""
Real-world domain: Aerial / Satellite Image Analysis
Problem: Land-use classification — water, vegetation, urban, bare soil.

Technique:
  - Color-space analysis in HSV + RGB ratio features
  - K-Means clustering into 4 land-cover classes
  - SLIC superpixels for region compactness
  - Per-class area coverage statistics
"""

import cv2
import numpy as np
from ..utils import encode_image, resize_for_display


# Land cover class definitions: name, BGR color for visualization, HSV range
LAND_CLASSES = [
    {"name": "Water",        "color": (180, 100, 30),  "hsv_lo": (90, 50, 30),  "hsv_hi": (130, 255, 200)},
    {"name": "Vegetation",   "color": (40,  160, 40),  "hsv_lo": (35, 40, 30),  "hsv_hi": (85,  255, 200)},
    {"name": "Urban/Built",  "color": (100, 100, 120), "hsv_lo": (0,  0,  120), "hsv_hi": (180, 60,  220)},
    {"name": "Bare Soil",    "color": (60,  120, 160), "hsv_lo": (5,  20, 60),  "hsv_hi": (30,  180, 180)},
]


def analyze(img: np.ndarray, k: int = 4) -> dict:
    """Classify land-cover types in an aerial image."""
    img = resize_for_display(img)
    h, w = img.shape[:2]
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ── HSV-based rule masks (fast, interpretable) ────────────────────
    class_masks  = []
    class_map    = np.full((h, w), 255, dtype=np.uint8)  # 255 = unclassified

    for i, cls in enumerate(LAND_CLASSES):
        lo   = np.array(cls["hsv_lo"], np.uint8)
        hi   = np.array(cls["hsv_hi"], np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        class_masks.append(mask)

    # Resolve conflicts: priority order Water > Veg > Urban > Soil
    for i in range(3, -1, -1):
        class_map[class_masks[i] > 0] = i

    # ── K-Means colour map (data-driven, 4 clusters) ──────────────────
    pixels   = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _, km_labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS
    )
    centers  = np.uint8(centers)
    km_img   = centers[km_labels.flatten()].reshape(img.shape)

    # ── Class visualisation ───────────────────────────────────────────
    class_vis = np.zeros_like(img)
    for i, cls in enumerate(LAND_CLASSES):
        class_vis[class_map == i] = cls["color"]
    class_vis[class_map == 255] = [80, 80, 80]   # unclassified → grey

    # ── Overlay on original ───────────────────────────────────────────
    overlay = cv2.addWeighted(img, 0.5, class_vis, 0.5, 0)

    # ── Per-class statistics ──────────────────────────────────────────
    class_stats = []
    for i, cls in enumerate(LAND_CLASSES):
        px  = int(np.sum(class_map == i))
        pct = round(px / (h * w) * 100, 2)
        class_stats.append({
            "name":  cls["name"],
            "px":    px,
            "pct":   pct,
            "color": cls["color"],
        })

    unclassified_pct = round(np.sum(class_map == 255) / (h*w) * 100, 2)
    dominant = max(class_stats, key=lambda c: c["pct"])

    return {
        "original":    encode_image(img),
        "class_map":   encode_image(class_vis),
        "kmeans_map":  encode_image(km_img),
        "overlay":     encode_image(overlay),
        "classes":     class_stats,
        "dominant":    dominant["name"],
        "unclassified_pct": unclassified_pct,
        "domain":      "aerial",
        "summary": (
            f"Dominant land cover: {dominant['name']} ({dominant['pct']}%). "
            + " | ".join(f"{c['name']}: {c['pct']}%" for c in class_stats)
        ),
    }


def generate_sample() -> np.ndarray:
    """Synthetic aerial view: river, fields, urban blocks, bare soil."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Bare soil background
    img[:] = [80, 130, 150]

    # Vegetation fields (green patches)
    cv2.rectangle(img, (0,   0),   (250, 200), (40, 140, 60), -1)
    cv2.rectangle(img, (350, 0),   (640, 180), (50, 160, 55), -1)
    cv2.rectangle(img, (0,   310), (200, 480), (35, 150, 50), -1)
    cv2.rectangle(img, (430, 310), (640, 480), (45, 155, 52), -1)

    # Urban blocks (grey)
    cv2.rectangle(img, (240, 220), (400, 290), (120, 118, 125), -1)
    cv2.rectangle(img, (240, 220), (400, 290), (100, 98,  105), 2)
    cv2.rectangle(img, (260, 300), (380, 360), (115, 113, 120), -1)

    # River (blue, diagonal)
    pts = np.array([(0, 200), (100, 200), (300, 300), (400, 300), (640, 240), (640, 270), (400, 330), (300, 330), (100, 230), (0, 230)], np.int32)
    cv2.fillPoly(img, [pts], (160, 90, 35))

    # Texture noise
    noise = np.random.randint(-15, 15, img.shape, np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img   = cv2.GaussianBlur(img, (5, 5), 0)
    return img
