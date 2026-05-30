"""
Real-world domain: Agriculture & Crop Analysis
Problem: Vegetation health mapping, disease zone detection, crop vs. weed segmentation.

Technique:
  - Excess Green Index (ExG) as a vegetation proxy (ExG = 2G - R - B)
  - K-Means clustering into: healthy crop, diseased, soil, weed
  - Disease severity estimation from brown/yellow pixel ratio
  - Coverage statistics
"""

import cv2
import numpy as np
from ..utils import encode_image, resize_for_display


def analyze(img: np.ndarray) -> dict:
    """Map vegetation health and segment crop regions."""
    img = resize_for_display(img)
    h, w = img.shape[:2]

    b = img[:,:,0].astype(np.float32)
    g = img[:,:,1].astype(np.float32)
    r = img[:,:,2].astype(np.float32)

    # ── Excess Green Index (vegetation proxy) ─────────────────────────
    exg = 2*g - r - b
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    exg_color = cv2.applyColorMap(exg_norm, cv2.COLORMAP_SUMMER)

    # ── Vegetation mask (positive ExG = plant material) ───────────────
    veg_mask = (exg > 10).astype(np.uint8) * 255
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Disease detection (yellow/brown areas within vegetation) ───────
    hsv           = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask   = cv2.inRange(hsv, (15, 50, 50),  (35, 255, 255))
    brown_mask    = cv2.inRange(hsv, (5,  40, 30),  (20, 200, 180))
    disease_mask  = cv2.bitwise_or(yellow_mask, brown_mask)
    disease_mask  = cv2.bitwise_and(disease_mask, veg_mask)

    # ── Healthy vegetation ─────────────────────────────────────────────
    healthy_mask = cv2.subtract(veg_mask, disease_mask)

    # ── Soil (non-vegetation) ──────────────────────────────────────────
    soil_mask = cv2.bitwise_not(veg_mask)

    # ── K-means 4-class map ───────────────────────────────────────────
    pixels   = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.3)
    _, labels, centers = cv2.kmeans(pixels, 4, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    centers  = np.uint8(centers)
    km_img   = centers[labels.flatten()].reshape(img.shape)

    # ── Annotation overlay ────────────────────────────────────────────
    annotated = img.copy()
    annotated[healthy_mask > 0] = (
        annotated[healthy_mask > 0] * 0.5 + np.array([0, 200, 50]) * 0.5
    ).astype(np.uint8)
    annotated[disease_mask > 0] = (
        annotated[disease_mask > 0] * 0.5 + np.array([0, 60, 220]) * 0.5
    ).astype(np.uint8)

    # ── Coverage stats ────────────────────────────────────────────────
    total     = h * w
    veg_px    = int(np.count_nonzero(veg_mask))
    healthy_px = int(np.count_nonzero(healthy_mask))
    disease_px = int(np.count_nonzero(disease_mask))
    soil_px   = total - veg_px

    disease_ratio = round(disease_px / veg_px * 100, 2) if veg_px > 0 else 0
    health_score  = max(0, round(100 - disease_ratio, 1))

    return {
        "original":      encode_image(img),
        "exg_map":       encode_image(exg_color),
        "veg_mask":      encode_image(veg_mask),
        "disease_mask":  encode_image(disease_mask),
        "healthy_mask":  encode_image(healthy_mask),
        "kmeans_map":    encode_image(km_img),
        "annotated":     encode_image(annotated),
        "stats": {
            "total_pixels":    total,
            "vegetation_px":   veg_px,
            "healthy_px":      healthy_px,
            "diseased_px":     disease_px,
            "soil_px":         soil_px,
            "veg_coverage_pct":    round(veg_px/total*100, 2),
            "healthy_pct":         round(healthy_px/total*100, 2),
            "disease_pct":         round(disease_px/total*100, 2),
            "disease_ratio_in_veg": disease_ratio,
            "crop_health_score":    health_score,
        },
        "domain": "agriculture",
        "summary": (
            f"Vegetation covers {round(veg_px/total*100,1)}% of frame. "
            f"Disease ratio: {disease_ratio}%. "
            f"Crop Health Score: {health_score}/100."
        ),
    }


def generate_sample() -> np.ndarray:
    """Synthetic crop field with healthy, diseased, and soil regions."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Soil background
    img[:] = [60, 100, 130]

    # Healthy crop rows (green)
    for row in range(6):
        y1 = 20 + row * 75
        y2 = y1 + 55
        img[y1:y2, :] = [30, 160, 50]
        noise = np.random.randint(-20, 20, (y2-y1, 640, 3), dtype=np.int16)
        img[y1:y2] = np.clip(img[y1:y2].astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Disease patches (yellowish/brown within crop rows)
    disease_zones = [
        (80, 60, 120, 50), (300, 135, 180, 45),
        (150, 210, 100, 40), (450, 290, 140, 50),
        (200, 365, 90, 45), (500, 60, 100, 55),
    ]
    for x, y, bw, bh in disease_zones:
        cv2.ellipse(img, (x+bw//2, y+bh//2), (bw//2, bh//2), 0, 0, 360,
                    (30, 140, 180), -1)   # yellowish in BGR

    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img
