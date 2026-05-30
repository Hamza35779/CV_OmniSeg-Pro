"""
Real-world domain: Industrial Quality Control
Problem: Surface defect detection on manufactured parts.

Pipeline:
  - Adaptive thresholding to find anomalies on uniform surfaces
  - Morphological filtering to isolate defect regions
  - Defect classification by shape and area (scratch, dent, crack, spot)
  - Severity scoring (pass / marginal / fail)
"""

import cv2
import numpy as np
from ..utils import encode_image, resize_for_display, to_gray


def analyze(img: np.ndarray, sensitivity: int = 11) -> dict:
    """Detect and classify surface defects.

    Args:
        img:         Input BGR image of a manufactured surface.
        sensitivity: Adaptive threshold block size (odd, 5–31).
                     Lower = more sensitive to fine defects.
    """
    img = resize_for_display(img)
    h, w = img.shape[:2]
    gray = to_gray(img)

    # ── Step 1: Background normalisation ─────────────────────────────
    blur_bg = cv2.GaussianBlur(gray, (51, 51), 0)
    normalised = cv2.subtract(blur_bg, gray)
    normalised = cv2.equalizeHist(normalised)

    # ── Step 2: Adaptive thresholding for defect mask ─────────────────
    block = max(5, sensitivity | 1)
    defect_mask = cv2.adaptiveThreshold(
        normalised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, -5
    )

    # ── Step 3: Morphological cleaning ────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Step 4: Find defect contours ──────────────────────────────────
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 30]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    annotated = img.copy()
    defects   = []

    for i, cnt in enumerate(contours):
        area    = cv2.contourArea(cnt)
        peri    = cv2.arcLength(cnt, True)
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect  = bw / max(bh, 1)
        circularity = (4 * np.pi * area / (peri * peri)) if peri > 0 else 0
        dtype   = _classify_defect(area, aspect, circularity)
        sev     = _severity(area, h, w)
        color   = _defect_color(sev)

        cv2.drawContours(annotated, [cnt], -1, color, 2)
        cv2.putText(annotated, dtype[0], (x, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        defects.append({
            "id":          i+1,
            "type":        dtype,
            "area_px":     int(area),
            "aspect_ratio": round(float(aspect), 2),
            "circularity": round(float(circularity), 3),
            "severity":    sev,
            "bbox":        [int(x), int(y), int(bw), int(bh)],
        })

    # ── Step 5: Heatmap ───────────────────────────────────────────────
    heat = cv2.applyColorMap(normalised, cv2.COLORMAP_HOT)

    total_defect_px = int(np.count_nonzero(defect_mask))
    defect_pct      = round(total_defect_px / (h * w) * 100, 3)
    verdict         = "PASS" if defect_pct < 0.5 else ("MARGINAL" if defect_pct < 2 else "FAIL")

    return {
        "original":     encode_image(img),
        "normalised":   encode_image(normalised),
        "defect_mask":  encode_image(defect_mask),
        "annotated":    encode_image(annotated),
        "heatmap":      encode_image(heat),
        "defects":      defects,
        "defect_count": len(defects),
        "defect_area_px":  total_defect_px,
        "defect_pct":   defect_pct,
        "verdict":      verdict,
        "domain":       "industrial",
        "summary": (
            f"QC Result: {verdict}. "
            f"{len(defects)} defect(s) detected. "
            f"Defect area: {defect_pct}% of surface."
        ),
    }


def _classify_defect(area, aspect, circularity):
    if circularity > 0.7:
        return "Spot" if area < 300 else "Dent"
    if aspect > 4:
        return "Scratch"
    if aspect > 2:
        return "Crack"
    return "Pit"


def _severity(area, h, w):
    rel = area / (h * w)
    if rel < 0.0005: return "Minor"
    if rel < 0.003:  return "Moderate"
    return "Major"


def _defect_color(severity):
    return {"Minor": (0,220,120), "Moderate": (0,150,255), "Major": (0,0,255)}[severity]


def generate_sample() -> np.ndarray:
    """Synthetic metal surface with scratches, dents, and pits."""
    # Uniform grey surface with subtle texture
    img = np.full((400, 500, 3), 160, dtype=np.uint8)
    texture = np.random.randint(-12, 12, img.shape, np.int16)
    img = np.clip(img.astype(np.int16) + texture, 0, 255).astype(np.uint8)

    # Scratches (dark elongated)
    cv2.line(img, (80, 100), (300, 120),  (80, 82, 85), 2)
    cv2.line(img, (200, 250), (450, 240), (75, 77, 78), 3)
    cv2.line(img, (350, 80), (370, 180),  (70, 72, 74), 2)

    # Dents (dark ellipses)
    cv2.ellipse(img, (150, 300), (30, 20), 15, 0, 360, (90, 92, 94), -1)
    cv2.ellipse(img, (400, 180), (22, 15), 0,  0, 360, (85, 87, 88), -1)

    # Pits (small dark dots)
    for x, y in [(240,150),(320,300),(100,200),(430,350),(60,320)]:
        cv2.circle(img, (x,y), 6, (88, 90, 91), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img
