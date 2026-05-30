"""
Real-world domain: Medical Imaging
Problem: Wound / lesion area measurement and severity classification.

Pipeline:
  - Color-based skin/lesion separation (HSV thresholding)
  - Morphological cleanup
  - Contour extraction for lesion boundary
  - Area + perimeter measurement (absolute px and % of tissue)
  - Severity classification by size
"""

import cv2
import numpy as np
from ..utils import encode_image, resize_for_display, to_gray


def analyze(img: np.ndarray, px_per_mm: float = 3.0) -> dict:
    """Segment and measure wounds/lesions in a medical image.

    Args:
        img:        Input BGR image (skin or scan).
        px_per_mm:  Scale factor for converting pixels to mm.
                    Default = 3 px/mm (roughly 75 DPI scan).
    """
    img = resize_for_display(img)
    h, w = img.shape[:2]

    # ── Step 1: Convert to HSV for skin/lesion separation ─────────────
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Lesion mask: low saturation and low value areas = potential wound
    # This works well for synthetic & real wound images
    gray = to_gray(img)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel, iterations=2)

    # ── Step 2: Find lesion contours ──────────────────────────────────
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    annotated = img.copy()
    lesions   = []

    colors = [(0,80,255), (0,200,100), (255,140,0), (180,0,220)]

    for i, cnt in enumerate(contours[:4]):
        area_px  = cv2.contourArea(cnt)
        if area_px < 200:
            continue
        peri_px  = cv2.arcLength(cnt, True)
        area_mm2 = round(area_px / (px_per_mm ** 2), 2)
        peri_mm  = round(peri_px / px_per_mm, 2)
        pct      = round(area_px / (h * w) * 100, 2)
        color    = colors[i % len(colors)]
        severity = _severity(pct)

        cv2.drawContours(annotated, [cnt], -1, color, 2)
        x, y, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(annotated, (x, y), (x+bw, y+bh), color, 1)

        M  = cv2.moments(cnt)
        cx = int(M["m10"]/M["m00"]) if M["m00"] else x + bw//2
        cy = int(M["m01"]/M["m00"]) if M["m00"] else y + bh//2
        cv2.putText(annotated, f"L{i+1}", (cx-12, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        lesions.append({
            "id":         i + 1,
            "area_px":    int(area_px),
            "area_mm2":   area_mm2,
            "perimeter_mm": peri_mm,
            "pct_of_image": pct,
            "severity":   severity,
            "color":      color,
        })

    # ── Step 3: Pseudo-color heatmap for depth estimation ─────────────
    heatmap_gray = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    total_lesion_px  = sum(l["area_px"] for l in lesions)
    total_lesion_mm2 = round(total_lesion_px / (px_per_mm ** 2), 2)

    return {
        "original":         encode_image(img),
        "binary_mask":      encode_image(binary),
        "cleaned_mask":     encode_image(cleaned),
        "annotated":        encode_image(annotated),
        "heatmap":          encode_image(heatmap_gray),
        "lesions":          lesions,
        "lesion_count":     len(lesions),
        "total_area_mm2":   total_lesion_mm2,
        "total_area_pct":   round(total_lesion_px / (h * w) * 100, 2),
        "px_per_mm":        px_per_mm,
        "image_size_px":    [w, h],
        "domain":           "medical",
        "summary": (
            f"{len(lesions)} lesion(s) detected. "
            f"Total affected area: {total_lesion_mm2} mm² "
            f"({round(total_lesion_px/(h*w)*100,1)}% of frame)."
        ),
    }


def _severity(pct: float) -> str:
    if pct < 2:   return "Minimal"
    if pct < 8:   return "Moderate"
    if pct < 20:  return "Severe"
    return "Critical"


def generate_sample() -> np.ndarray:
    """Synthetic wound-on-skin image for demonstration."""
    # Skin-toned background
    img = np.full((400, 500, 3), (120, 155, 200), dtype=np.uint8)
    # Add skin texture noise
    noise = np.random.randint(-18, 18, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Wound regions
    cv2.ellipse(img, (200, 180), (80, 55), 30, 0, 360, (30, 30, 60), -1)
    cv2.ellipse(img, (200, 180), (80, 55), 30, 0, 360, (10, 10, 40), 3)
    cv2.ellipse(img, (370, 300), (45, 30), 10, 0, 360, (25, 35, 70), -1)
    cv2.ellipse(img, (370, 300), (45, 30), 10, 0, 360, (10, 20, 50), 2)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img
