"""
Contour-based image segmentation and object extraction.

Workflow:
  1. Convert to grayscale → binary mask (Otsu or manual threshold)
  2. Find contours using cv2.findContours
  3. Filter by area to remove noise
  4. Draw bounding boxes, extract each object as a crop
  5. Compute shape descriptors per contour
"""

import cv2
import numpy as np
from .utils import to_gray, encode_image, resize_for_display


class ContourSegmenter:
    """Detect, analyse, and extract objects using contour detection."""

    def __init__(self, img: np.ndarray):
        self.original = resize_for_display(img)
        self.gray     = to_gray(self.original)

    # ------------------------------------------------------------------
    # Binary mask
    # ------------------------------------------------------------------

    def _binary_mask(self, blur_ksize: int = 5) -> np.ndarray:
        blurred = cv2.GaussianBlur(self.gray, (blur_ksize, blur_ksize), 0)
        _, mask  = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # Morphological closing fills small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------

    def detect(self, min_area: int = 200, max_objects: int = 20) -> dict:
        """Detect contours and return annotated image + per-object data.
        
        Args:
            min_area:    Minimum contour area in pixels (noise filter).
            max_objects: Maximum number of objects returned.
        """
        mask = self._binary_mask()
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and sort by area (largest first)
        contours = [
            c for c in contours if cv2.contourArea(c) >= min_area
        ]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = contours[:max_objects]

        # --- Annotated overview ---
        annotated = self.original.copy()
        objects   = []

        palette = _color_palette(len(contours))

        for i, cnt in enumerate(contours):
            color = palette[i]
            area  = cv2.contourArea(cnt)
            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

            # Contour outline
            cv2.drawContours(annotated, [cnt], -1, color, 2)

            # Label
            M  = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] else x + w // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] else y + h // 2
            cv2.putText(
                annotated, str(i + 1), (cx - 8, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            # Crop the object from the original
            crop = self.original[y:y+h, x:x+w]

            # Shape classification
            shape = _classify_shape(len(approx))
            circularity = (4 * np.pi * area / (peri * peri)) if peri > 0 else 0

            objects.append({
                "id":          i + 1,
                "area":        int(area),
                "perimeter":   round(float(peri), 1),
                "bounding_box": {"x": x, "y": y, "w": int(w), "h": int(h)},
                "centroid":    {"x": cx, "y": cy},
                "shape":       shape,
                "vertices":    len(approx),
                "circularity": round(float(circularity), 3),
                "crop":        encode_image(crop) if crop.size > 0 else None,
            })

        return {
            "original":   encode_image(self.original),
            "mask":       encode_image(mask),
            "annotated":  encode_image(annotated),
            "objects":    objects,
            "count":      len(objects),
            "method":     "Contour Detection",
        }

    # ------------------------------------------------------------------
    # Edge-based variant (Canny)
    # ------------------------------------------------------------------

    def canny_contours(
        self, low: int = 50, high: int = 150
    ) -> dict:
        """Use Canny edge detection to find contours."""
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, low, high)

        # Dilate edges slightly to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges  = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [c for c in contours if cv2.contourArea(c) >= 200]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        overlay = self.original.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 128), 2)

        return {
            "original":  encode_image(self.original),
            "edges":     encode_image(edges),
            "overlay":   encode_image(overlay),
            "count":     len(contours),
            "method":    "Canny Edge Contours",
            "thresholds": {"low": low, "high": high},
        }


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _classify_shape(vertices: int) -> str:
    shapes = {
        3: "Triangle",
        4: "Quadrilateral",
        5: "Pentagon",
        6: "Hexagon",
    }
    if vertices <= 2:
        return "Line"
    if vertices > 10:
        return "Circle / Ellipse"
    return shapes.get(vertices, f"{vertices}-gon")


def _color_palette(n: int) -> list[tuple[int, int, int]]:
    """Generate n visually distinct BGR colours using HSV spacing."""
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        hsv = np.uint8([[[hue, 220, 220]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors
