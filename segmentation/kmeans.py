"""
K-Means color-based image segmentation.

Groups pixels by color similarity into k clusters. Each cluster
becomes a segment. Excellent for scenes with distinct color regions
such as vegetation maps, medical tissue types, or land-use classes.
"""

import cv2
import numpy as np
from .utils import encode_image, resize_for_display


class KMeansSegmenter:
    """Segment an image by clustering pixel colors with K-Means."""

    def __init__(self, img: np.ndarray):
        self.original = resize_for_display(img)

    def segment(self, k: int = 4, attempts: int = 10) -> dict:
        """Run K-Means clustering and return per-cluster results.

        Args:
            k:        Number of color clusters (2–10).
            attempts: Number of times algorithm runs with different seeds.
        """
        k = max(2, min(k, 10))
        h, w = self.original.shape[:2]

        # Reshape to a list of pixels (each: [B, G, R])
        pixels = self.original.reshape(-1, 3).astype(np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100, 0.2
        )
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
        )

        centers   = np.uint8(centers)
        labels_2d = labels.reshape(h, w)

        # Build per-cluster coloured segmentation image
        segmented = centers[labels.flatten()].reshape(self.original.shape)

        clusters = []
        palette  = []
        overlay  = self.original.copy()

        for i in range(k):
            mask      = (labels_2d == i).astype(np.uint8) * 255
            area_px   = int(np.count_nonzero(mask))
            pct       = round(area_px / (h * w) * 100, 2)
            color_bgr = centers[i].tolist()
            color_hex = "#{:02x}{:02x}{:02x}".format(
                color_bgr[2], color_bgr[1], color_bgr[0]
            )

            # Draw cluster boundary
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1,
                             [int(c) for c in color_bgr], 2)

            # Cropped region (fill outside mask with black)
            region = cv2.bitwise_and(
                self.original, self.original, mask=mask
            )

            clusters.append({
                "id":       i + 1,
                "color_bgr": color_bgr,
                "color_hex": color_hex,
                "area_px":  area_px,
                "pct":      pct,
                "mask":     encode_image(mask),
                "region":   encode_image(region),
            })
            palette.append(color_hex)

        # Sort clusters by area
        clusters.sort(key=lambda c: c["area_px"], reverse=True)

        return {
            "original":  encode_image(self.original),
            "segmented": encode_image(segmented),
            "overlay":   encode_image(overlay),
            "k":         k,
            "clusters":  clusters,
            "palette":   palette,
            "method":    "K-Means Color Segmentation",
        }

    def dominant_colors(self, k: int = 5) -> list[dict]:
        """Return the k most dominant colors in the image."""
        result = self.segment(k=k, attempts=5)
        return sorted(result["clusters"], key=lambda c: c["pct"], reverse=True)
