"""
Thresholding-based image segmentation.

Covers three classical techniques:
  1. Global thresholding  — a single fixed threshold splits foreground/background
  2. Adaptive thresholding — threshold varies locally across the image
  3. Otsu's method        — automatically picks the optimal global threshold
"""

import cv2
import numpy as np
from .utils import to_gray, encode_image, resize_for_display


class ThresholdSegmenter:
    """Apply thresholding segmentation to an image."""

    def __init__(self, img: np.ndarray):
        self.original = resize_for_display(img)
        self.gray     = to_gray(self.original)

    # ------------------------------------------------------------------
    # 1. Global thresholding
    # ------------------------------------------------------------------

    def global_threshold(self, thresh_val: int = 127) -> dict:
        """Binary threshold: pixels above thresh_val → white, else black.
        
        Args:
            thresh_val: Threshold pixel value (0–255).
        """
        _, binary = cv2.threshold(
            self.gray, thresh_val, 255, cv2.THRESH_BINARY
        )
        _, binary_inv = cv2.threshold(
            self.gray, thresh_val, 255, cv2.THRESH_BINARY_INV
        )

        # Apply mask back on the original image to show segmented region
        masked = cv2.bitwise_and(
            self.original, self.original, mask=binary
        )

        return {
            "binary":     encode_image(binary),
            "binary_inv": encode_image(binary_inv),
            "masked":     encode_image(masked),
            "threshold":  thresh_val,
            "method":     "Global Thresholding",
            "stats": _region_stats(binary, self.gray),
        }

    # ------------------------------------------------------------------
    # 2. Adaptive thresholding
    # ------------------------------------------------------------------

    def adaptive_threshold(
        self,
        block_size: int = 11,
        C: int = 2,
        method: str = "gaussian",
    ) -> dict:
        """Locally adaptive threshold — great for uneven lighting.
        
        Args:
            block_size: Neighbourhood size for computing local threshold.
            C:          Constant subtracted from computed mean/weighted mean.
            method:     'gaussian' or 'mean'.
        """
        adapt_method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if method == "gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        # block_size must be odd and >= 3
        block_size = max(3, block_size | 1)

        binary = cv2.adaptiveThreshold(
            self.gray, 255, adapt_method,
            cv2.THRESH_BINARY, block_size, C
        )
        binary_inv = cv2.bitwise_not(binary)
        masked = cv2.bitwise_and(
            self.original, self.original, mask=binary_inv
        )

        return {
            "binary":     encode_image(binary),
            "binary_inv": encode_image(binary_inv),
            "masked":     encode_image(masked),
            "block_size": block_size,
            "C":          C,
            "adapt_method": method,
            "method":     "Adaptive Thresholding",
            "stats": _region_stats(binary, self.gray),
        }

    # ------------------------------------------------------------------
    # 3. Otsu's thresholding
    # ------------------------------------------------------------------

    def otsu_threshold(self) -> dict:
        """Automatically determine the optimal threshold using Otsu's method.
        
        Otsu minimises intra-class variance, perfect for bimodal histograms.
        """
        # Gaussian blur reduces noise before Otsu
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        otsu_val, binary = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binary_inv = cv2.bitwise_not(binary)
        masked = cv2.bitwise_and(
            self.original, self.original, mask=binary
        )

        # Histogram data for the UI chart
        hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
        histogram = [float(v[0]) for v in hist]

        return {
            "binary":     encode_image(binary),
            "binary_inv": encode_image(binary_inv),
            "masked":     encode_image(masked),
            "otsu_threshold": float(otsu_val),
            "method":     "Otsu's Thresholding",
            "histogram":  histogram,
            "stats": _region_stats(binary, self.gray),
        }

    # ------------------------------------------------------------------
    # Run all methods in one call
    # ------------------------------------------------------------------

    def run_all(self, thresh_val: int = 127) -> dict:
        return {
            "original":  encode_image(self.original),
            "global":    self.global_threshold(thresh_val),
            "adaptive":  self.adaptive_threshold(),
            "otsu":      self.otsu_threshold(),
        }


# ──────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────

def _region_stats(binary: np.ndarray, gray: np.ndarray) -> dict:
    """Compute basic statistics about the segmented foreground region."""
    total_px   = binary.size
    fore_px    = int(np.count_nonzero(binary))
    back_px    = total_px - fore_px
    mean_fore  = float(gray[binary > 0].mean()) if fore_px > 0 else 0.0
    return {
        "total_pixels":      total_px,
        "foreground_pixels": fore_px,
        "background_pixels": back_px,
        "foreground_pct":    round(fore_px / total_px * 100, 2),
        "mean_intensity_fg": round(mean_fore, 2),
    }
