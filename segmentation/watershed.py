"""
Watershed segmentation for scene understanding.

The watershed algorithm treats a grayscale image like a topographic map
and floods it from marked 'seeds', stopping where different flood regions
would meet — those boundaries become the segment borders.

Pipeline:
  1. Threshold + morphology to get sure foreground/background
  2. Distance transform to find confident foreground seeds
  3. Unknown region = dilation - sure foreground
  4. Label markers and run cv2.watershed
  5. Colour each region and return results
"""

import cv2
import numpy as np
from scipy import ndimage
from .utils import to_gray, encode_image, resize_for_display


class WatershedSegmenter:
    """Marker-based watershed segmentation."""

    def __init__(self, img: np.ndarray):
        self.original = resize_for_display(img)
        self.gray     = to_gray(self.original)

    # ------------------------------------------------------------------
    # Main watershed
    # ------------------------------------------------------------------

    def segment(
        self,
        dist_threshold: float = 0.5,
        blur_ksize: int = 7,
    ) -> dict:
        """Run the full watershed pipeline.
        
        Args:
            dist_threshold: Fraction of max distance transform value used
                            as the sure-foreground seed threshold (0.3–0.8).
            blur_ksize:     Gaussian blur kernel size before thresholding.
        """
        # ── Step 1: Preprocessing ──────────────────────────────────────
        blurred = cv2.GaussianBlur(
            self.gray, (blur_ksize, blur_ksize), 0
        )
        _, binary = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # ── Step 2: Noise removal with morphology ─────────────────────
        kernel     = np.ones((3, 3), np.uint8)
        sure_bg    = cv2.dilate(binary, kernel, iterations=3)
        opened     = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # ── Step 3: Distance transform → sure foreground ─────────────
        dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
        max_dist       = dist_transform.max()
        _, sure_fg     = cv2.threshold(
            dist_transform, dist_threshold * max_dist, 255, 0
        )
        sure_fg = np.uint8(sure_fg)

        # ── Step 4: Unknown region ────────────────────────────────────
        unknown = cv2.subtract(sure_bg, sure_fg)

        # ── Step 5: Marker labelling ──────────────────────────────────
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1                    # background label = 1 (not 0)
        markers[unknown == 255] = 0     # unknown region = 0

        # ── Step 6: Watershed ─────────────────────────────────────────
        markers_ws = cv2.watershed(self.original.copy(), markers.copy())

        # ── Step 7: Build coloured result ─────────────────────────────
        result, region_data, label_count = self._colorize(markers_ws)

        # Overlay watershed boundaries on original
        boundary_overlay = self.original.copy()
        boundary_overlay[markers_ws == -1] = [0, 0, 255]

        return {
            "original":         encode_image(self.original),
            "binary":           encode_image(binary),
            "sure_bg":          encode_image(sure_bg),
            "sure_fg":          encode_image(sure_fg),
            "dist_transform":   encode_image(
                                    self._normalise_dist(dist_transform)
                                ),
            "unknown":          encode_image(unknown),
            "result":           encode_image(result),
            "boundary_overlay": encode_image(boundary_overlay),
            "region_count":     label_count,
            "regions":          region_data,
            "dist_threshold":   dist_threshold,
            "method":           "Watershed Segmentation",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _colorize(
        self, markers: np.ndarray
    ) -> tuple[np.ndarray, list[dict], int]:
        """Assign a random colour to each watershed region."""
        result    = np.zeros_like(self.original)
        labels    = np.unique(markers)
        labels    = labels[(labels > 1)]   # skip background (1) and boundary (-1)

        np.random.seed(42)
        region_data = []
        for label in labels:
            mask  = (markers == label).astype(np.uint8)
            color = np.random.randint(60, 230, 3).tolist()
            result[markers == label] = color

            area   = int(mask.sum())
            ys, xs = np.where(mask)
            cx = int(xs.mean()) if len(xs) else 0
            cy = int(ys.mean()) if len(ys) else 0

            region_data.append({
                "label":    int(label),
                "area":     area,
                "centroid": {"x": cx, "y": cy},
                "color":    color,
            })

        return result, region_data, len(labels)

    @staticmethod
    def _normalise_dist(dist: np.ndarray) -> np.ndarray:
        """Normalise a distance transform to 0–255 for display."""
        normed = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(normed)

    # ------------------------------------------------------------------
    # GrabCut variant
    # ------------------------------------------------------------------

    def grabcut(self, margin_frac: float = 0.1) -> dict:
        """Use GrabCut with a central rectangle as the initial foreground.
        
        GrabCut iteratively refines the foreground/background mask using
        Gaussian Mixture Models — more accurate than pure watershed on
        natural images.
        """
        h, w = self.original.shape[:2]
        m    = int(min(h, w) * margin_frac)
        rect = (m, m, w - 2 * m, h - 2 * m)

        mask_gc = np.zeros((h, w), np.uint8)
        bgd_mdl = np.zeros((1, 65), np.float64)
        fgd_mdl = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            self.original, mask_gc, rect,
            bgd_mdl, fgd_mdl, 5, cv2.GC_INIT_WITH_RECT
        )

        # Pixels marked as probable or definite foreground
        fg_mask = np.where(
            (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD),
            255, 0
        ).astype(np.uint8)

        result = cv2.bitwise_and(
            self.original, self.original, mask=fg_mask
        )

        return {
            "original": encode_image(self.original),
            "fg_mask":  encode_image(fg_mask),
            "result":   encode_image(result),
            "method":   "GrabCut Segmentation",
        }
