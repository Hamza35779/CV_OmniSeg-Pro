"""
SLIC Superpixel segmentation.

SLIC (Simple Linear Iterative Clustering) groups nearby pixels
with similar colors into compact, roughly uniform superpixels.
Each superpixel is a small coherent region — far better than
pixel-level processing for downstream tasks like classification.

Useful for: satellite imagery, medical scans, scene parsing.
"""

import cv2
import numpy as np
from .utils import encode_image, resize_for_display

try:
    from skimage.segmentation import slic, mark_boundaries
    from skimage.util import img_as_float
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


class SuperpixelSegmenter:
    """SLIC superpixel segmentation via scikit-image."""

    def __init__(self, img: np.ndarray):
        self.original = resize_for_display(img)

    def segment(
        self,
        n_segments: int = 100,
        compactness: float = 10.0,
        sigma: float = 1.0,
    ) -> dict:
        """Generate SLIC superpixels.

        Args:
            n_segments:   Approximate number of superpixels.
            compactness:  Balance between color and spatial proximity.
                          Higher = more square superpixels.
            sigma:        Width of Gaussian smoothing before segmentation.
        """
        if not SKIMAGE_AVAILABLE:
            return self._fallback_segment(n_segments)

        # scikit-image expects RGB, OpenCV gives BGR
        rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        img_float = img_as_float(rgb)

        segments = slic(
            img_float,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            start_label=0,
        )

        # Mark superpixel boundaries on the original
        boundaries_rgb = mark_boundaries(img_float, segments, color=(0, 1, 1))
        boundaries_bgr = cv2.cvtColor(
            (boundaries_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )

        # Mean-colour image: each superpixel filled with its average color
        mean_img = np.zeros_like(self.original)
        actual_segments = np.unique(segments)
        for seg_id in actual_segments:
            mask_bool = segments == seg_id
            mean_color = self.original[mask_bool].mean(axis=0).astype(np.uint8)
            mean_img[mask_bool] = mean_color

        # Per-superpixel stats (for the top 20 largest by area)
        seg_stats = []
        for seg_id in actual_segments:
            mask_bool = segments == seg_id
            area = int(mask_bool.sum())
            mean_color = self.original[mask_bool].mean(axis=0).astype(int).tolist()
            ys, xs = np.where(mask_bool)
            seg_stats.append({
                "id":    int(seg_id),
                "area":  area,
                "color": mean_color,
                "cx":    int(xs.mean()),
                "cy":    int(ys.mean()),
            })

        seg_stats.sort(key=lambda s: s["area"], reverse=True)

        return {
            "original":    encode_image(self.original),
            "boundaries":  encode_image(boundaries_bgr),
            "mean_color":  encode_image(mean_img),
            "n_superpixels": int(len(actual_segments)),
            "top_segments":  seg_stats[:20],
            "compactness": compactness,
            "method":      "SLIC Superpixels",
        }

    def _fallback_segment(self, n_segments: int) -> dict:
        """Fallback using OpenCV's SLIC-like algorithm if scikit-image missing."""
        slic_cv = cv2.ximgproc.createSuperpixelSLIC(
            self.original, cv2.ximgproc.SLICO, region_size=20
        )
        slic_cv.iterate(10)
        labels   = slic_cv.getLabels()
        mask_slic = slic_cv.getLabelContourMask()

        boundaries = self.original.copy()
        boundaries[mask_slic == 255] = [0, 255, 200]

        return {
            "original":      encode_image(self.original),
            "boundaries":    encode_image(boundaries),
            "mean_color":    encode_image(boundaries),
            "n_superpixels": int(labels.max() + 1),
            "top_segments":  [],
            "method":        "SLIC Superpixels (OpenCV fallback)",
        }
