"""
Shared image utility helpers.

- encode_image: converts a NumPy BGR image to a base64 PNG data URI
  so it can be sent directly over the JSON API.
- load_image: decodes a base64 data URI back to a NumPy array.
"""

import base64
import numpy as np
import cv2


def encode_image(img: np.ndarray) -> str:
    """Encode a BGR NumPy image to a base64 PNG data URI string."""
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise ValueError("cv2.imencode failed")
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def load_image(data_uri: str) -> np.ndarray:
    """Decode a base64 data URI into a BGR NumPy array."""
    # Strip the header: "data:image/...;base64,"
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    raw = base64.b64decode(data_uri)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from data URI")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale if not already."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_for_display(img: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Downscale an image so its longest side is at most max_side px."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
