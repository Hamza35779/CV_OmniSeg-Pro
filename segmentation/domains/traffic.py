"""
Real-world domain: Traffic & Road Scene Analysis
Problem: Lane line detection + vehicle/pedestrian region extraction.

Pipeline:
  - ROI masking (lower half = road)
  - Canny edge detection
  - Hough line transform for lane lines
  - Watershed/contour segmentation for vehicle blobs
  - Per-vehicle bounding box + position classification
"""

import cv2
import numpy as np
from ..utils import encode_image, resize_for_display, to_gray


def analyze(img: np.ndarray) -> dict:
    """Detect lane lines and segment vehicles in a road scene."""
    img = resize_for_display(img)
    h, w = img.shape[:2]
    gray = to_gray(img)

    # ── Lane detection ────────────────────────────────────────────────
    lane_overlay, lane_mask, lanes = _detect_lanes(img, gray, h, w)

    # ── Vehicle/object segmentation ───────────────────────────────────
    vehicles_overlay, vehicles, vehicle_mask = _detect_vehicles(img, gray, h, w)

    # ── Combined result ───────────────────────────────────────────────
    combined = img.copy()
    combined[lane_mask > 0] = (
        combined[lane_mask > 0] * 0.6 + np.array([0, 200, 100]) * 0.4
    ).astype(np.uint8)
    for v in vehicles:
        x, y, bw, bh = v["bbox"]
        color = (0, 120, 255) if v["zone"] == "near" else (0, 200, 255)
        cv2.rectangle(combined, (x,y), (x+bw, y+bh), color, 2)
        label = f"{v['id']} [{v['zone']}]"
        cv2.putText(combined, label, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Draw lanes on combined
    for lane in lanes:
        x1, y1, x2, y2 = lane["coords"]
        cv2.line(combined, (x1,y1), (x2,y2), (0,255,80), 3)

    # ── Road mask (drivable area) ─────────────────────────────────────
    road_mask = _drivable_area(img, h, w)

    danger_count = sum(1 for v in vehicles if v["zone"] == "near")

    return {
        "original":       encode_image(img),
        "lane_overlay":   encode_image(lane_overlay),
        "vehicle_mask":   encode_image(vehicle_mask),
        "road_mask":      encode_image(road_mask),
        "combined":       encode_image(combined),
        "lanes":          lanes,
        "vehicles":       vehicles,
        "lane_count":     len(lanes),
        "vehicle_count":  len(vehicles),
        "danger_count":   danger_count,
        "domain":         "traffic",
        "summary": (
            f"{len(lanes)} lane lines detected. "
            f"{len(vehicles)} vehicle(s) segmented. "
            f"{danger_count} in near zone."
        ),
    }


def _detect_lanes(img, gray, h, w):
    # Focus on lower 55% of image (road region)
    roi_mask = np.zeros_like(gray)
    poly     = np.array([[(0, h), (0, int(h*0.45)),
                           (w, int(h*0.45)), (w, h)]], dtype=np.int32)
    cv2.fillPoly(roi_mask, poly, 255)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)
    roi_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

    lines = cv2.HoughLinesP(
        roi_edges, rho=1, theta=np.pi/180,
        threshold=40, minLineLength=60, maxLineGap=30
    )

    overlay   = img.copy()
    lane_mask = np.zeros(gray.shape, np.uint8)
    lane_list = []

    if lines is not None:
        for i, line in enumerate(lines[:12]):
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            if abs(angle) < 15:   # nearly horizontal → skip
                continue
            side  = "left" if (x1+x2)/2 < w/2 else "right"
            cv2.line(overlay, (x1,y1), (x2,y2), (0,255,100), 3)
            cv2.line(lane_mask, (x1,y1), (x2,y2), 255, 8)
            lane_list.append({
                "id":    i+1,
                "coords": [int(x1),int(y1),int(x2),int(y2)],
                "angle":  round(float(angle), 1),
                "side":   side,
            })

    return overlay, lane_mask, lane_list


def _detect_vehicles(img, gray, h, w):
    # Focus on upper-to-mid region where vehicles appear
    roi_top = int(h * 0.15)
    roi_bot = int(h * 0.75)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological ops to merge vehicle blobs
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if 800 < cv2.contourArea(c) < w*h*0.5]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    overlay      = img.copy()
    vehicle_mask = np.zeros(gray.shape, np.uint8)
    vehicles     = []

    for i, cnt in enumerate(contours):
        x, y, bw, bh = cv2.boundingRect(cnt)
        if y < roi_top or y + bh > roi_bot + bh//2:
            continue
        area = cv2.contourArea(cnt)
        # Near = bottom of frame, far = top
        zone = "near" if (y + bh) > h * 0.65 else "far"
        color = (0, 80, 255) if zone == "near" else (100, 200, 255)
        cv2.drawContours(vehicle_mask, [cnt], -1, 255, -1)
        cv2.rectangle(overlay, (x,y), (x+bw, y+bh), color, 2)
        vehicles.append({
            "id":   i+1,
            "bbox": [int(x),int(y),int(bw),int(bh)],
            "area": int(area),
            "zone": zone,
        })

    return overlay, vehicles, vehicle_mask


def _drivable_area(img, h, w):
    """Simple perspective-corrected road mask."""
    mask = np.zeros((h, w), np.uint8)
    poly = np.array([[(int(w*0.1), h), (int(w*0.35), int(h*0.55)),
                       (int(w*0.65), int(h*0.55)), (int(w*0.9), h)]], np.int32)
    cv2.fillPoly(mask, poly, 255)
    result = img.copy()
    result[mask == 0] = (result[mask == 0] * 0.4).astype(np.uint8)
    return result


def generate_sample() -> np.ndarray:
    """Synthetic road scene: sky, road, vehicles, lane markings."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Sky gradient
    for y in range(200):
        t = y / 200
        img[y, :] = [int(200*(1-t)+100*t), int(180*(1-t)+140*t), int(100*(1-t)+80*t)]
    # Road surface
    img[240:, :] = [70, 80, 85]
    # Perspective lines for road
    for x_off in [150, 490]:
        cv2.line(img, (320, 240), (x_off, 480), (90, 100, 105), 3)
    # Lane markings
    for y in range(280, 460, 50):
        alpha = (y - 280) / 180
        xc = 320
        hw = int(60 + 80 * alpha)
        cv2.line(img, (xc-hw, y), (xc-hw+30, y), (220, 220, 220), 3)
        cv2.line(img, (xc+hw-30, y), (xc+hw, y), (220, 220, 220), 3)
    # Vehicles
    cv2.rectangle(img, (260, 290), (380, 360), (80, 90, 180), -1)   # far car
    cv2.rectangle(img, (260, 290), (380, 360), (50, 60, 140), 2)
    cv2.rectangle(img, (180, 370), (340, 460), (60, 130, 180), -1)  # near car
    cv2.rectangle(img, (180, 370), (340, 460), (40, 100, 140), 2)
    cv2.rectangle(img, (420, 360), (560, 460), (100, 60, 60), -1)   # truck
    cv2.rectangle(img, (420, 360), (560, 460), (80, 40, 40), 2)
    # Windows
    cv2.rectangle(img, (280, 298), (360, 325), (180, 200, 220), -1)
    cv2.rectangle(img, (200, 378), (320, 410), (180, 200, 220), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img
