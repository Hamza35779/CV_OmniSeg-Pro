"""
Flask API server — Image Segmentation Real-World Problem Solver

Endpoints:
    GET  /                          → Dashboard
    GET  /domain/sample/<name>      → Built-in domain sample image
    POST /domain/<name>             → Domain-specific analysis
    POST /segment/threshold         → Thresholding (all 3 methods)
    POST /segment/contours          → Contour detection + Canny
    POST /segment/watershed         → Watershed + GrabCut
    POST /segment/kmeans            → K-Means color segmentation
    POST /segment/superpixels       → SLIC superpixel segmentation
    POST /segment/all               → Run all classical methods
    GET  /sample/<name>             → Built-in classical sample
"""

import os
import sys
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from segmentation.utils import load_image, encode_image
from segmentation.thresholding import ThresholdSegmenter
from segmentation.contours import ContourSegmenter
from segmentation.watershed import WatershedSegmenter
from segmentation.kmeans import KMeansSegmenter
from segmentation.superpixels import SuperpixelSegmenter
from segmentation.domains import DOMAINS

app = Flask(__name__, static_folder="web", static_url_path="")
CORS(app)

# ────────────────────────────────────────────────────────────────────
# Classical sample images
# ────────────────────────────────────────────────────────────────────

def _gen_coins():
    img = np.full((400, 600, 3), 180, dtype=np.uint8)
    np.random.seed(7)
    for _ in range(14):
        cx = np.random.randint(60, 540)
        cy = np.random.randint(60, 340)
        r  = np.random.randint(25, 55)
        s  = np.random.randint(30, 90)
        cv2.circle(img, (cx, cy), r, (s, s+10, s+20), -1)
        cv2.circle(img, (cx, cy), r, (s-20, s-10, s), 2)
    return cv2.GaussianBlur(img, (3,3), 0)

def _gen_shapes():
    img = np.full((480, 640, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (60, 50),  (260, 210), (60, 80, 220), -1)
    cv2.ellipse(img,  (380, 120), (80, 80), 0, 0, 360, (60, 200, 80),  -1)
    cv2.rectangle(img, (160, 270), (280, 390), (220, 80, 80),  -1)
    cv2.fillPoly(img, [np.array([(450,80),(550,80),(500,200)])], (200,120,220))
    cv2.ellipse(img, (100, 380), (60, 60), 0, 0, 360, (220,200, 60), -1)
    cv2.rectangle(img, (330, 300), (510, 400), (60, 200, 180), -1)
    return img

def _gen_scene():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:180, :] = [200, 160, 100]
    img[360:, :] = [80, 90, 95]
    img[150:360, 50:200]  = [120, 130, 140]
    img[100:150, 80:170]  = [110, 120, 130]
    img[180:360, 240:420] = [140, 145, 150]
    img[130:180, 270:390] = [130, 135, 140]
    img[200:360, 480:600] = [100, 110, 120]
    for bx,by,bw,bh in [(60,160,20,20),(60,210,20,20),(120,160,20,20),(120,210,20,20),
                         (260,190,25,25),(350,190,25,25),(260,240,25,25),(350,240,25,25)]:
        img[by:by+bh, bx:bx+bw] = [220, 210, 180]
    for xi in range(50, 590, 80):
        img[395:415, xi:xi+40] = [220, 220, 220]
    return cv2.GaussianBlur(img, (3,3), 0)

CLASSICAL_SAMPLES = {
    "coins":  _gen_coins,
    "shapes": _gen_shapes,
    "scene":  _gen_scene,
}

# ────────────────────────────────────────────────────────────────────
# Routes — Static
# ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("web", "index.html")

@app.route("/sample/<name>")
def classical_sample(name):
    if name not in CLASSICAL_SAMPLES:
        return jsonify({"error": f"Unknown sample '{name}'"}), 404
    img = CLASSICAL_SAMPLES[name]()
    return jsonify({"image": encode_image(img), "name": name})

# ────────────────────────────────────────────────────────────────────
# Routes — Domain analysis
# ────────────────────────────────────────────────────────────────────

@app.route("/domain/sample/<name>")
def domain_sample(name):
    if name not in DOMAINS:
        return jsonify({"error": f"Unknown domain '{name}'"}), 404
    img = DOMAINS[name].generate_sample()
    return jsonify({"image": encode_image(img), "domain": name})

@app.route("/domain/<name>", methods=["POST"])
def run_domain(name):
    if name not in DOMAINS:
        return jsonify({"error": f"Unknown domain '{name}'"}), 404
    data = request.get_json(force=True)
    try:
        img    = _load_img(data)
        kwargs = {k: v for k, v in data.items() if k not in ("image", "sample")}
        result = DOMAINS[name].analyze(img, **kwargs)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ────────────────────────────────────────────────────────────────────
# Routes — Classical segmentation
# ────────────────────────────────────────────────────────────────────

@app.route("/segment/threshold", methods=["POST"])
def seg_threshold():
    data = request.get_json(force=True)
    try:
        img = _load_img(data)
        return jsonify(ThresholdSegmenter(img).run_all(int(data.get("thresh_val", 127))))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/segment/contours", methods=["POST"])
def seg_contours():
    data = request.get_json(force=True)
    try:
        img = _load_img(data)
        seg = ContourSegmenter(img)
        return jsonify({
            "detection": seg.detect(int(data.get("min_area", 200))),
            "canny":     seg.canny_contours(int(data.get("canny_low", 50)),
                                             int(data.get("canny_high", 150))),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/segment/watershed", methods=["POST"])
def seg_watershed():
    data = request.get_json(force=True)
    try:
        img = _load_img(data)
        seg = WatershedSegmenter(img)
        return jsonify({
            "watershed": seg.segment(float(data.get("dist_threshold", 0.5))),
            "grabcut":   seg.grabcut(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/segment/kmeans", methods=["POST"])
def seg_kmeans():
    data = request.get_json(force=True)
    try:
        img = _load_img(data)
        return jsonify(KMeansSegmenter(img).segment(int(data.get("k", 4))))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/segment/superpixels", methods=["POST"])
def seg_superpixels():
    data = request.get_json(force=True)
    try:
        img = _load_img(data)
        return jsonify(SuperpixelSegmenter(img).segment(
            int(data.get("n_segments", 100)),
            float(data.get("compactness", 10.0)),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/segment/all", methods=["POST"])
def seg_all():
    data = request.get_json(force=True)
    try:
        img = _load_img(data)
        cs  = ContourSegmenter(img)
        ws  = WatershedSegmenter(img)
        return jsonify({
            "threshold":   ThresholdSegmenter(img).run_all(),
            "contours":    cs.detect(),
            "canny":       cs.canny_contours(),
            "watershed":   ws.segment(),
            "grabcut":     ws.grabcut(),
            "kmeans":      KMeansSegmenter(img).segment(4),
            "superpixels": SuperpixelSegmenter(img).segment(80),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────

def _load_img(data: dict) -> np.ndarray:
    sample = data.get("sample")
    domain = data.get("domain_sample")
    if domain and domain in DOMAINS:
        return DOMAINS[domain].generate_sample()
    if sample and sample in CLASSICAL_SAMPLES:
        return CLASSICAL_SAMPLES[sample]()
    img_uri = data.get("image")
    if not img_uri:
        raise ValueError("Provide 'image' (data URI) or 'sample' name.")
    return load_image(img_uri)


if __name__ == "__main__":
    print("Image Segmentation server: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
