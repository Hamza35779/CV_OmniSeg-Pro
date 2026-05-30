"""
OmniSeg Pro — Local Testing Dataset Generator
Generates realistic, high-fidelity testing images for all 5 domains
and saves them directly into the samples/ directory.
"""

import os
import numpy as np
import cv2

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

def create_dirs():
    for domain in ["medical", "traffic", "agriculture", "industrial", "aerial"]:
        os.makedirs(os.path.join(SAMPLES_DIR, domain), exist_ok=True)

def generate_medical_dataset():
    print("Generating Medical dataset...")
    path = os.path.join(SAMPLES_DIR, "medical")
    
    # 1. Skin Lesion
    img1 = np.full((480, 640, 3), (195, 210, 245), dtype=np.uint8) # skin tone
    # Add main lesion
    cv2.ellipse(img1, (320, 240), (95, 75), 25, 0, 360, (40, 50, 70), -1)
    # Add irregular boundaries (lesion details)
    np.random.seed(42)
    for _ in range(35):
        rx = 320 + np.random.randint(-70, 70)
        ry = 240 + np.random.randint(-55, 55)
        rr = np.random.randint(15, 35)
        color = (np.random.randint(30, 60), np.random.randint(40, 70), np.random.randint(50, 90))
        cv2.circle(img1, (rx, ry), rr, color, -1)
    # Smooth & blur
    img1 = cv2.GaussianBlur(img1, (15, 15), 0)
    cv2.imwrite(os.path.join(path, "test_skin_lesion.jpg"), img1)

    # 2. Chest X-Ray
    img2 = np.zeros((512, 512, 3), dtype=np.uint8)
    # Ribcage silhouette
    cv2.ellipse(img2, (256, 256), (180, 240), 0, 0, 360, (20, 20, 20), -1)
    # Left & Right Lungs (dark regions)
    cv2.ellipse(img2, (170, 250), (60, 160), 10, 0, 360, (5, 5, 5), -1)
    cv2.ellipse(img2, (342, 250), (60, 160), -10, 0, 360, (5, 5, 5), -1)
    # Spine/sternum shadow
    cv2.rectangle(img2, (246, 50), (266, 460), (45, 45, 45), -1)
    # Ribs simulation (horizontal curved lines)
    for y in range(120, 420, 35):
        cv2.ellipse(img2, (120, y), (100, 30), 15, 0, 180, (55, 55, 55), 3)
        cv2.ellipse(img2, (392, y), (100, 30), -15, 0, 180, (55, 55, 55), 3)
    # Heart silhouette
    cv2.ellipse(img2, (280, 280), (55, 65), 35, 0, 360, (70, 70, 70), -1)
    img2 = cv2.GaussianBlur(img2, (9, 9), 0)
    cv2.imwrite(os.path.join(path, "test_chest_xray.jpg"), img2)

    # 3. Brain MRI
    img3 = np.zeros((480, 480, 3), dtype=np.uint8)
    # Skull
    cv2.circle(img3, (240, 240), 190, (30, 30, 30), 8)
    # Brain matter contour
    cv2.ellipse(img3, (240, 240), (160, 175), 0, 0, 360, (80, 80, 80), -1)
    # Sulci & Gyri (brain folds)
    np.random.seed(99)
    for _ in range(80):
        x = np.random.randint(100, 380)
        y = np.random.randint(100, 380)
        r = np.random.randint(10, 30)
        cv2.circle(img3, (x, y), r, (110, 110, 110), -1)
    # Ventricles (fluid cavities - dark)
    cv2.ellipse(img3, (215, 220), (15, 45), 15, 0, 360, (15, 15, 15), -1)
    cv2.ellipse(img3, (265, 220), (15, 45), -15, 0, 360, (15, 15, 15), -1)
    # Lesion simulation (bright white region)
    cv2.circle(img3, (160, 180), 22, (230, 230, 230), -1)
    img3 = cv2.GaussianBlur(img3, (11, 11), 0)
    cv2.imwrite(os.path.join(path, "test_brain_mri.jpg"), img3)

def generate_traffic_dataset():
    print("Generating Traffic dataset...")
    path = os.path.join(SAMPLES_DIR, "traffic")

    # 1. Highway Road
    img1 = np.full((480, 640, 3), (40, 40, 45), dtype=np.uint8) # dark asphalt
    # Sky
    cv2.rectangle(img1, (0, 0), (640, 160), (190, 150, 100), -1)
    # Drivable road boundaries (perspective lines)
    road_pts = np.array([[50, 480], [280, 160], [360, 160], [590, 480]], dtype=np.int32)
    cv2.fillPoly(img1, [road_pts], (75, 75, 78))
    # Lane divider lines (dashed)
    cv2.line(img1, (320, 160), (320, 480), (220, 220, 220), 4) # yellow center
    # Left road shoulder
    cv2.line(img1, (280, 160), (50, 480), (240, 240, 240), 3)
    # Right road shoulder
    cv2.line(img1, (360, 160), (590, 480), (240, 240, 240), 3)
    # Vehicle in front (Bbox, tail lights)
    cv2.rectangle(img1, (290, 230), (350, 275), (90, 90, 95), -1)
    cv2.circle(img1, (300, 260), 6, (0, 0, 240), -1) # left light
    cv2.circle(img1, (340, 260), 6, (0, 0, 240), -1) # right light
    # Farther vehicle
    cv2.rectangle(img1, (315, 175), (335, 195), (60, 65, 70), -1)
    cv2.imwrite(os.path.join(path, "test_highway.jpg"), img1)

    # 2. Urban Street
    img2 = np.full((480, 640, 3), (60, 60, 65), dtype=np.uint8)
    # Sidewalks
    cv2.rectangle(img2, (0, 0), (120, 480), (95, 100, 105), -1)
    cv2.rectangle(img2, (520, 0), (640, 480), (95, 100, 105), -1)
    # Road lanes
    cv2.line(img2, (250, 0), (250, 480), (220, 220, 220), 3)
    cv2.line(img2, (390, 0), (390, 480), (220, 220, 220), 3)
    # Vehicles parked
    cv2.rectangle(img2, (30, 80), (90, 170), (180, 60, 60), -1)
    cv2.rectangle(img2, (550, 220), (610, 310), (60, 120, 190), -1)
    # Moving vehicle
    cv2.rectangle(img2, (280, 180), (360, 290), (45, 45, 48), -1)
    cv2.imwrite(os.path.join(path, "test_urban_road.jpg"), img2)

def generate_agriculture_dataset():
    print("Generating Agriculture dataset...")
    path = os.path.join(SAMPLES_DIR, "agriculture")

    # 1. Canopy Field
    img1 = np.full((480, 640, 3), (60, 85, 135), dtype=np.uint8) # soil base
    # Add green crops in rows
    for offset in range(30, 640, 110):
        for y in range(0, 480, 20):
            rx = offset + np.random.randint(-15, 15)
            # ExG healthy vegetation
            g = np.random.randint(140, 220)
            r = np.random.randint(40, 80)
            b = np.random.randint(30, 70)
            cv2.circle(img1, (rx, y), np.random.randint(18, 32), (b, g, r), -1)
    cv2.imwrite(os.path.join(path, "test_wheat_canopy.jpg"), img1)

    # 2. Diseased Plant Leaves
    img2 = np.full((480, 640, 3), (35, 45, 55), dtype=np.uint8) # dark table
    # Big leaf silhouette
    leaf_pts = np.array([[320, 40], [450, 180], [480, 340], [320, 440], [160, 340], [190, 180]], dtype=np.int32)
    cv2.fillPoly(img2, [leaf_pts], (45, 165, 85)) # green leaf
    # Disease spots (yellow/brown lesions)
    np.random.seed(111)
    for _ in range(12):
        cx = np.random.randint(220, 420)
        cy = np.random.randint(120, 360)
        cr = np.random.randint(10, 25)
        cv2.circle(img2, (cx, cy), cr, (40, 130, 185), -1) # brownish spots
        cv2.circle(img2, (cx, cy), cr - 4, (30, 95, 145), -1)
    img2 = cv2.GaussianBlur(img2, (7, 7), 0)
    cv2.imwrite(os.path.join(path, "test_diseased_leaves.jpg"), img2)

def generate_industrial_dataset():
    print("Generating Industrial dataset...")
    path = os.path.join(SAMPLES_DIR, "industrial")

    # 1. Brushed Metal Sheet with defects
    img1 = np.full((480, 640, 3), 190, dtype=np.uint8)
    # Add metallic horizontal texture lines
    for y in range(0, 480, 4):
        cv2.line(img1, (0, y), (640, y), (180, 180, 180), 1)
    # Add scratch defect (dark long line)
    cv2.line(img1, (120, 180), (380, 290), (40, 40, 40), 2)
    # Add pit/dent defect (dark round spots)
    cv2.circle(img1, (450, 130), 7, (75, 75, 75), -1)
    cv2.circle(img1, (450, 130), 4, (30, 30, 30), -1)
    cv2.circle(img1, (240, 350), 9, (80, 80, 80), -1)
    cv2.circle(img1, (240, 350), 5, (25, 25, 25), -1)
    cv2.imwrite(os.path.join(path, "test_metal_surface.jpg"), img1)

    # 2. PCB Circuit Inspection
    img2 = np.full((480, 640, 3), (35, 110, 45), dtype=np.uint8) # green PCB
    # Copper traces
    for x in range(80, 600, 90):
        cv2.line(img2, (x, 0), (x, 480), (80, 175, 215), 5)
        cv2.line(img2, (x, 150), (x+50, 200), (80, 175, 215), 5)
        cv2.line(img2, (x+50, 200), (x+50, 480), (80, 175, 215), 5)
    # IC Chips (black square blocks)
    cv2.rectangle(img2, (120, 100), (220, 200), (20, 20, 20), -1)
    cv2.rectangle(img2, (380, 240), (510, 370), (20, 20, 20), -1)
    # Solder pads (silver circles)
    for y in range(40, 440, 60):
        cv2.circle(img2, (290, y), 8, (190, 190, 195), -1)
    # Scratch defect cutting a trace (anomaly)
    cv2.line(img2, (220, 130), (320, 150), (35, 110, 45), 4) # cuts the trace
    cv2.imwrite(os.path.join(path, "test_circuit_board.jpg"), img2)

def generate_aerial_dataset():
    print("Generating Aerial dataset...")
    path = os.path.join(SAMPLES_DIR, "aerial")

    # 1. City / Farmland Top-down
    img1 = np.full((512, 512, 3), (85, 110, 100), dtype=np.uint8) # soil base
    # Farmland grid patches
    # Green fields
    cv2.rectangle(img1, (20, 20), (230, 230), (50, 150, 65), -1)
    cv2.rectangle(img1, (260, 20), (490, 200), (70, 180, 90), -1)
    # Water canal (blue line)
    cv2.line(img1, (0, 250), (512, 250), (190, 110, 40), 22)
    # Urban housing block (gray shapes)
    cv2.rectangle(img1, (40, 290), (210, 470), (110, 115, 120), -1)
    for x in range(60, 190, 40):
        for y in range(310, 450, 40):
            cv2.rectangle(img1, (x, y), (x+20, y+20), (180, 185, 190), -1)
    cv2.imwrite(os.path.join(path, "test_satellite_grid.jpg"), img1)

def main():
    create_dirs()
    generate_medical_dataset()
    generate_traffic_dataset()
    generate_agriculture_dataset()
    generate_industrial_dataset()
    generate_aerial_dataset()
    print("\nDataset generated successfully in: samples/")

if __name__ == "__main__":
    main()
