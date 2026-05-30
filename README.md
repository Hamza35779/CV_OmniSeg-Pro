# Image Segmentation for Scene Understanding
**Course:** Computer Vision Lab | **Batch:** AI-23

A full Python + OpenCV project implementing classical image segmentation techniques, with an interactive web dashboard for visualization.

## Techniques Implemented

| Technique | Method | Description |
|---|---|---|
| **Thresholding** | Global, Adaptive, Otsu | Converts grayscale images to binary masks |
| **Contours** | findContours + Canny | Detects and extracts object boundaries |
| **Watershed** | Marker-based | Flood-fill segmentation from seeds |
| **GrabCut** | GMM-based | Iterative foreground/background separation |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python server.py

# Open dashboard → http://127.0.0.1:5000
```

## Project Structure

```
image-segmentation/
├── segmentation/
│   ├── thresholding.py   # Global, Adaptive, Otsu thresholding
│   ├── contours.py       # Contour detection & object extraction
│   ├── watershed.py      # Watershed + GrabCut segmentation
│   └── utils.py          # Image encode/decode utilities
├── web/
│   ├── index.html        # 4-tab interactive dashboard
│   ├── style.css         # Dark-mode styling
│   └── app.js            # API calls & image rendering
├── server.py             # Flask REST API (5 endpoints)
└── requirements.txt
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/segment/threshold` | POST | All 3 thresholding variants |
| `/segment/contours` | POST | Contour detection + Canny |
| `/segment/watershed` | POST | Watershed + GrabCut |
| `/segment/all` | POST | All methods at once |
| `/sample/<name>` | GET | Built-in sample image |

## Lab Objectives Covered

- ✅ Understand image segmentation concepts
- ✅ Differentiate thresholding, contours, and watershed
- ✅ Implement classical techniques using OpenCV
- ✅ Extract and visualize objects from images
