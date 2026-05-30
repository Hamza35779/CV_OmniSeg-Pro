import sys
sys.path.insert(0, '.')
from segmentation.domains import DOMAINS

for name, mod in DOMAINS.items():
    img = mod.generate_sample()
    result = mod.analyze(img)
    summary = result.get("summary","")[:80]
    print(f"[{name}] OK - {summary}")

from segmentation.kmeans import KMeansSegmenter
from segmentation.superpixels import SuperpixelSegmenter
import numpy as np, cv2

test_img = np.zeros((200, 300, 3), dtype=np.uint8)
cv2.circle(test_img, (100, 100), 60, (200, 80, 80), -1)
cv2.circle(test_img, (220, 100), 50, (80, 200, 80), -1)

km = KMeansSegmenter(test_img).segment(k=3)
print(f"[kmeans] OK - {km['k']} clusters, palette: {km['palette']}")

sp = SuperpixelSegmenter(test_img).segment(n_segments=30)
print(f"[slic] OK - {sp['n_superpixels']} superpixels")

print()
print("ALL DOMAIN TESTS PASSED")
