"""
Sample image downloader for the Image Segmentation project.

Downloads one real public-domain image per domain from Wikimedia Commons,
NASA, and other open sources into the samples/ directory.

Usage:
    python download_samples.py
"""

import os
import sys
import urllib.request
import urllib.error
import ssl

# Allow HTTPS on machines with strict SSL configs
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode    = ssl.CERT_NONE

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Image definitions — each entry is a list of URLs tried in order.
# The first URL that downloads successfully is saved.
# ─────────────────────────────────────────────────────────────────────

IMAGE_SETS = {

    # ── Medical ──────────────────────────────────────────────────────
    "medical": [
        {
            "filename": "medical_skin_lesion.jpg",
            "desc":     "Dermoscopy skin lesion (ISIC public sample)",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Melanoma.jpg/640px-Melanoma.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Basal_cell_carcinoma.jpg/640px-Basal_cell_carcinoma.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Skin_tags.JPG/640px-Skin_tags.JPG",
            ],
        },
        {
            "filename": "medical_xray_chest.jpg",
            "desc":     "Chest X-ray (public domain NIH-style)",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Chest_Xray_PA_3-8-2010.png/640px-Chest_Xray_PA_3-8-2010.png",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/CXR_typical_pneumonia.jpg/640px-CXR_typical_pneumonia.jpg",
            ],
        },
        {
            "filename": "medical_mri_brain.jpg",
            "desc":     "Brain MRI scan (public domain)",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/MRI_of_Human_Brain.jpg/640px-MRI_of_Human_Brain.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Brain_MRI_segmentation.png/640px-Brain_MRI_segmentation.png",
            ],
        },
    ],

    # ── Traffic ───────────────────────────────────────────────────────
    "traffic": [
        {
            "filename": "traffic_highway.jpg",
            "desc":     "Highway with multiple lanes and vehicles",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Zufahrt_Autobahn_A4.jpg/640px-Zufahrt_Autobahn_A4.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/US_101_-_Marin.jpg/640px-US_101_-_Marin.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/A4_Highway_Poland.jpg/640px-A4_Highway_Poland.jpg",
            ],
        },
        {
            "filename": "traffic_urban_road.jpg",
            "desc":     "Urban street with traffic",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Good_Food_Display_-_NCI_Visuals_Online.jpg/640px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Busy_Delhi_road.jpg/640px-Busy_Delhi_road.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Karachi_traffic.jpg/640px-Karachi_traffic.jpg",
            ],
        },
        {
            "filename": "traffic_intersection.jpg",
            "desc":     "Road intersection aerial view",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Road_intersection_by_aerial_photography.jpg/640px-Road_intersection_by_aerial_photography.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Kluang_intersection.JPG/640px-Kluang_intersection.JPG",
            ],
        },
    ],

    # ── Agriculture ───────────────────────────────────────────────────
    "agriculture": [
        {
            "filename": "agriculture_wheat_field.jpg",
            "desc":     "Wheat crop field",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Wheat_close-up.JPG/640px-Wheat_close-up.JPG",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Wheat_field_1.jpg/640px-Wheat_field_1.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Wheat_close-up.jpg/640px-Wheat_close-up.jpg",
            ],
        },
        {
            "filename": "agriculture_diseased_leaves.jpg",
            "desc":     "Plant leaves with visible disease spots",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Late_blight_on_potato_leaf.jpg/640px-Late_blight_on_potato_leaf.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Leaf_blight.jpg/640px-Leaf_blight.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Rust_disease_on_wheat.JPG/640px-Rust_disease_on_wheat.JPG",
            ],
        },
        {
            "filename": "agriculture_rice_paddy.jpg",
            "desc":     "Rice paddy aerial view",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Rice_growing_in_rice_paddy.jpg/640px-Rice_growing_in_rice_paddy.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Rice-paddy.jpg/640px-Rice-paddy.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Paddy_fields.jpg/640px-Paddy_fields.jpg",
            ],
        },
    ],

    # ── Industrial QC ─────────────────────────────────────────────────
    "industrial": [
        {
            "filename": "industrial_metal_surface.jpg",
            "desc":     "Metal surface with texture",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Metal_surface_texture.jpg/640px-Metal_surface_texture.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Aluminium_foil.jpg/640px-Aluminium_foil.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Steel_surface.jpg/640px-Steel_surface.jpg",
            ],
        },
        {
            "filename": "industrial_corroded_pipe.jpg",
            "desc":     "Corroded metal pipe surface — defects visible",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Corrosion_on_a_bolt.jpg/640px-Corrosion_on_a_bolt.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Rust_on_a_bolt.jpg/640px-Rust_on_a_bolt.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Rusty_metal_surface.jpg/640px-Rusty_metal_surface.jpg",
            ],
        },
        {
            "filename": "industrial_circuit_board.jpg",
            "desc":     "PCB circuit board inspection",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Laptop-hard-drive-exposed.jpg/640px-Laptop-hard-drive-exposed.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Dip_PCB_board.jpg/640px-Dip_PCB_board.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/FC_PCB.jpg/640px-FC_PCB.jpg",
            ],
        },
    ],

    # ── Aerial / Satellite ────────────────────────────────────────────
    "aerial": [
        {
            "filename": "aerial_city_top.jpg",
            "desc":     "City aerial top-down view (NASA public domain)",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/640px-24701-nature-natural-beauty.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Satellite_image_of_Islamabad.jpg/640px-Satellite_image_of_Islamabad.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Karachi_from_Space.jpg/640px-Karachi_from_Space.jpg",
            ],
        },
        {
            "filename": "aerial_farmland.jpg",
            "desc":     "Agricultural farmland from above",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/640px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Aerial_view_of_farms_along_the_Murrumbidgee_River.jpg/640px-Aerial_view_of_farms_along_the_Murrumbidgee_River.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Checkerboard_farms.jpg/640px-Checkerboard_farms.jpg",
            ],
        },
        {
            "filename": "aerial_river_delta.jpg",
            "desc":     "River delta satellite view — water + vegetation + soil",
            "urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Nile_River_and_Delta_from_orbit.jpg/640px-Nile_River_and_Delta_from_orbit.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Indus_River_Delta.jpg/640px-Indus_River_Delta.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Mississippi_River_Delta_from_space.jpg/640px-Mississippi_River_Delta_from_space.jpg",
            ],
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────
# Downloader
# ─────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def download(url: str, dest: str) -> bool:
    """Try to download url → dest. Return True on success."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=20) as resp:
            data = resp.read()
        if len(data) < 5000:   # suspiciously small → likely error page
            return False
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        return False


def download_set(domain: str, images: list) -> None:
    print(f"\n{'='*55}")
    print(f"  Domain: {domain.upper()}")
    print(f"{'='*55}")

    domain_dir = os.path.join(SAMPLES_DIR, domain)
    os.makedirs(domain_dir, exist_ok=True)

    ok_count = 0
    for img_def in images:
        dest = os.path.join(domain_dir, img_def["filename"])
        if os.path.exists(dest):
            size_kb = os.path.getsize(dest) // 1024
            print(f"  [SKIP]  {img_def['filename']} (already exists, {size_kb} KB)")
            ok_count += 1
            continue

        downloaded = False
        for url in img_def["urls"]:
            print(f"  [TRY]   {img_def['filename']} ...", end=" ", flush=True)
            if download(url, dest):
                size_kb = os.path.getsize(dest) // 1024
                print(f"OK ({size_kb} KB)")
                downloaded = True
                ok_count += 1
                break
            else:
                print("failed, trying next...")

        if not downloaded:
            print(f"  [FAIL]  {img_def['filename']} — all URLs failed")

    print(f"\n  {ok_count}/{len(images)} images ready in samples/{domain}/")


def main():
    print("\nImage Segmentation — Dataset Downloader")
    print("Downloading public-domain sample images...\n")

    for domain, images in IMAGE_SETS.items():
        download_set(domain, images)

    print("\n" + "="*55)
    print("  Done! Images saved to: samples/")
    print("  Load them in the dashboard via 'Upload Image'")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
