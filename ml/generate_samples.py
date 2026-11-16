"""
ml/generate_samples.py — High-Resolution Botanical Leaf Sample Generator

Generates realistic 400x400 RGB leaf photographs/renders in `sample_images/`
representing PlantVillage diagnostic classes so users and automated tests can
test one-click inference immediately:
- `healthy_leaf.jpg`     (Healthy canopy leaf)
- `early_blight.jpg`     (Alternaria solani concentric target spots + yellow halo)
- `late_blight.jpg`      (Phytophthora infestans water-soaked charcoal blotches)
- `bacterial_spot.jpg`   (Xanthomonas clustered dark greasy flecks + yellow margins)
- `powdery_mildew.jpg`   (Podosphaera white talcum mycelial colonies)
"""

from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = BASE_DIR / "sample_images"


def _create_base_leaf_canvas(width: int = 400, height: int = 400, seed: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a studio-lit dark slate background with a realistic ovate botanical leaf
    complete with midrib and lateral venation. Returns (rgb_float_array, leaf_mask).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width * 0.5) / (width * 0.36)
    ny = (yy - height * 0.52) / (height * 0.40)

    # Pointed ovate leaf geometry
    taper = 1.0 - 0.32 * ny
    serration = 0.025 * np.sin(18.0 * np.arctan2(ny, nx + 1e-5))
    dist = (nx / np.clip(taper, 0.35, 1.4)) ** 2 + ny ** 2
    leaf_mask = dist <= (1.0 + serration)

    # Dark botanical studio background
    bg_r = 16.0 + 10.0 * (1.0 - dist * 0.3)
    bg_g = 32.0 + 14.0 * (1.0 - dist * 0.3)
    bg_b = 26.0 + 12.0 * (1.0 - dist * 0.3)

    img = np.stack([bg_r, bg_g, bg_b], axis=-1)

    # Natural chlorophyll base on leaf lamina
    texture = rng.normal(0.0, 4.5, size=(height, width))
    leaf_r = 46.0 + 18.0 * (1.0 - np.abs(nx)) + texture
    leaf_g = 148.0 + 34.0 * (1.0 - 0.5 * dist) + texture
    leaf_b = 52.0 + 12.0 * (1.0 - dist) + texture * 0.5

    # Primary midrib & lateral veins
    midrib = np.exp(-((nx * 32.0) ** 2)) * (np.abs(ny) < 0.92)
    vein_angle = np.abs(ny - (-0.65 * np.abs(nx) + np.round((ny + 0.65 * np.abs(nx)) * 4.5) / 4.5))
    lateral_veins = np.exp(-((vein_angle * 45.0) ** 2)) * (np.abs(nx) < 0.78)
    veins = np.clip(midrib + 0.55 * lateral_veins, 0.0, 1.0)

    leaf_r = leaf_r + veins * 28.0
    leaf_g = leaf_g + veins * 38.0
    leaf_b = leaf_b + veins * 18.0

    img[leaf_mask, 0] = leaf_r[leaf_mask]
    img[leaf_mask, 1] = leaf_g[leaf_mask]
    img[leaf_mask, 2] = leaf_b[leaf_mask]

    return img, leaf_mask


def generate_sample_leaves() -> dict:
    """Generates all 5 sample leaf images in `sample_images/` and returns metadata."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    height, width = 400, 400
    yy, xx = np.mgrid[0:height, 0:width]

    # 1. Healthy Leaf
    img_healthy, _ = _create_base_leaf_canvas(width, height, seed=101)
    Image.fromarray(np.clip(img_healthy, 0, 255).astype(np.uint8)).save(
        SAMPLE_DIR / "healthy_leaf.jpg", quality=92
    )

    # 2. Early Blight (Concentric brown target rings + bright yellow chlorotic halos)
    img_eb, mask_eb = _create_base_leaf_canvas(width, height, seed=202)
    lesion_centers = [
        (150, 145, 34),
        (245, 175, 38),
        (175, 255, 42),
        (235, 270, 30),
        (135, 210, 26),
        (210, 120, 28),
    ]
    for cx, cy, rad in lesion_centers:
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # Yellow chlorotic halo around lesion
        halo = np.clip(1.0 - d / (rad * 1.65), 0.0, 1.0) * mask_eb
        img_eb[:, :, 0] = img_eb[:, :, 0] * (1.0 - 0.75 * halo) + 218.0 * (0.75 * halo)
        img_eb[:, :, 1] = img_eb[:, :, 1] * (1.0 - 0.75 * halo) + 188.0 * (0.75 * halo)
        img_eb[:, :, 2] = img_eb[:, :, 2] * (1.0 - 0.85 * halo) + 28.0 * (0.85 * halo)

        # Warm brown necrotic spot with concentric target rings
        core = (d <= rad) & mask_eb
        rings = 0.5 + 0.5 * np.cos(d * 0.65)
        img_eb[core, 0] = 112.0 + 32.0 * rings[core]
        img_eb[core, 1] = 68.0 + 22.0 * rings[core]
        img_eb[core, 2] = 28.0 + 10.0 * rings[core]

    Image.fromarray(np.clip(img_eb, 0, 255).astype(np.uint8)).save(
        SAMPLE_DIR / "early_blight.jpg", quality=92
    )

    # 3. Late Blight (Large water-soaked dark charcoal/olive marginal blotches)
    img_lb, mask_lb = _create_base_leaf_canvas(width, height, seed=303)
    blotches = [
        (130, 130, 88),
        (260, 155, 95),
        (225, 265, 92),
        (145, 275, 78),
    ]
    for cx, cy, rad in blotches:
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        blotch = np.clip(1.0 - (d / rad) ** 1.8, 0.0, 1.0) * mask_lb
        img_lb[:, :, 0] = img_lb[:, :, 0] * (1.0 - 0.88 * blotch) + 38.0 * (0.88 * blotch)
        img_lb[:, :, 1] = img_lb[:, :, 1] * (1.0 - 0.88 * blotch) + 42.0 * (0.88 * blotch)
        img_lb[:, :, 2] = img_lb[:, :, 2] * (1.0 - 0.88 * blotch) + 34.0 * (0.88 * blotch)

    Image.fromarray(np.clip(img_lb, 0, 255).astype(np.uint8)).save(
        SAMPLE_DIR / "late_blight.jpg", quality=92
    )

    # 4. Bacterial Spot (Numerous small angular dark flecks + yellow coalescing tissue)
    img_bs, mask_bs = _create_base_leaf_canvas(width, height, seed=404)
    rng_bs = np.random.default_rng(404)
    # Add yellow chlorotic background stress across leaf
    img_bs[mask_bs, 0] = np.clip(img_bs[mask_bs, 0] * 1.65 + 55.0, 0, 230)
    img_bs[mask_bs, 1] = np.clip(img_bs[mask_bs, 1] * 1.05 + 15.0, 0, 210)
    img_bs[mask_bs, 2] = np.clip(img_bs[mask_bs, 2] * 0.45, 0, 65)

    for _ in range(95):
        cx = int(rng_bs.integers(95, 305))
        cy = int(rng_bs.integers(85, 325))
        rad = float(rng_bs.uniform(3.2, 6.5))
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        fleck = (d <= rad) & mask_bs
        halo = (d > rad) & (d <= rad * 1.9) & mask_bs
        img_bs[halo, 0] = 215.0
        img_bs[halo, 1] = 185.0
        img_bs[halo, 2] = 32.0
        img_bs[fleck, 0] = 68.0
        img_bs[fleck, 1] = 42.0
        img_bs[fleck, 2] = 22.0

    Image.fromarray(np.clip(img_bs, 0, 255).astype(np.uint8)).save(
        SAMPLE_DIR / "bacterial_spot.jpg", quality=92
    )

    # 5. Powdery Mildew (Superficial white/ash powdery mycelial colonies across leaf blade)
    img_pm, mask_pm = _create_base_leaf_canvas(width, height, seed=505)
    rng_pm = np.random.default_rng(505)
    noise = rng_pm.uniform(0.82, 1.08, size=(height, width))
    colonies = [
        (165, 145, 58),
        (235, 165, 62),
        (195, 225, 68),
        (145, 245, 52),
        (248, 250, 54),
        (200, 120, 46),
    ]
    for cx, cy, rad in colonies:
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        powder = np.clip(1.0 - (d / rad) ** 1.5, 0.0, 1.0) * mask_pm * noise
        powder = np.clip(powder, 0.0, 0.92)
        img_pm[:, :, 0] = img_pm[:, :, 0] * (1.0 - powder) + 228.0 * powder
        img_pm[:, :, 1] = img_pm[:, :, 1] * (1.0 - powder) + 234.0 * powder
        img_pm[:, :, 2] = img_pm[:, :, 2] * (1.0 - powder) + 226.0 * powder

    Image.fromarray(np.clip(img_pm, 0, 255).astype(np.uint8)).save(
        SAMPLE_DIR / "powdery_mildew.jpg", quality=92
    )

    return {
        "healthy_leaf.jpg": "Healthy",
        "early_blight.jpg": "Early Blight",
        "late_blight.jpg": "Late Blight",
        "bacterial_spot.jpg": "Bacterial Spot",
        "powdery_mildew.jpg": "Powdery Mildew",
    }


if __name__ == "__main__":
    generated = generate_sample_leaves()
    print(f"[AgroScan Samples] Generated {len(generated)} sample leaf images in {SAMPLE_DIR}")
