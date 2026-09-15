import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageChops, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

_CANONICAL_WIDTH = 1440   # fallback logical width when a frame's own width is unavailable
_REGION_MIN_AREA = 400    # ignore diff blobs smaller than this (noise filter)
_MAX_REGIONS = 10         # cap number of reported regions


@dataclass
class DiffRegion:
    x: int
    y: int
    width: int
    height: int
    diff_percent: float   # % of pixels in this region that differ


@dataclass
class CompareResult:
    diff_percent: float           # overall % of pixels that differ
    regions: list[DiffRegion]     # bounding boxes of changed areas
    diff_image: bytes             # PNG bytes — original with red boxes drawn
    diff_mask: bytes              # PNG bytes — grayscale diff heatmap


def _load_image(data: bytes, label: str) -> Image.Image:
    if not data:
        raise ValueError(f"{label} image is empty (0 bytes)")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if img.width == 0 or img.height == 0:
        raise ValueError(f"{label} image has zero dimensions")
    return img


def _normalize(img: Image.Image, label: str, canonical_width: int = _CANONICAL_WIDTH) -> Image.Image:
    """
    Resize to the target frame's logical width, preserving aspect ratio.
    Both Figma @2x exports and Playwright @2x screenshots land at 2x the
    frame's logical width — we normalize both down to that same logical
    width so diffs are computed on the same grid.
    """
    if img.width == 0:
        raise ValueError(f"{label}: zero-width image cannot be normalized")
    scale = canonical_width / img.width
    new_h = int(img.height * scale)
    resized = img.resize((canonical_width, new_h), Image.LANCZOS)
    logger.debug(f"{label}: {img.width}x{img.height} → {canonical_width}x{new_h}")
    return resized


def _match_heights(a: Image.Image, b: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Pad the shorter image at the bottom with white to match heights."""
    if a.height == b.height:
        return a, b
    target_h = max(a.height, b.height)

    def pad(img: Image.Image) -> Image.Image:
        if img.height == target_h:
            return img
        canvas = Image.new("RGB", (img.width, target_h), (255, 255, 255))
        canvas.paste(img, (0, 0))
        return canvas

    return pad(a), pad(b)


def _find_regions(diff_gray: Image.Image, threshold: int = 30) -> list[tuple[int, int, int, int]]:
    """
    Find bounding boxes of changed regions in a grayscale diff image.
    Returns list of (x, y, w, h) tuples, largest first.
    """
    # Threshold → binary mask
    binary = diff_gray.point(lambda p: 255 if p > threshold else 0)
    # Dilate slightly to merge nearby blobs
    binary = binary.filter(ImageFilter.MaxFilter(15))

    pixels = binary.load()
    width, height = binary.size
    visited = [[False] * height for _ in range(width)]
    regions = []

    def bfs(sx: int, sy: int) -> tuple[int, int, int, int] | None:
        queue = [(sx, sy)]
        visited[sx][sy] = True
        min_x = max_x = sx
        min_y = max_y = sy
        count = 0
        while queue:
            cx, cy = queue.pop()
            count += 1
            min_x = min(min_x, cx)
            max_x = max(max_x, cx)
            min_y = min(min_y, cy)
            max_y = max(max_y, cy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and not visited[nx][ny] and pixels[nx, ny] > 0:
                    visited[nx][ny] = True
                    queue.append((nx, ny))
        area = (max_x - min_x + 1) * (max_y - min_y + 1)
        return (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1) if area >= _REGION_MIN_AREA else None

    for x in range(width):
        for y in range(height):
            if pixels[x, y] > 0 and not visited[x][y]:
                region = bfs(x, y)
                if region:
                    regions.append(region)

    # Sort by area descending, cap at _MAX_REGIONS
    regions.sort(key=lambda r: r[2] * r[3], reverse=True)
    return regions[:_MAX_REGIONS]


def _blur_for_diff(img: Image.Image, radius: int = 1) -> Image.Image:
    """
    Light Gaussian blur before differencing.
    Eliminates 1–2px sub-pixel anti-aliasing artefacts that produce spurious
    diffs on text and borders — the single biggest source of false positives.
    """
    return img.filter(ImageFilter.GaussianBlur(radius))


def compare(
    figma_bytes: bytes,
    live_bytes: bytes,
    diff_threshold: float = 0.05,
    canonical_width: int = _CANONICAL_WIDTH,
) -> CompareResult:
    """
    Compare a Figma frame image against a live full-page screenshot.

    diff_threshold: pixel-level sensitivity (0.0–1.0).
      Differences below this fraction of max channel distance are ignored (noise filter).
    canonical_width: logical width (matching the Figma frame's own width) both
      images are normalized to before diffing.
    """
    figma_img = _load_image(figma_bytes, "Figma")
    live_img = _load_image(live_bytes, "Live")

    # Normalize both to the same logical resolution
    figma_norm = _normalize(figma_img, "Figma", canonical_width)
    live_norm = _normalize(live_img, "Live", canonical_width)

    # Match heights by padding the shorter one to enable full-page comparison
    figma_norm, live_norm = _match_heights(figma_norm, live_norm)

    # Pre-blur both images to suppress sub-pixel AA noise before diff
    figma_blurred = _blur_for_diff(figma_norm)
    live_blurred  = _blur_for_diff(live_norm)

    # Pixel diff on blurred images (noise-reduced), but regions refer to original coords
    diff = ImageChops.difference(figma_blurred, live_blurred)
    diff_gray = diff.convert("L")

    # Apply threshold — treat differences below threshold*255 as identical
    threshold_val = int(diff_threshold * 255)
    diff_thresholded = diff_gray.point(lambda p: p if p > threshold_val else 0)

    # Overall diff percentage
    total_pixels = figma_norm.width * figma_norm.height
    diff_pixels = sum(1 for p in diff_thresholded.getdata() if p > 0)
    diff_percent = round((diff_pixels / total_pixels) * 100, 2)

    logger.info(f"Diff: {diff_percent}% pixels differ ({diff_pixels}/{total_pixels})")

    # Find changed regions
    raw_regions = _find_regions(diff_thresholded)

    regions = []
    for (x, y, w, h) in raw_regions:
        region_total = w * h
        region_diff = sum(
            1 for px in list(diff_thresholded.crop((x, y, x + w, y + h)).getdata())
            if px > 0
        )
        regions.append(DiffRegion(
            x=x, y=y, width=w, height=h,
            diff_percent=round((region_diff / region_total) * 100, 2),
        ))

    # Build annotated diff image — live screenshot with red boxes on changed regions
    annotated = live_norm.copy()
    draw = ImageDraw.Draw(annotated)
    for r in regions:
        draw.rectangle([r.x, r.y, r.x + r.width, r.y + r.height], outline=(255, 0, 0), width=3)

    # Encode outputs
    diff_image_buf = io.BytesIO()
    annotated.save(diff_image_buf, format="PNG")

    diff_mask_buf = io.BytesIO()
    # Enhance the mask for visibility
    diff_thresholded.save(diff_mask_buf, format="PNG")

    return CompareResult(
        diff_percent=diff_percent,
        regions=regions,
        diff_image=diff_image_buf.getvalue(),
        diff_mask=diff_mask_buf.getvalue(),
    )