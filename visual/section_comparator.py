"""
Section comparator — compares a Figma section crop against a live DOM section
screenshot using block-SSIM (structural similarity) and pixel diff.

numpy is available transitively via sentence-transformers / faiss-cpu.
No OpenCV required.
"""
import io
import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

_CANONICAL_W = 800   # resize both crops to this width before comparing
_BLOCK_SIZE  = 8     # SSIM block size in pixels
_C1 = (0.01 * 255) ** 2   # SSIM stability constants
_C2 = (0.03 * 255) ** 2


@dataclass
class SectionCompareResult:
    figma_section_name: str
    live_section_name:  str
    match_confidence:   float          # 0–100 from section_matcher
    ssim_score:         float          # 0 (worst) – 1 (identical)
    diff_percent:       float          # % of pixels above threshold
    has_significant_diff: bool
    figma_image_bytes:  bytes = field(default=b"", repr=False)
    live_image_bytes:   bytes = field(default=b"", repr=False)
    diff_image_bytes:   bytes = field(default=b"", repr=False)


def _load_gray(img_bytes: bytes, target_w: int) -> np.ndarray:
    """Open image, resize to target_w preserving aspect ratio, convert to float32 grayscale."""
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    scale = target_w / img.width
    target_h = max(1, int(img.height * scale))
    img = img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(img, dtype=np.float32)


def _pad_to_same_height(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad the shorter array at the bottom with 255 (white)."""
    if a.shape[0] == b.shape[0]:
        return a, b
    target_h = max(a.shape[0], b.shape[0])
    w = a.shape[1]  # both have the same width after _load_gray

    def pad(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == target_h:
            return arr
        pad_rows = np.full((target_h - arr.shape[0], w), 255.0, dtype=np.float32)
        return np.vstack([arr, pad_rows])

    return pad(a), pad(b)


def _block_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute mean SSIM over non-overlapping 8×8 blocks.
    Returns a value in [-1, 1]; typically 0.7–1.0 for similar images.
    """
    h, w = a.shape
    scores: list[float] = []

    for y in range(0, h - _BLOCK_SIZE + 1, _BLOCK_SIZE):
        for x in range(0, w - _BLOCK_SIZE + 1, _BLOCK_SIZE):
            ba = a[y:y + _BLOCK_SIZE, x:x + _BLOCK_SIZE].ravel()
            bb = b[y:y + _BLOCK_SIZE, x:x + _BLOCK_SIZE].ravel()

            mu_a, mu_b = ba.mean(), bb.mean()
            var_a = ba.var()
            var_b = bb.var()
            cov   = float(np.mean((ba - mu_a) * (bb - mu_b)))

            num   = (2 * mu_a * mu_b + _C1) * (2 * cov + _C2)
            denom = (mu_a ** 2 + mu_b ** 2 + _C1) * (var_a + var_b + _C2)
            scores.append(num / denom if denom else 1.0)

    return float(np.mean(scores)) if scores else 1.0


def _build_diff_image(a: np.ndarray, b: np.ndarray, threshold: float) -> bytes:
    """Produce a grayscale diff heatmap as PNG bytes."""
    diff = np.abs(a - b)
    diff[diff <= threshold] = 0
    diff_img = Image.fromarray(diff.clip(0, 255).astype(np.uint8)).convert("L")
    buf = io.BytesIO()
    diff_img.save(buf, format="PNG")
    return buf.getvalue()


def compare_section_pair(
    figma_sec:       dict,
    live_sec:        dict,
    match_confidence: float,
    diff_threshold:  float = 0.05,
    ssim_threshold:  float = 0.82,
    diff_pct_threshold: float = 8.0,
) -> SectionCompareResult:
    """
    Compare one matched (figma_sec, live_sec) pair.

    figma_sec must have 'image_bytes'.
    live_sec  must have 'screenshot' (bytes or None).

    Returns a SectionCompareResult with SSIM, diff_percent, and a
    has_significant_diff flag that controls whether AI analysis is triggered.
    """
    figma_bytes = figma_sec.get("image_bytes", b"") or b""
    live_bytes  = live_sec.get("screenshot",   b"") or b""
    live_name   = live_sec.get("heading") or live_sec.get("id") or live_sec.get("sectionType") or "section"

    if not figma_bytes or not live_bytes:
        return SectionCompareResult(
            figma_section_name=figma_sec.get("name", ""),
            live_section_name=live_name,
            match_confidence=match_confidence,
            ssim_score=0.0,
            diff_percent=100.0,
            has_significant_diff=True,
            figma_image_bytes=figma_bytes,
            live_image_bytes=live_bytes,
        )

    try:
        figma_arr = _load_gray(figma_bytes, _CANONICAL_W)
        live_arr  = _load_gray(live_bytes,  _CANONICAL_W)
        figma_arr, live_arr = _pad_to_same_height(figma_arr, live_arr)

        # Light blur to suppress sub-pixel AA noise (same strategy as visual_comparator)
        figma_blur = np.array(
            Image.fromarray(figma_arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)),
            dtype=np.float32,
        )
        live_blur = np.array(
            Image.fromarray(live_arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)),
            dtype=np.float32,
        )

        ssim = _block_ssim(figma_blur, live_blur)

        threshold_val = diff_threshold * 255
        diff = np.abs(figma_blur - live_blur)
        diff_pixels  = int(np.sum(diff > threshold_val))
        total_pixels = figma_blur.size
        diff_percent = round((diff_pixels / total_pixels) * 100, 2)

        diff_img_bytes = _build_diff_image(figma_blur, live_blur, threshold_val)
        significant = (ssim < ssim_threshold) or (diff_percent > diff_pct_threshold)

        logger.info(
            f"Section compare — '{figma_sec.get('name', '?')}' vs '{live_name}': "
            f"ssim={ssim:.3f}, diff={diff_percent}%, significant={significant}"
        )

        return SectionCompareResult(
            figma_section_name=figma_sec.get("name", ""),
            live_section_name=live_name,
            match_confidence=match_confidence,
            ssim_score=round(ssim, 4),
            diff_percent=diff_percent,
            has_significant_diff=significant,
            figma_image_bytes=figma_bytes,
            live_image_bytes=live_bytes,
            diff_image_bytes=diff_img_bytes,
        )

    except Exception as exc:
        logger.warning(
            f"Section compare failed — '{figma_sec.get('name', '?')}' vs '{live_name}': {exc}"
        )
        return SectionCompareResult(
            figma_section_name=figma_sec.get("name", ""),
            live_section_name=live_name,
            match_confidence=match_confidence,
            ssim_score=0.0,
            diff_percent=100.0,
            has_significant_diff=True,
            figma_image_bytes=figma_bytes,
            live_image_bytes=live_bytes,
        )