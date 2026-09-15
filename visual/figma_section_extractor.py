"""
Figma section extractor — crops named sections from a full Figma frame PNG.

Figma exports the entire frame as one image (@2x scale).  The section
descriptors in frame["sections"] hold coordinates relative to the frame's
top-left corner in *Figma logical pixels*.  We scale those to the actual
downloaded PNG resolution and crop.
"""
import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)


def crop_figma_sections(frame: dict) -> list[dict]:
    """
    Crop each section from the frame's full PNG.

    Args:
        frame: dict from figma_extractor with keys:
               image_bytes, width (logical), height (logical), sections

    Returns:
        List of section dicts, each with an added 'image_bytes' key containing
        the cropped JPEG bytes.  Sections that map outside the image are skipped.
    """
    img_bytes = frame.get("image_bytes")
    raw_sections = frame.get("sections", [])

    if not img_bytes or not raw_sections:
        return []

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning(f"Cannot open Figma frame image: {exc}")
        return []

    img_w, img_h = img.size
    frame_w = frame.get("width") or img_w
    frame_h = frame.get("height") or img_h

    # Scale factors: Figma exports at @2x so img_w = frame_w * 2 typically,
    # but we use actual ratios to be safe.
    sx = img_w / frame_w
    sy = img_h / frame_h

    result = []
    for sec in raw_sections:
        x1 = max(0, int(sec["rel_x"] * sx))
        y1 = max(0, int(sec["rel_y"] * sy))
        x2 = min(img_w, int((sec["rel_x"] + sec["width"])  * sx))
        y2 = min(img_h, int((sec["rel_y"] + sec["height"]) * sy))

        if x2 <= x1 or y2 <= y1:
            logger.debug(f"Figma section '{sec['name']}' maps outside image — skipped")
            continue

        try:
            crop = img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=90)
            result.append({
                **sec,
                "image_bytes": buf.getvalue(),
                "crop_x1": x1, "crop_y1": y1,
                "crop_x2": x2, "crop_y2": y2,
            })
        except Exception as exc:
            logger.warning(f"Crop failed for Figma section '{sec['name']}': {exc}")

    logger.info(
        f"Figma section crops — frame='{frame.get('name', '?')}', "
        f"total={len(raw_sections)}, cropped={len(result)}"
    )
    return result