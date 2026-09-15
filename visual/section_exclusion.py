"""
Section exclusion — lets a job skip specific semantic sections (header,
footer, collection grid, etc.) so issues in those areas never reach the
final report, without needing per-run DOM/CSS configuration from the caller.

Reuses section_alignment_engine's keyword-based section-type classifier so
"header"/"footer"/"collection" behave consistently with how sections are
matched elsewhere in the pipeline.
"""
import io

from PIL import Image, ImageDraw

from visual.section_alignment_engine import _detect_type, _section_label

# Friendlier aliases users are likely to type, mapped to _detect_type's
# canonical category names.
_TYPE_ALIASES = {
    "header": "nav",
    "navigation": "nav",
    "nav": "nav",
    "menu": "nav",
    "footer": "footer",
    "collection": "collection",
    "collections": "collection",
    "hero": "hero",
    "newsletter": "newsletter",
    "subscribe": "newsletter",
    "faq": "faq",
    "testimonial": "testimonial",
    "testimonials": "testimonial",
    "announcement": "announcement",
}


def normalize_exclude_types(raw: list[str] | None) -> set[str]:
    """Map user-provided section names to canonical section-type categories."""
    if not raw:
        return set()
    return {_TYPE_ALIASES.get(r.strip().lower(), r.strip().lower()) for r in raw if r.strip()}


def excluded_regions(
    dom_sections: list[dict],
    figma_sections: list[dict],
    exclude_types: set[str],
) -> list[tuple[float, float, float, float]]:
    """
    Return (x, y, width, height) boxes for any DOM or Figma section whose
    classified type is in exclude_types. dom_sections must already be
    normalized to canonical width; figma_sections are frame-relative, which
    is the same canonical space by construction.
    """
    if not exclude_types:
        return []

    boxes = []
    for sec in dom_sections:
        if _detect_type(_section_label(sec)) in exclude_types:
            boxes.append((sec.get("x", 0), sec.get("y", 0), sec.get("width", 0), sec.get("height", 0)))

    for sec in figma_sections:
        if _detect_type(sec.get("name", "")) in exclude_types:
            boxes.append((sec.get("rel_x", 0), sec.get("rel_y", 0), sec.get("width", 0), sec.get("height", 0)))

    return boxes


def filter_out_sections(
    dom_sections: list[dict],
    figma_sections: list[dict],
    exclude_types: set[str],
) -> tuple[list[dict], list[dict]]:
    """
    Return (dom_sections, figma_sections) with excluded-type sections removed.
    Used to keep the section-matching pipeline from spending AI vision calls
    on sections that will be dropped from the report anyway.
    """
    if not exclude_types:
        return dom_sections, figma_sections

    kept_dom = [s for s in dom_sections if _detect_type(_section_label(s)) not in exclude_types]
    kept_figma = [s for s in figma_sections if _detect_type(s.get("name", "")) not in exclude_types]
    return kept_dom, kept_figma


def _overlap_fraction(issue_box: tuple, excl_box: tuple) -> float:
    """Fraction of issue_box's area that overlaps excl_box (0.0–1.0)."""
    ix, iy, iw, ih = issue_box
    ex, ey, ew, eh = excl_box
    if iw <= 0 or ih <= 0:
        return 0.0
    ox = max(0, min(ix + iw, ex + ew) - max(ix, ex))
    oy = max(0, min(iy + ih, ey + eh) - max(iy, ey))
    issue_area = iw * ih
    return (ox * oy) / issue_area if issue_area else 0.0


def mask_excluded_regions(
    img_bytes: bytes,
    exclude_boxes: list[tuple[float, float, float, float]],
    canonical_width: int,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> bytes:
    """
    Paint over excluded-section boxes (header/footer/etc.) directly in the
    image, in place of the previous approach of only filtering issues whose
    region *mostly* overlapped an excluded box after the fact. That let a
    single merged diff region spanning both an excluded and a kept section
    (e.g. footer + product grid) survive if the excluded portion was under the
    overlap threshold — so header/footer content still leaked into a kept
    issue's crop and into the full-page image sent to the vision model.

    Masking before comparison/analysis instead guarantees excluded content
    never enters a diff region and never reaches the vision model at all.

    exclude_boxes are in canonical_width space (same as dom/figma section
    coords); img_bytes may be at a different native resolution (e.g. a @2x
    export), so boxes are scaled by the image's actual width ratio.
    """
    if not exclude_boxes:
        return img_bytes

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    scale = img.width / canonical_width if canonical_width else 1.0

    draw = ImageDraw.Draw(img)
    for (x, y, w, h) in exclude_boxes:
        x1, y1 = x * scale, y * scale
        x2, y2 = (x + w) * scale, (y + h) * scale
        draw.rectangle([x1, y1, x2, y2], fill=fill)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def filter_excluded_issues(
    issues: list[dict],
    exclude_boxes: list[tuple[float, float, float, float]],
    min_overlap: float = 0.4,
) -> list[dict]:
    """Drop issues whose region substantially overlaps any excluded box."""
    if not exclude_boxes:
        return issues

    kept = []
    for issue in issues:
        box = (issue.get("x", 0), issue.get("y", 0), issue.get("width", 0), issue.get("height", 0))
        if any(_overlap_fraction(box, ex) >= min_overlap for ex in exclude_boxes):
            continue
        kept.append(issue)
    return kept
