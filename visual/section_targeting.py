"""
Section targeting — locate one named section (by a free-text query, e.g.
"Curated Selection (Featured Pieces)") on both the live DOM and the Figma
design, so a QA run can be scoped to exactly that section instead of the
whole page.

The live side is straightforward: DOM sections carry real headings/text, so
matching is just fuzzy text similarity (reusing section_matcher's scoring).

The Figma side is harder when the frame wasn't broken into separately-named
sub-layers (a single "01-Home Page" frame with no child sections, as opposed
to a design with a named layer per section) — there's no sub-region to look
up by name. In that case we anchor on the Figma TEXT node whose own content
best matches the section's heading, then derive a region around it scaled by
the live section's height (proportional to the live-page-height ↔
figma-frame-height ratio, since the two are rarely pixel-identical).
"""
import io
import logging

from PIL import Image

from visual.section_matcher import _similarity
from visual.section_alignment_engine import _detect_type, _section_label

logger = logging.getLogger(__name__)

# Below this word/char-similarity score, treat it as "no confident match"
# rather than silently picking the least-bad candidate. Split into two
# constants: live-side DOM headings tend to be fairly unique per section, so
# the original permissive floor is kept there; the Figma-side anchor search
# needs a stricter floor — on a page with several near-identical repeating
# cards sharing boilerplate phrasing, 0.15 was loose enough to lock onto a
# text node from an entirely different card (confirmed root cause of a real
# false positive).
_MIN_LIVE_MATCH_SCORE = 0.15
_MIN_FIGMA_ANCHOR_SCORE = 0.35


def select_first_n_sections(
    dom_sections: list[dict],
    n: int,
    exclude_types: set[str] | None = None,
    from_end: bool = False,
) -> list[dict]:
    """
    Pick N sections in page order (top to bottom), after dropping any
    section whose classified type is in exclude_types (e.g. {"nav",
    "footer"}) — so "test the first 3 sections" means the first 3 sections a
    user would actually scroll through as real content, not the first 3
    entries in whatever order the DOM happened to list them, and not header/
    footer even if they appear early/late in that raw order.

    from_end=True picks the LAST N instead (e.g. "test the last 3 sections"
    — a closing/footer-adjacent block, contact section, etc.) — still
    returned in page order (top to bottom), not reversed, so downstream
    per-section iteration reads top-to-bottom either way.
    """
    exclude_types = exclude_types or set()
    kept = [
        s for s in dom_sections
        if _detect_type(_section_label(s)) not in exclude_types
    ]
    kept.sort(key=lambda s: s.get("y", 0))
    if from_end:
        return kept[-n:] if n > 0 else []
    return kept[:n]


def _label_for_live_section(sec: dict) -> str:
    return " ".join(filter(None, [
        sec.get("heading", ""),
        sec.get("textSnippet", "")[:150],
        sec.get("id", ""),
    ]))


def find_live_section(query: str, dom_sections: list[dict]) -> dict | None:
    """Best-matching live DOM section for a free-text section-name query."""
    best, best_score = None, 0.0
    for sec in dom_sections:
        score = _similarity(query, _label_for_live_section(sec))
        if score > best_score:
            best, best_score = sec, score

    if best is None or best_score < _MIN_LIVE_MATCH_SCORE:
        logger.info(f"No confident live-section match for {query!r} (best={best_score:.2f})")
        return None

    logger.info(
        f"Section target match — query={query!r} -> "
        f"heading={best.get('heading')!r} (score={best_score:.2f})"
    )
    return best


def find_figma_anchor_node(
    query: str,
    figma_text_nodes: list[dict],
    expected_y: float | None = None,
    search_radius: float | None = None,
    used_indices: set[int] | None = None,
) -> tuple[dict, int] | None:
    """
    Best-matching Figma TEXT node for the query — used to locate a section's
    approximate position when the Figma frame has no separately-named
    sub-layer for it.

    When expected_y/search_radius are given, searches only the nodes within
    that vertical window first (skipping any index already in used_indices),
    only falling back to a full-page search if nothing in the window clears
    the similarity floor. On a page with several near-identical repeating
    cards sharing boilerplate phrasing, an unscoped global search can lock
    onto a text node from a different card purely because it scores
    marginally higher — preferring a nearby candidate first avoids that.

    used_indices lets a caller exclude nodes already committed as another
    section's anchor in the same run — without this, two different live
    sections can independently resolve to the SAME Figma anchor.

    Returns (node, index) so the caller can track which node was used, or
    None if nothing cleared the floor.
    """
    used_indices = used_indices or set()

    def _search(indices) -> tuple[dict, int] | None:
        best, best_idx, best_score = None, None, 0.0
        for i in indices:
            if i in used_indices:
                continue
            score = _similarity(query, figma_text_nodes[i].get("text", ""))
            if score > best_score:
                best, best_idx, best_score = figma_text_nodes[i], i, score
        if best is None or best_score < _MIN_FIGMA_ANCHOR_SCORE:
            return None
        return best, best_idx

    if expected_y is not None and search_radius is not None:
        windowed = [
            i for i, node in enumerate(figma_text_nodes)
            if abs(node.get("y", 0) - expected_y) <= search_radius
        ]
        result = _search(windowed)
        if result is not None:
            return result

    return _search(range(len(figma_text_nodes)))


def derive_figma_region(
    live_section: dict,
    figma_text_nodes: list[dict],
    live_page_height: float,
    figma_frame_height: float,
    frame_width: float,
    used_anchor_indices: set[int] | None = None,
) -> dict:
    """
    Derive a Figma region {x, y, width, height} (frame-relative, logical px)
    corresponding to a live DOM section, for frames with no separate named
    sub-layer to crop directly.

    used_anchor_indices: pass a set that's shared/reused across every section
    in one run (e.g. run_multi_section_qa's per-section loop) so a Figma text
    node already claimed as an earlier section's anchor can't also become a
    later section's anchor — mutated in place, so the caller's set
    accumulates automatically. Without this, two different live sections
    (e.g. two cards in a repeating "fleet" listing) can independently
    resolve to the SAME Figma anchor, producing two report entries that
    quote the same Figma text with reversed, contradictory numbers.
    """
    query = live_section.get("heading") or live_section.get("textSnippet", "")[:60]

    scale = (figma_frame_height / live_page_height) if live_page_height else 1.0
    height = live_section.get("height", 0) * scale
    # Used both as the anchor search's positional prior AND the no-anchor
    # fallback position — the section's own proportional position is a
    # reasonable "expected" Figma y even before any text match is attempted.
    expected_y = (live_section.get("y", 0) / live_page_height) * figma_frame_height if live_page_height else 0.0
    search_radius = max(3 * height, 200)

    result = find_figma_anchor_node(
        query, figma_text_nodes,
        expected_y=expected_y, search_radius=search_radius,
        used_indices=used_anchor_indices,
    )

    if result is not None:
        anchor, anchor_idx = result
        if used_anchor_indices is not None:
            used_anchor_indices.add(anchor_idx)
        # The section itself usually starts a bit above its own heading text
        # (an eyebrow label, vertical padding) — nudge the top up slightly
        # rather than starting exactly at the heading's own top edge.
        y = max(0.0, anchor["y"] - height * 0.15)
        logger.info(
            f"Figma region anchored on text node {anchor.get('text', '')[:40]!r} at y={anchor['y']:.0f}"
        )
    else:
        # No text anchor found — fall back to the live section's own relative
        # vertical position, scaled onto the Figma frame's height.
        y = expected_y
        logger.info("No Figma text anchor found — falling back to proportional position")

    return {"x": 0.0, "y": y, "width": frame_width, "height": height}


def filter_nodes_in_region(nodes: list[dict], y_start: float, y_end: float) -> list[dict]:
    """Keep only text nodes whose vertical center falls within [y_start, y_end]."""
    kept = []
    for n in nodes:
        center = n.get("y", 0) + n.get("height", 0) / 2
        if y_start <= center <= y_end:
            kept.append(n)
    return kept


def crop_frame_region(image_bytes: bytes, region: dict, canonical_width: float) -> bytes:
    """Crop a region (logical px, relative to `canonical_width`) out of a
    full-frame/full-page PNG, returning PNG bytes."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    scale = (img.width / canonical_width) if canonical_width else 1.0

    x1 = max(0, int(region["x"] * scale))
    y1 = max(0, int(region["y"] * scale))
    x2 = min(img.width, int((region["x"] + region["width"]) * scale))
    y2 = min(img.height, int((region["y"] + region["height"]) * scale))
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)

    cropped = img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()
