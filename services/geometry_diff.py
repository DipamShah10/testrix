"""
Geometry diff engine — compares real element dimensions (images, buttons) and
real rendered spacing (gaps above/below elements) between Figma's design
geometry and the live DOM's rendered layout.

Same philosophy as typography_diff.py: measured facts from actual geometry on
both sides — real pixel boxes from Figma's API and real getBoundingClientRect()
boxes from the live page — not a vision model's guess from two screenshots.
"""
import logging
from difflib import SequenceMatcher

from services.typography_diff import match_text_nodes
from visual.section_matcher import match_sections

logger = logging.getLogger(__name__)

_SIZE_TOLERANCE_PCT = 8.0      # relative width/height difference below which "same size"
_SIZE_MIN_ABS_PX = 4           # ignore differences smaller than this outright (sub-pixel/rounding noise)

# Spacing tolerance is a flat 2px absolute floor, per explicit direction — a
# gap that's off by ≤2px is normal sub-pixel/rounding noise and gets ignored
# outright, regardless of the gap's own size. This is deliberately NOT a
# percentage-of-gap tolerance (a relative rule would let big gaps drift by
# 20px+ while a "same absolute miss" on a small gap gets flagged) — every
# spacing check in this file uses the same fixed 2px bar.
_SPACING_MIN_ABS_PX = 2

_MIN_MATCH_SIMILARITY = 0.35   # label similarity floor for matching two elements as "the same one"


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def match_elements(
    figma_items: list[dict],
    live_items: list[dict],
    label_field: str,
) -> list[tuple[dict, dict]]:
    """
    Greedy best-match pairing of Figma elements to live elements, by label
    similarity (alt text / button caption) with positional proximity as a
    tiebreaker — same approach as typography_diff.match_text_nodes.

    When neither side has a label (e.g. a purely decorative image with no alt
    text), falls back to positional proximity alone.
    """
    if not figma_items or not live_items:
        return []

    candidates = []
    for fi, f in enumerate(figma_items):
        f_label = f.get(label_field, "") or ""
        for li, l in enumerate(live_items):
            l_label = l.get(label_field, "") or ""
            has_label = bool(f_label or l_label)
            sim = _text_similarity(f_label, l_label) if has_label else 0.5
            if has_label and sim < _MIN_MATCH_SIMILARITY:
                continue
            dist = (
                (f.get("x", 0) - l.get("x", 0)) ** 2
                + (f.get("y", 0) - l.get("y", 0)) ** 2
            ) ** 0.5
            score = sim - (dist / 5000)
            candidates.append((score, fi, li))

    candidates.sort(key=lambda c: -c[0])

    used_f: set[int] = set()
    used_l: set[int] = set()
    pairs: list[tuple[dict, dict]] = []
    for score, fi, li in candidates:
        if fi in used_f or li in used_l:
            continue
        used_f.add(fi)
        used_l.add(li)
        pairs.append((figma_items[fi], live_items[li]))

    return pairs


def compare_dimensions(
    figma_items: list[dict],
    live_items: list[dict],
    label_field: str,
    element_kind: str,
) -> list[dict]:
    """
    Compare width/height between matched Figma/live elements (images or
    buttons). Returns a list of issue dicts (issue_type='dimension').
    """
    pairs = match_elements(figma_items, live_items, label_field)
    issues: list[dict] = []

    for fitem, litem in pairs:
        fw, fh = fitem.get("width", 0), fitem.get("height", 0)
        lw, lh = litem.get("width", 0), litem.get("height", 0)
        if not fw or not fh:
            continue

        w_diff_pct = abs(fw - lw) / fw * 100
        h_diff_pct = abs(fh - lh) / fh * 100
        w_flag = abs(fw - lw) >= _SIZE_MIN_ABS_PX and w_diff_pct > _SIZE_TOLERANCE_PCT
        h_flag = abs(fh - lh) >= _SIZE_MIN_ABS_PX and h_diff_pct > _SIZE_TOLERANCE_PCT
        if not (w_flag or h_flag):
            continue

        label = fitem.get(label_field) or litem.get(label_field) or element_kind
        parts = []
        if w_flag:
            parts.append(f"width is {lw:.0f}px on the live site vs {fw:.0f}px in Figma ({w_diff_pct:.0f}% off)")
        if h_flag:
            parts.append(f"height is {lh:.0f}px on the live site vs {fh:.0f}px in Figma ({h_diff_pct:.0f}% off)")

        magnitude = max(w_diff_pct if w_flag else 0.0, h_diff_pct if h_flag else 0.0)

        issues.append({
            "element": f"{element_kind}: {label}"[:80] if label else element_kind,
            "description": f'"{label}" — ' + "; ".join(parts) + "." if label else "; ".join(parts) + ".",
            "user_impact": (
                f"The {element_kind} renders at a different size than designed, "
                "which can throw off surrounding layout and visual balance."
            ),
            "suggested_fix": f"Match the live {element_kind}'s dimensions to the Figma spec ({fw:.0f}×{fh:.0f}px).",
            "issue_type": "dimension",
            "x": litem.get("x", 0), "y": litem.get("y", 0),
            "width": lw, "height": lh,
            # The matched Figma element's own box, kept separate from the
            # live box above — see typography_diff.py's compare_typography
            # for why: reusing live coords against the Figma image produces
            # a blank/wrong crop.
            "figma_x": fitem.get("x", 0), "figma_y": fitem.get("y", 0),
            "figma_width": fw, "figma_height": fh,
            "diff_percent": min(100.0, magnitude),
            # Consumed directly by severity_classifier, same pattern as
            # typography's magnitude_score — a 9% size miss shouldn't rank the
            # same as a completely wrong-sized element.
            "magnitude_score": min(40.0, magnitude),
        })

    return issues


def _effective_box(item: dict) -> dict:
    """
    The box to use for neighbor-gap math — a node's own padding-aware
    `spacing_box` when it has one (a short/icon-like text node wrapped in a
    padded container: a callout, a chip, a labeled icon), otherwise its own
    tight box. Using the raw tight box for a node like a lone "!" measures
    gaps from the glyph's own edge, not the visual container a human actually
    perceives — silently folding the container's internal padding into
    whatever gap is being measured next to it.
    """
    box = item.get("spacing_box")
    return box if box else item


def _union_box(a: dict, b: dict) -> dict:
    """Bounding box that encloses both a and b — used so a spacing issue's
    crop actually shows the whitespace gap being measured (target + its
    neighbor + the space between them), instead of a tight crop around just
    the target that cuts the gap off entirely and leaves nothing for a human
    reviewer to actually verify."""
    x0 = min(a["x"], b["x"])
    y0 = min(a["y"], b["y"])
    x1 = max(a["x"] + a["width"], b["x"] + b["width"])
    y1 = max(a["y"] + a["height"], b["y"] + b["height"])
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def _find_nearest_above(target: dict, boxes: list[dict]) -> tuple[dict | None, float | None]:
    """Nearest box directly above `target` with horizontal overlap."""
    t = _effective_box(target)
    best, best_gap = None, None
    tx0, tx1 = t["x"], t["x"] + t["width"]
    for b in boxes:
        if b is target:
            continue
        bx = _effective_box(b)
        bx0, bx1 = bx["x"], bx["x"] + bx["width"]
        if bx1 <= tx0 or bx0 >= tx1:
            continue
        bottom = bx["y"] + bx["height"]
        if bottom <= t["y"]:
            gap = t["y"] - bottom
            if best_gap is None or gap < best_gap:
                best, best_gap = b, gap
    return best, best_gap


def _find_nearest_below(target: dict, boxes: list[dict]) -> tuple[dict | None, float | None]:
    """Nearest box directly below `target` with horizontal overlap."""
    t = _effective_box(target)
    best, best_gap = None, None
    tx0, tx1 = t["x"], t["x"] + t["width"]
    for b in boxes:
        if b is target:
            continue
        bx = _effective_box(b)
        bx0, bx1 = bx["x"], bx["x"] + bx["width"]
        if bx1 <= tx0 or bx0 >= tx1:
            continue
        if bx["y"] >= t["y"] + t["height"]:
            gap = bx["y"] - (t["y"] + t["height"])
            if best_gap is None or gap < best_gap:
                best, best_gap = b, gap
    return best, best_gap


def _find_nearest_left(target: dict, boxes: list[dict]) -> tuple[dict | None, float | None]:
    """Nearest box directly to the left of `target` with vertical overlap —
    for side-by-side rows (button rows, image grids) where the horizontal
    gap is the one that actually matters, not the vertical one."""
    t = _effective_box(target)
    best, best_gap = None, None
    ty0, ty1 = t["y"], t["y"] + t["height"]
    for b in boxes:
        if b is target:
            continue
        bx = _effective_box(b)
        by0, by1 = bx["y"], bx["y"] + bx["height"]
        if by1 <= ty0 or by0 >= ty1:
            continue
        right = bx["x"] + bx["width"]
        if right <= t["x"]:
            gap = t["x"] - right
            if best_gap is None or gap < best_gap:
                best, best_gap = b, gap
    return best, best_gap


def _find_nearest_right(target: dict, boxes: list[dict]) -> tuple[dict | None, float | None]:
    """Nearest box directly to the right of `target` with vertical overlap."""
    t = _effective_box(target)
    best, best_gap = None, None
    ty0, ty1 = t["y"], t["y"] + t["height"]
    for b in boxes:
        if b is target:
            continue
        bx = _effective_box(b)
        by0, by1 = bx["y"], bx["y"] + bx["height"]
        if by1 <= ty0 or by0 >= ty1:
            continue
        if bx["x"] >= t["x"] + t["width"]:
            gap = bx["x"] - (t["x"] + t["width"])
            if best_gap is None or gap < best_gap:
                best, best_gap = b, gap
    return best, best_gap


# Left/right spacing of a line of running text (a paragraph, a list item) is
# driven by the text container's own width, which legitimately differs across
# viewport widths and reflows — that's not a design defect, just responsive
# layout doing its job. Structural left/right gaps (between two image/button/
# card columns, or a genuinely displaced heading) are real defects and should
# stay flagged at any magnitude. Rather than suppress body-text side-spacing
# outright, only suppress it below this anomaly threshold — a gap large enough
# to suggest an actual layout problem (a wildly indented heading, a big unex-
# plained void) still gets reported. Chosen from real report data: normal
# per-viewport text-reflow drift observed in practice tops out well under
# 100px; genuine layout anomalies observed ran 250px+.
_SIDE_TEXT_ANOMALY_PX = 150


def compare_spacing(
    figma_boxes: list[dict],
    live_boxes: list[dict],
    matched_targets: list[tuple[dict, dict]],
    label_field: str = "label",
    suppress_minor_side_gaps: bool = False,
) -> list[dict]:
    """
    For each matched (figma_target, live_target) pair, measure the rendered
    gap above/below AND left/right against the nearest neighboring element on
    each side, and flag when a gap differs meaningfully between design and
    live. Checking both axes matters because "spacing between two elements"
    is a horizontal gap for a side-by-side row (a button row, an image grid)
    and a vertical gap for stacked content (title above subtitle) — the same
    two elements can be neighbors on either axis depending on layout.

    figma_boxes / live_boxes: the FULL set of positioned boxes on each side
    (text nodes + images + buttons + sections combined) used as neighbors for
    gap measurement — not just the matched targets themselves, since the
    nearest neighbor to a button is very often a text line or image, not
    another button.

    suppress_minor_side_gaps: set for running-text targets (paragraphs, list
    items) — left/right gaps below _SIDE_TEXT_ANOMALY_PX are dropped since
    they're just responsive text reflow, not a real mismatch. Vertical (above/
    below) gaps are never affected by this — stacked-content spacing is a real
    defect at any size regardless of element type.
    """
    issues: list[dict] = []

    for ftarget, ltarget in matched_targets:
        f_above, f_gap_above = _find_nearest_above(ftarget, figma_boxes)
        l_above, l_gap_above = _find_nearest_above(ltarget, live_boxes)
        f_below, f_gap_below = _find_nearest_below(ftarget, figma_boxes)
        l_below, l_gap_below = _find_nearest_below(ltarget, live_boxes)
        f_left, f_gap_left = _find_nearest_left(ftarget, figma_boxes)
        l_left, l_gap_left = _find_nearest_left(ltarget, live_boxes)
        f_right, f_gap_right = _find_nearest_right(ftarget, figma_boxes)
        l_right, l_gap_right = _find_nearest_right(ltarget, live_boxes)

        label = ftarget.get(label_field) or ltarget.get(label_field) or "element"

        for direction, f_gap, l_gap, f_neighbor, l_neighbor in (
            ("above", f_gap_above, l_gap_above, f_above, l_above),
            ("below", f_gap_below, l_gap_below, f_below, l_below),
            ("left of", f_gap_left, l_gap_left, f_left, l_left),
            ("right of", f_gap_right, l_gap_right, f_right, l_right),
        ):
            if f_gap is None or l_gap is None:
                continue
            diff = abs(f_gap - l_gap)
            # Flat 2px absolute floor — a gap off by 2px or less is normal
            # rendering noise regardless of how big the gap itself is.
            if diff <= _SPACING_MIN_ABS_PX:
                continue
            if suppress_minor_side_gaps and direction in ("left of", "right of") \
                    and diff < _SIDE_TEXT_ANOMALY_PX:
                continue
            pct = (diff / f_gap * 100) if f_gap > 1 else 100.0

            # Crop region spans target + neighbor (+ the gap between them) on
            # each side, rather than just the target's own tight box — a
            # crop that doesn't show the whitespace being complained about
            # gives a human reviewer nothing to actually verify the claim
            # against. Falls back to the target's own box only if a neighbor
            # is somehow missing (shouldn't happen given f_gap/l_gap above
            # are already non-None, but stay defensive).
            live_region = _union_box(_effective_box(ltarget), _effective_box(l_neighbor)) \
                if l_neighbor is not None else ltarget
            figma_region = _union_box(_effective_box(ftarget), _effective_box(f_neighbor)) \
                if f_neighbor is not None else ftarget

            issues.append({
                "element": f"spacing {direction} {label}"[:80],
                "description": (
                    f'Spacing {direction} "{label}" is {l_gap:.0f}px on the live site '
                    f'vs {f_gap:.0f}px in Figma ({diff:.0f}px off).'
                ),
                "user_impact": "Inconsistent spacing disrupts visual rhythm and can make the layout feel unpolished.",
                "suggested_fix": f"Adjust the {direction} spacing to {f_gap:.0f}px to match the Figma spec.",
                "issue_type": "spacing",
                "x": live_region.get("x", 0), "y": live_region.get("y", 0),
                "width": live_region.get("width", 0), "height": live_region.get("height", 0),
                # The matched Figma element's own (widened) box — see
                # compare_dimensions above / typography_diff.compare_typography
                # for why this must stay separate from the live box
                # (Figma-side crop correctness).
                "figma_x": figma_region.get("x", 0), "figma_y": figma_region.get("y", 0),
                "figma_width": figma_region.get("width", 0), "figma_height": figma_region.get("height", 0),
                # Structured numeric fields (as opposed to only baking these
                # into the free-text description) — intended for
                # bug_report_generator to eventually recognize "same
                # direction, same before/after gap, different element" as
                # one systemic spacing rule hitting several instances,
                # instead of N unrelated bugs that happen to read similarly.
                # NOTE: that consumer (_same_spacing_pattern) is currently
                # disabled — see its docstring — because a naive version
                # over-grouped on a real page. These fields are harmless to
                # keep populated as the foundation for a safer future
                # attempt (e.g. requiring an exact direction match plus a
                # real shared-cause signal, not magnitude coincidence alone).
                "gap_direction": direction,
                "figma_gap_px": round(f_gap, 1),
                "live_gap_px": round(l_gap, 1),
                "diff_percent": min(100.0, pct),
                # Severity magnitude scales with the real pixel difference, not
                # a percentage — a 3px miss just past the tolerance floor should
                # rank low; a 25px+ miss should rank meaningfully higher,
                # regardless of how big the underlying gap is.
                "magnitude_score": min(30.0, diff),
            })

    return issues


def _labeled(items: list[dict], label_key: str) -> list[dict]:
    """Ensure every item carries a 'label' key, derived from label_key when
    missing — sections don't have one by default (unlike text/image/button
    nodes, which already do)."""
    out = []
    for it in items or []:
        if it.get("label"):
            out.append(it)
        else:
            out.append({**it, "label": it.get(label_key, "")})
    return out


def compare_all_spacing(
    figma_text_nodes: list[dict],
    live_text_nodes: list[dict],
    figma_image_nodes: list[dict],
    live_image_elements: list[dict],
    figma_button_nodes: list[dict],
    live_button_elements: list[dict],
    figma_sections: list[dict] | None = None,
    live_sections: list[dict] | None = None,
    neighbor_figma_boxes: list[dict] | None = None,
    neighbor_live_boxes: list[dict] | None = None,
) -> list[dict]:
    """
    Run spacing comparison across every element category in one pass:
    section-to-section, text-to-text (covers title/subtitle gaps and any
    other adjacent text lines), image-to-image, and button spacing — all
    measured against one shared neighbor pool per side, so "what's the
    nearest thing above/below" isn't artificially restricted to elements of
    the same type (the nearest thing below a heading is very often an image
    or a button, not another heading).

    neighbor_figma_boxes / neighbor_live_boxes: the neighbor pool used for
    gap measurement, if it needs to be WIDER than the target elements being
    compared — e.g. a caller scoping targets to one section of a page must
    still supply the FULL page's boxes here, or an element sitting near that
    section's own top/bottom edge will have its true nearest neighbor (which
    may be a paragraph or heading just outside the section boundary) excluded
    from the search entirely. That silently falls through to a much farther,
    wrong "nearest" pick and reports a wildly inflated gap for an element that
    isn't actually misplaced — confirmed against real reports: a target whose
    genuine 10px neighbor sat just past its section's clipped boundary came
    back as "72px off" because the search skipped straight past it. When
    omitted, falls back to the target elements themselves (correct only when
    the caller's inputs already cover every possible neighbor, e.g. a
    whole-page run with no section clipping).
    """
    figma_sections = _labeled(figma_sections, "name")
    live_sections = _labeled(live_sections, "heading")

    all_figma_boxes = neighbor_figma_boxes if neighbor_figma_boxes is not None \
        else figma_text_nodes + figma_image_nodes + figma_button_nodes + figma_sections
    all_live_boxes = neighbor_live_boxes if neighbor_live_boxes is not None \
        else live_text_nodes + live_image_elements + live_button_elements + live_sections

    matched_text = [(f, l) for f, l, _ in match_text_nodes(figma_text_nodes, live_text_nodes)]
    matched_images = match_elements(figma_image_nodes, live_image_elements, "label")
    matched_buttons = match_elements(figma_button_nodes, live_button_elements, "label")
    matched_sections = (
        [(f, l) for f, l, _ in match_sections(figma_sections, live_sections)]
        if figma_sections and live_sections else []
    )

    issues: list[dict] = []
    # Text targets get the side-gap anomaly filter — running-text left/right
    # position reflows with viewport width and isn't a real defect at normal
    # magnitudes. Images, buttons, and sections are structural (card/column
    # layouts, button rows) — their left/right gaps stay fully checked, since
    # that's exactly the "spacing between two columns" the anomaly filter must
    # never suppress.
    issues += compare_spacing(
        all_figma_boxes, all_live_boxes, matched_text, label_field="text",
        suppress_minor_side_gaps=True,
    )
    issues += compare_spacing(all_figma_boxes, all_live_boxes, matched_images, label_field="label")
    issues += compare_spacing(all_figma_boxes, all_live_boxes, matched_buttons, label_field="label")
    issues += compare_spacing(all_figma_boxes, all_live_boxes, matched_sections, label_field="label")
    return issues
