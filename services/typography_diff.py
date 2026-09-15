"""
Typography diff engine — compares precise per-element style specs from Figma
TEXT nodes against computed CSS from the live DOM, instead of relying on the
vision LLM to guess font-weight/size/color from a screenshot.
"""
import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_WEIGHT_NAMES = {
    100: "Thin", 200: "ExtraLight", 300: "Light", 400: "Regular", 500: "Medium",
    600: "SemiBold", 700: "Bold", 800: "ExtraBold", 900: "Black",
}

_FAMILY_SUFFIX_RE = re.compile(
    # Custom/self-hosted webfonts commonly bake the weight into the family
    # name with a hyphen (e.g. "Quasimoda-Medium", "Raw-Bold") rather than a
    # space — consume that separator too, or it's left dangling after strip().
    r"[\s\-]*(Regular|Bold|Italic|Medium|SemiBold|Light|Black|ExtraBold|ExtraLight|Thin)\s*$",
    re.IGNORECASE,
)

_MIN_TEXT_SIMILARITY = 0.55
_MAX_MATCH_DISTANCE_PX = 120   # positional tolerance when text similarity is imperfect

_COLOR_TOLERANCE = 24          # Euclidean RGB distance below which colors are "the same"
_SIZE_TOLERANCE_PX = 1.5       # px difference below which font-size is "the same"


def _normalize_family(name: str | None) -> str:
    """Strip weight/style suffixes and quoting so 'Inter Bold' == 'Inter'."""
    if not name:
        return ""
    primary = name.split(",")[0].strip()   # drop CSS fallback stack, keep primary face
    primary = primary.strip('"').strip("'").strip()
    primary = _FAMILY_SUFFIX_RE.sub("", primary).strip()
    return primary.lower()


def _weight_label(weight) -> str:
    try:
        w = int(round(float(weight)))
    except (TypeError, ValueError):
        return str(weight)
    name = _WEIGHT_NAMES.get(w, "")
    return f"{w} ({name})" if name else str(w)


def _parse_css_color(color_str: str | None) -> tuple[int, int, int] | None:
    if not color_str:
        return None
    m = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", color_str)
    if not m:
        return None
    return tuple(int(x) for x in m.groups())


def _color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def match_text_nodes(
    figma_nodes: list[dict],
    live_nodes: list[dict],
) -> list[tuple[dict, dict, float]]:
    """
    Pair each Figma TEXT node to its best-matching live text element.

    Matching is primarily by text-content similarity (design and live copy
    should usually read the same, differing only in style), with positional
    proximity as a tiebreaker/guard against matching the wrong repeated string
    (e.g. multiple "Shop Now" buttons on one page).

    Returns a list of (figma_node, live_node, confidence) — confidence is not
    a probability, just a relative score used for greedy assignment.
    """
    if not figma_nodes or not live_nodes:
        return []

    candidates = []
    for fi, fnode in enumerate(figma_nodes):
        if not fnode.get("text"):
            continue
        for li, lnode in enumerate(live_nodes):
            sim = _text_similarity(fnode["text"], lnode.get("text", ""))
            if sim < _MIN_TEXT_SIMILARITY:
                continue
            dist = (
                (fnode.get("x", 0) - lnode.get("x", 0)) ** 2
                + (fnode.get("y", 0) - lnode.get("y", 0)) ** 2
            ) ** 0.5
            # Distant matches are only trusted when the text match is near-exact
            if dist > _MAX_MATCH_DISTANCE_PX and sim < 0.9:
                continue
            score = sim - (dist / 5000)
            candidates.append((score, fi, li))

    candidates.sort(key=lambda c: -c[0])

    used_f: set[int] = set()
    used_l: set[int] = set()
    pairs: list[tuple[dict, dict, float]] = []
    for score, fi, li in candidates:
        if fi in used_f or li in used_l:
            continue
        used_f.add(fi)
        used_l.add(li)
        pairs.append((figma_nodes[fi], live_nodes[li], round(score, 2)))

    return pairs


def _diff_pair(fnode: dict, lnode: dict) -> list[tuple[str, str, str, float]]:
    """
    Return a list of (field, figma_value_label, live_value_label, magnitude) mismatches.

    magnitude is a 0-40 severity-weight for that single mismatch, scaled by how
    far off the live value is — a font-family swapped entirely is a much bigger
    problem than a button one weight-step lighter than spec, and severity should
    reflect that instead of every typography hit landing in the same tier.
    """
    mismatches: list[tuple[str, str, str, float]] = []

    f_family = _normalize_family(fnode.get("font_family"))
    l_family = _normalize_family(lnode.get("font_family"))
    if f_family and l_family and f_family != l_family:
        # A completely different typeface is the most visually disruptive
        # typography defect — flat weight, not proportional to anything.
        mismatches.append(("font family", fnode.get("font_family") or "", lnode.get("font_family") or "", 40.0))

    f_weight, l_weight = fnode.get("font_weight"), lnode.get("font_weight")
    try:
        if f_weight and l_weight and int(round(float(f_weight))) != int(round(float(l_weight))):
            steps = abs(int(round(float(f_weight))) - int(round(float(l_weight)))) / 100.0
            # One weight step (e.g. Regular vs Medium) is a minor, often barely
            # perceptible difference; multiple steps (Regular vs Bold+) is not.
            mismatches.append(("font weight", _weight_label(f_weight), _weight_label(l_weight), min(30.0, steps * 12.0)))
    except (TypeError, ValueError):
        pass

    f_size, l_size = fnode.get("font_size"), lnode.get("font_size")
    try:
        if f_size and l_size and abs(float(f_size) - float(l_size)) > _SIZE_TOLERANCE_PX:
            pct = abs(float(f_size) - float(l_size)) / float(f_size) * 100.0
            mismatches.append(("font size", f"{f_size}px", f"{l_size}px", min(40.0, pct)))
    except (TypeError, ValueError):
        pass

    f_color = fnode.get("color")
    l_color = _parse_css_color(lnode.get("color"))
    if f_color and l_color:
        dist = _color_distance(f_color, l_color)
        if dist > _COLOR_TOLERANCE:
            # Max possible RGB distance is ~441 (black vs white) — scale to 0-20.
            mismatches.append(("color", f"rgb{tuple(f_color)}", f"rgb{l_color}", min(20.0, dist / 441.0 * 20.0)))

    return mismatches


def compare_typography(
    figma_nodes: list[dict],
    live_nodes: list[dict],
) -> list[dict]:
    """
    Compare matched Figma text specs against live computed styles.

    Both node lists must already be in the same logical coordinate space
    (the page's canonical width) — figma_nodes are frame-relative by
    construction; live_nodes must be normalized via normalize_section_coords
    before calling this.

    Returns a list of issue dicts (issue_type='typography') shaped to slot
    directly into the same pipeline as vision-based issues (severity
    classification, fix recommendations, report building).
    """
    pairs = match_text_nodes(figma_nodes, live_nodes)
    issues: list[dict] = []

    for fnode, lnode, confidence in pairs:
        mismatches = _diff_pair(fnode, lnode)
        if not mismatches:
            continue

        desc = "; ".join(
            f"{field} is {live_val} on the live site vs {figma_val} in Figma"
            for field, figma_val, live_val, _ in mismatches
        )
        text_snippet = fnode.get("text", "")[:60]
        element_label = fnode.get("name") or text_snippet or "text element"
        magnitude_score = sum(m[3] for m in mismatches)

        issues.append({
            "element": element_label,
            "description": f'"{text_snippet}" — {desc}.',
            "user_impact": (
                "Text does not match the approved Figma spec, creating visual "
                "inconsistency and undermining brand consistency."
            ),
            "suggested_fix": "; ".join(
                f"set {field.replace(' ', '-')} to {figma_val}"
                for field, figma_val, _, _ in mismatches
            ),
            "issue_type": "typography",
            "x": lnode.get("x", 0),
            "y": lnode.get("y", 0),
            "width": lnode.get("width", 0),
            "height": lnode.get("height", 0),
            # Not a pixel-diff percentage — a severity signal proportional to
            # how many style properties disagree, so severity_classifier's
            # diff_percent-based rules route these to at least "Medium".
            "diff_percent": min(100.0, 20.0 * len(mismatches)),
            "match_confidence": confidence,
            "typography_mismatches": [
                {"field": f, "figma": fv, "live": lv} for f, fv, lv, _ in mismatches
            ],
            # Aggregate magnitude across all mismatched fields on this element —
            # consumed directly by severity_classifier instead of guessing
            # severity from element-name keywords or an LLM call.
            "magnitude_score": round(magnitude_score, 1),
        })

    if issues:
        logger.info(f"Typography diff — {len(pairs)} matched pairs, {len(issues)} mismatch(es)")
    return issues
