"""
Section alignment engine — matches DOM sections from the live Shopify page to
Figma frames, and annotates diff regions with the DOM section they belong to.
"""
import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Semantic keywords for typed matching.
#
# Order matters: _detect_type returns the FIRST type whose keywords match, so
# structural/positional types (nav, footer, announcement — a section's own
# role, usually signaled by its id/class naming) are listed before
# content-topic types (hero, product, features, testimonial, faq, newsletter
# — signaled by loose body text/keywords). A real-world footer very commonly
# bundles a "Subscribe" newsletter box alongside its link lists; without this
# ordering, "subscribe" (newsletter) would win over the footer's own "footer"
# class/id, and an exclude_sections=["footer"] request would silently miss it
# — exactly the bug this ordering fixes.
_SECTION_TYPES = {
    "nav":          ["nav", "navigation", "header", "menu", "topbar"],
    "footer":       ["footer", "bottom", "copyright", "links", "sitemap"],
    "announcement": ["announcement", "bar", "notice", "promo"],
    "hero":         ["hero", "banner", "slideshow", "slide", "masthead", "jumbotron"],
    "product":      ["product", "pdp", "item", "detail", "buy", "add-to-cart"],
    "collection":   ["collection", "plp", "catalog", "shop", "grid", "listing"],
    "features":     ["feature", "benefit", "why", "about", "highlight", "usp"],
    "testimonial":  ["testimonial", "review", "rating", "customer", "trust", "quote"],
    "faq":          ["faq", "question", "answer", "accordion", "help"],
    "newsletter":   ["newsletter", "subscribe", "email", "signup"],
}


def _normalize_for_type_match(text: str) -> str:
    """Treat '-' and '_' as word separators, e.g. Shopify's own
    "shopify-section" or "template--123__footer" ID convention — otherwise
    regex \\b doesn't see a boundary next to '_' (it's a \\w character)."""
    return re.sub(r"[-_]+", " ", text.lower())


def _detect_type(text: str) -> str:
    """
    Classify a blob of text (class names, heading, id) into a semantic type.

    Matches whole words/phrases only (word-boundary regex on both sides,
    hyphen/underscore-normalized), not raw substrings — every Shopify theme
    wraps sections in the boilerplate class "shopify-section", and a naive
    `"shop" in text` check matches inside "shopify" on literally every section
    of every Shopify store, making the "collection" type fire almost
    unconditionally and drowning out real classification.
    """
    t = _normalize_for_type_match(text)
    for type_name, keywords in _SECTION_TYPES.items():
        if any(re.search(rf"\b{re.escape(_normalize_for_type_match(kw))}\b", t) for kw in keywords):
            return type_name
    return "unknown"


def _text_similarity(a: str, b: str) -> float:
    """0–1 similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _section_label(section: dict) -> str:
    """Concatenate all descriptive text for a DOM section."""
    return " ".join(filter(None, [
        section.get("heading", ""),
        section.get("classes", ""),
        section.get("id", ""),
        section.get("sectionType", ""),
        section.get("textSnippet", "")[:60],
    ]))


def _frame_label(frame: dict) -> str:
    return frame.get("name", "")


def _score(dom_sec: dict, figma_frame: dict) -> float:
    """Score compatibility between a DOM section and a Figma frame (higher = better)."""
    score = 0.0

    sec_label   = _section_label(dom_sec)
    frame_label = _frame_label(figma_frame)
    sec_type    = _detect_type(sec_label)
    frame_type  = _detect_type(frame_label)

    # Strong match: same semantic type
    if sec_type == frame_type and sec_type != "unknown":
        score += 10.0

    # Text similarity between frame name and section label
    score += _text_similarity(frame_label, sec_label) * 8.0

    # Word overlap: every word in frame name that appears in section text
    frame_words = [w for w in frame_label.lower().split() if len(w) > 3]
    sec_text = sec_label.lower()
    for word in frame_words:
        if word in sec_text:
            score += 3.0

    # Heading match with frame name
    heading = dom_sec.get("heading", "").lower()
    if heading and any(w in heading for w in frame_label.lower().split() if len(w) > 3):
        score += 4.0

    return score


def match_sections_to_frames(
    dom_sections: list[dict],
    figma_frames: list[dict],
) -> list[tuple[dict, dict]]:
    """
    Greedy best-match alignment of DOM sections → Figma frames.

    Returns a list of (dom_section, figma_frame) pairs in DOM order.
    Each Figma frame is used at most once. Falls back to positional order
    when no semantic match is found.
    """
    if not dom_sections or not figma_frames:
        return []

    used_frames: set[int] = set()
    pairs: list[tuple[dict, dict]] = []

    for sec in dom_sections:
        best_idx   = -1
        best_score = -1.0

        for fi, frame in enumerate(figma_frames):
            if fi in used_frames:
                continue
            sc = _score(sec, frame)
            if sc > best_score:
                best_score = sc
                best_idx   = fi

        if best_idx >= 0:
            used_frames.add(best_idx)
            pairs.append((sec, figma_frames[best_idx]))
            logger.debug(
                f"Section match — DOM '{_section_label(sec)[:40]}' "
                f"→ Figma '{figma_frames[best_idx]['name']}' (score={best_score:.1f})"
            )

    # If sections outnumber frames, assign remaining sections to last frame
    assigned_secs = {id(p[0]) for p in pairs}
    for sec in dom_sections:
        if id(sec) not in assigned_secs and figma_frames:
            pairs.append((sec, figma_frames[-1]))

    return pairs


def find_section_for_region(
    region_y: int,
    region_height: int,
    dom_sections: list[dict],
) -> dict | None:
    """
    Return the DOM section that most overlaps with a diff region.
    Coordinates are in canonical (1440px) normalized space, same as DiffRegion.
    DOM section coordinates are also in 1440px space after normalization.
    """
    if not dom_sections:
        return None

    region_end = region_y + region_height
    best: dict | None = None
    best_overlap = 0

    for sec in dom_sections:
        sec_y   = sec.get("y", 0)
        sec_end = sec_y + sec.get("height", 0)

        overlap = max(0, min(region_end, sec_end) - max(region_y, sec_y))
        if overlap > best_overlap:
            best_overlap = overlap
            best = sec

    return best


def normalize_section_coords(
    sections: list[dict],
    original_width: int,
    canonical_width: int = 1440,
) -> list[dict]:
    """
    Scale section x/y/width/height from the captured screenshot width to canonical width.
    Must be called AFTER full-page capture so coordinates match the comparison space.
    """
    if original_width == canonical_width or original_width == 0:
        return sections

    scale = canonical_width / original_width
    normalized = []
    for sec in sections:
        entry = {
            **sec,
            "x":      int(sec.get("x", 0)      * scale),
            "y":      int(sec.get("y", 0)      * scale),
            "width":  int(sec.get("width", 0)  * scale),
            "height": int(sec.get("height", 0) * scale),
        }
        # spacing_box (the padding-aware container box a short/icon-like text
        # node carries — see shopify_scraper._EXTRACT_TEXT_NODES_JS) is in the
        # same raw capture-width coordinate space as the node's own x/y/width/
        # height and must be rescaled identically, or it silently ends up in
        # the wrong coordinate space relative to everything else once this
        # scale factor isn't 1:1.
        box = sec.get("spacing_box")
        if box:
            entry["spacing_box"] = {
                "x":      int(box.get("x", 0)      * scale),
                "y":      int(box.get("y", 0)      * scale),
                "width":  int(box.get("width", 0)  * scale),
                "height": int(box.get("height", 0) * scale),
            }
        normalized.append(entry)
    return normalized