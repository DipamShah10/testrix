"""
Section matcher — pairs Figma section crops with live DOM section screenshots
using semantic type detection and text similarity.

Scoring breakdown (max 100 pts):
  40 pts — same semantic type (hero, nav, product, footer, …)
  40 pts — text/label similarity between Figma section name and DOM section text
  20 pts — positional order bonus (sections appear in similar vertical order)
"""
import logging
import re
from difflib import SequenceMatcher

import numpy as np
from scipy.optimize import linear_sum_assignment

from visual.section_alignment_engine import _detect_type  # noqa: F401 — re-exported for callers

logger = logging.getLogger(__name__)

# Filtered out of word-overlap similarity — common enough in both a short
# Figma layer name and any live heading/copy that they'd inflate the overlap
# score without actually indicating the two sections are the same content.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "our", "your", "you", "we", "us", "at", "by", "from",
}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 1}


def _word_overlap(a: str, b: str) -> float:
    """Jaccard similarity over meaningful words — robust to length/order
    differences that sink SequenceMatcher (e.g. a Figma layer named "Best
    Sellers" against a live heading "SHOP OUR BEST SELLERS" shares both
    meaningful words but scores poorly on raw character-sequence overlap)."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _word_containment(a: str, b: str) -> float:
    """
    Overlap coefficient (intersection / smaller set) rather than Jaccard.

    A short Figma layer name ("Best Sellers") against a live section's full
    label — heading plus a chunk of its body text/prices/product names — is
    an inherently asymmetric comparison: Jaccard's union denominator grows
    with the long side's unrelated words and drags the score down even when
    every word of the short side is genuinely present. Containment instead
    asks "how much of the shorter side shows up in the longer one," which is
    the question that actually matters here.

    Confidence-discounted by the smaller side's word count: a 1-2 word
    overlap being 100% of the smaller side's words is weak evidence of "same
    entity", not strong evidence — e.g. "Six Generations Of Master Cutlers"
    vs. an unrelated "Six Generations On" reduces, after stopword-stripping
    "On", to {six, generations} ⊂ {six, generations, master, cutlers}, a
    perfect *coincidental* containment score that would otherwise beat a
    genuinely near-identical string. Full confidence needs ≥3 shared words.
    """
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    smaller = min(len(wa), len(wb))
    raw = len(wa & wb) / smaller
    confidence = min(1.0, smaller / 3)
    return raw * confidence


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    # Take whichever signal is stronger — character-sequence similarity
    # catches near-identical strings and typos; word overlap/containment
    # catch the same content phrased as a short label vs. a full sentence.
    # Taking the max can only raise a score where one signal has real
    # evidence, never silently lowers it.
    char_sim = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    jaccard = _word_overlap(a, b)
    containment = _word_containment(a, b)
    return max(char_sim, jaccard, containment)


def _figma_type(sec: dict) -> str:
    return _detect_type(sec.get("name", ""))


def _live_type(sec: dict) -> str:
    text = " ".join(filter(None, [
        sec.get("sectionType", ""),
        sec.get("classes", ""),
        sec.get("id", ""),
        sec.get("heading", ""),
        sec.get("textSnippet", ""),
    ]))
    return _detect_type(text)


def _live_label(sec: dict) -> str:
    return " ".join(filter(None, [
        sec.get("heading", ""),
        sec.get("sectionType", ""),
        sec.get("id", ""),
        sec.get("classes", "")[:60],
        sec.get("textSnippet", "")[:120],
    ]))


def _score_pair(figma_sec: dict, live_sec: dict, fi: int, li: int, n_figma: int, n_live: int) -> float:
    score = 0.0

    ft = _figma_type(figma_sec)
    lt = _live_type(live_sec)

    # Semantic type match
    if ft == lt and ft != "unknown":
        score += 40.0
    elif ft != "unknown" and lt != "unknown":
        score += 5.0  # partial credit — both typed but different

    # Text / label similarity
    figma_label = figma_sec.get("name", "")
    live_label  = _live_label(live_sec)
    score += _similarity(figma_label, live_label) * 40.0

    # Positional order: reward pairs that are in roughly the same vertical order
    # Normalize positions to [0, 1] range within their respective lists.
    figma_pos = fi / max(1, n_figma - 1) if n_figma > 1 else 0.5
    live_pos  = li / max(1, n_live  - 1) if n_live  > 1 else 0.5
    order_sim = max(0.0, 1.0 - abs(figma_pos - live_pos) * 2)
    score += order_sim * 20.0

    return score


def match_sections(
    figma_sections: list[dict],
    live_sections: list[dict],
    min_confidence: float = 25.0,
) -> list[tuple[dict, dict, float]]:
    """
    Optimal (Hungarian-algorithm) pairing of Figma sections -> live DOM
    sections — replaces a previous greedy implementation that processed
    Figma sections in order and committed irrevocably on each pick. On a
    page with several near-identical repeating sections (the type/text
    signals in _score_pair degenerate to noise across them), greedy's
    first-come-first-served commitment had no way to reconsider an early
    wrong pick, cascading misalignment down the rest of the list — this is
    the documented "greedy matching can misalign sections on pages with many
    similar-looking containers" limitation. Optimal assignment instead picks
    the pairing that maximizes the SUM of scores across all pairs at once:
    for a row of positionally-ordered repeating sections, the order-
    preserving assignment maximizes the 20-pt positional term far more than
    any crossed assignment does, so it naturally wins even when the 40+40
    type/text terms are tied/degenerate across the row.

    Returns [(figma_sec, live_sec, confidence), ...] where confidence is 0–100.
    Only pairs scoring above min_confidence are returned — Hungarian is
    forced to assign every row when n_figma <= n_live, so this threshold
    filter (identical to the old greedy behavior) is still what prevents a
    genuinely bad pair from surviving into the result.
    """
    if not figma_sections or not live_sections:
        return []

    n_figma = len(figma_sections)
    n_live  = len(live_sections)

    cost = np.empty((n_figma, n_live), dtype=float)
    for fi, fsec in enumerate(figma_sections):
        for li, lsec in enumerate(live_sections):
            cost[fi, li] = -_score_pair(fsec, lsec, fi, li, n_figma, n_live)

    figma_idx, live_idx = linear_sum_assignment(cost)

    pairs: list[tuple[dict, dict, float]] = []
    for fi, li in zip(figma_idx, live_idx):
        score = -cost[fi, li]
        if score >= min_confidence:
            conf = round(score, 1)
            pairs.append((figma_sections[fi], live_sections[li], conf))
            logger.debug(
                f"Section match — Figma '{figma_sections[fi].get('name', '?')}' "
                f"→ live '{_live_label(live_sections[li])[:40]}' "
                f"(conf={conf})"
            )
        else:
            logger.debug(
                f"Section unmatched — Figma '{figma_sections[fi].get('name', '?')}' "
                f"(best_score={score:.1f} < {min_confidence})"
            )

    logger.info(
        f"Section matching — {len(figma_sections)} Figma, {len(live_sections)} live "
        f"→ {len(pairs)} pairs matched"
    )
    return pairs


def unmatched_figma_sections(
    figma_sections: list[dict],
    pairs: list[tuple[dict, dict, float]],
) -> list[dict]:
    """
    Figma sections that match_sections did not pair with any live section
    (below min_confidence, or n_live < n_figma) — a caller can surface these
    as "needs_recheck" (a Figma section that may not exist on the live page,
    or couldn't be confidently located) instead of silently dropping them.
    """
    matched_ids = {id(fsec) for fsec, _, _ in pairs}
    return [fsec for fsec in figma_sections if id(fsec) not in matched_ids]