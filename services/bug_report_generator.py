import base64
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher

from services.db import update_vqa_job
from services.visual_comparator import CompareResult

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
_GROUP_SIMILARITY_THRESHOLD = 0.8


def _b64(data: bytes | None) -> str | None:
    return base64.b64encode(data).decode() if data else None


_GAP_GROUP_TOLERANCE_PX = 1.5  # figma_gap_px/live_gap_px within this are "the same gap"

# EXPERIMENT, TRIED AND REVERTED: matching spacing issues purely by numeric
# gap similarity (regardless of direction) looked correct on a small,
# hand-picked real example (three pill-row elements all reporting "34px vs
# 10px"), but broke badly on a full real page run — it collapsed 39 issues
# spanning all 5 page sections, including a horizontal "left of" gap grouped
# with vertical "above"/"below" gaps, into one report entry. That's not one
# systemic CSS rule; it's the tolerance being far too loose once a real page
# has many independently-occurring gaps of similar magnitude — and grouping
# is a one-way door: 38 potentially-real, distinct defects would be hidden
# behind a single entry with no way for a reviewer to recover them. Hiding
# real bugs is a worse failure than the 3-duplicate noise this was meant to
# fix, so this is reverted rather than shipped without being able to fully
# verify its safety. geometry_diff.py still emits gap_direction/figma_gap_px/
# live_gap_px on every spacing issue — a future attempt should require an
# exact direction match AND probably tag *why* two gaps are suspected to
# share a cause (e.g. same neighbor element, same CSS class) rather than
# coincidence of measured magnitude alone.
def _same_spacing_pattern(a: dict, b: dict) -> bool:
    return False


def _group_similar_issues(issues: list[dict]) -> list[dict]:
    """
    Collapse issues that are the same underlying defect hitting multiple
    instances (e.g. the same "Shop Now" font-weight mismatch on 3 separate
    product cards, or 4 copies of the vision model's identical "different
    product image" observation) into one representative issue carrying an
    occurrence count — instead of listing each instance as its own bug.

    Grouped within the same issue_type by description text similarity (not
    exact match — vision-generated descriptions vary in wording even when
    describing the same underlying observation). Also checks
    _same_spacing_pattern first, which is currently disabled (always False,
    see its docstring for why) — kept as the extension point for a future,
    safer numeric-gap-matching attempt.
    """
    groups: list[dict] = []  # each: {"rep": issue, "occurrences": [issue, ...]}

    for issue in issues:
        desc = (issue.get("description") or "").lower()
        itype = issue.get("issue_type")
        match = None
        for g in groups:
            if g["rep"].get("issue_type") != itype:
                continue
            if _same_spacing_pattern(issue, g["rep"]):
                match = g
                break
            sim = SequenceMatcher(None, desc, (g["rep"].get("description") or "").lower()).ratio()
            if sim >= _GROUP_SIMILARITY_THRESHOLD:
                match = g
                break
        if match:
            match["occurrences"].append(issue)
        else:
            groups.append({"rep": issue, "occurrences": [issue]})

    grouped: list[dict] = []
    for g in groups:
        occurrences = g["occurrences"]
        rep = dict(g["rep"])
        if len(occurrences) > 1:
            rep["occurrence_count"] = len(occurrences)
            rep["affected_elements"] = list(dict.fromkeys(
                o.get("element", "Unknown element") for o in occurrences
            ))
            # Worst severity among the duplicates wins — one instance being
            # slightly more off than its siblings shouldn't get buried.
            rep["severity"] = min(
                (o.get("severity", "Low") for o in occurrences),
                key=lambda s: _SEVERITY_ORDER.get(s, 99),
            )
        grouped.append(rep)

    if len(grouped) < len(issues):
        logger.info(f"Grouped {len(issues)} issue(s) into {len(grouped)} distinct finding(s)")
    return grouped


def build_page_report(
    page_name: str,
    shopify_url: str,
    figma_frame: dict,
    live_screenshot: bytes,
    compare_result: CompareResult,
    issues: list[dict],
) -> dict:
    """
    Build a structured report dict for one page comparison.
    Images are stored as base64 strings so they can be embedded in the dashboard.
    """
    # capture_failure / expected_variance are not design defects — they mean
    # "this region couldn't be reliably compared" (blank capture, rotating
    # carousel/UGC content). Counting them alongside real defects would let a
    # handful of unrenderable regions drag a clean page down to "Critical" and
    # bury the signal QA/dev teams actually need to act on. needs_recheck is
    # the same idea for a different reason: the Figma<->live PAIRING itself
    # was low-confidence (vision-LLM content gate, or an unmatched section
    # from the optimal matcher) — not confirmed as either a real defect or a
    # non-issue, so it goes to the same "needs manual review" bucket rather
    # than being reported as a confident finding either way.
    _NON_DEFECT_TYPES = {"capture_failure", "expected_variance", "needs_recheck"}
    defect_issues = _group_similar_issues([i for i in issues if i.get("issue_type") not in _NON_DEFECT_TYPES])
    non_defect_issues = _group_similar_issues([i for i in issues if i.get("issue_type") in _NON_DEFECT_TYPES])

    severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for issue in defect_issues:
        sev = issue.get("severity", "Low")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # Overall page health: worst severity found among real defects only
    if severity_counts["Critical"] > 0:
        overall = "Critical"
    elif severity_counts["High"] > 0:
        overall = "High"
    elif severity_counts["Medium"] > 0:
        overall = "Medium"
    elif severity_counts["Low"] > 0:
        overall = "Low"
    else:
        overall = "Pass"

    return {
        "page": page_name,
        "url": shopify_url,
        "figma_frame": figma_frame.get("name", "Unknown"),
        "overall_severity": overall,
        "diff_percent": compare_result.diff_percent,
        "issue_count": len(defect_issues),
        "severity_counts": severity_counts,
        "issues": defect_issues,
        "needs_recheck_count": len(non_defect_issues),
        "needs_recheck_issues": non_defect_issues,
        "figma_image_b64": _b64(figma_frame.get("image_bytes")),
        "live_image_b64": _b64(live_screenshot),
        "diff_image_b64": _b64(compare_result.diff_image),
        "diff_mask_b64": _b64(compare_result.diff_mask),
        "compared_at": datetime.now(timezone.utc).isoformat(),
    }


def build_full_report(
    job_id: str,
    shopify_url: str,
    figma_url: str,
    page_reports: list[dict],
) -> dict:
    """
    Combine all page reports into a top-level report and persist to MongoDB.
    Never stores the raw FIGMA_API_TOKEN — only the URL/file key.
    """
    total_issues = sum(r["issue_count"] for r in page_reports)
    total_needs_recheck = sum(r.get("needs_recheck_count", 0) for r in page_reports)
    all_severities = [r["overall_severity"] for r in page_reports if r["overall_severity"] != "Pass"]

    severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Pass": 4}
    overall = min(all_severities, key=lambda s: severity_order.get(s, 99)) if all_severities else "Pass"

    # Aggregate severity counts across all pages
    total_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for r in page_reports:
        for sev, count in r.get("severity_counts", {}).items():
            total_counts[sev] = total_counts.get(sev, 0) + count

    report = {
        "job_id": job_id,
        "shopify_url": shopify_url,
        "figma_url": figma_url,
        "overall_severity": overall,
        "total_issues": total_issues,
        "needs_recheck_count": total_needs_recheck,
        "severity_counts": total_counts,
        "pages_tested": len(page_reports),
        "pages_passed": sum(1 for r in page_reports if r["overall_severity"] == "Pass"),
        "page_reports": page_reports,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Persist result to MongoDB job record
    update_vqa_job(
        job_id,
        status="complete",
        progress="Done",
        result=_strip_heavy_images(report),
    )
    logger.info(
        f"Report saved — job={job_id}, overall={overall}, "
        f"issues={total_issues}, pages={len(page_reports)}"
    )

    return report


def _strip_heavy_images(report: dict) -> dict:
    """
    Return a copy of the report with per-page images removed from MongoDB storage.
    The full report (with images) is returned to the caller; MongoDB only stores metadata + issues.
    Images are large (~500KB each) — storing them all in MongoDB is fine for MVP
    but we strip from the DB copy to keep documents manageable.
    """
    import copy
    light = copy.deepcopy(report)
    for page in light.get("page_reports", []):
        page.pop("figma_image_b64", None)
        page.pop("live_image_b64", None)
        page.pop("diff_image_b64", None)
        page.pop("diff_mask_b64", None)
        for issue in page.get("issues", []) + page.get("needs_recheck_issues", []):
            issue.pop("expected_crop_b64", None)
            issue.pop("actual_crop_b64", None)
            issue.pop("diff_crop_b64", None)
    return light