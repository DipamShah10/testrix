import base64
import io
import json
import logging
import os
import re
from pathlib import Path

from openai import OpenAI
from PIL import Image

from services.visual_comparator import CompareResult, DiffRegion

logger = logging.getLogger(__name__)

# Matches the vision model admitting, in its own words, that it found nothing
# concrete ("no specific/clear/notable/material/significant ... difference").
# These are non-findings dressed up as issue cards — pure noise in a report
# meant to surface real, actionable defects — and are dropped rather than
# surfaced.
_EMPTY_FINDING_RE = re.compile(
    r"no\s+(?:specific|clear|notable|material|significant)\b[^.]{0,60}\bdifferen",
    re.IGNORECASE,
)


def _is_empty_finding(description: str) -> bool:
    return bool(_EMPTY_FINDING_RE.search(description or ""))


# Groq deprecated meta-llama/llama-4-scout-17b-16e-instruct (2026-06-17) with no
# direct vision-capable replacement of its own. qwen/qwen3.6-27b was the vision
# model at that time; Groq has since moved this org's account to qwen3.8-27b
# (qwen3.6-27b now 404s — confirmed via GET /v1/models on 2026-09-15). Max 3
# images/request, 20MB/image — well within what this module sends (2 images/call).
_VISION_MODEL = "qwen/qwen3.8-27b"
# 1568 pushed a full-page comparison to ~8310 tokens, over this org's 8000 TPM
# cap on the on_demand tier. Image token cost barely moved at 1280 (vision
# models tokenize in fixed patch grids with a large base cost), so dropping
# much further to leave real headroom.
_MAX_IMAGE_DIM = 700
_CANONICAL_WIDTH = 1440   # fallback logical width when a frame's own width is unavailable
_CROP_PADDING = 30        # px of context around each diff region

# analyze_section_pairs has no way to verify the two crops it hands the
# vision model actually correspond to the same design element — the greedy
# section matcher can pair two genuinely different sections/cards (worst on
# pages with several near-identical repeating containers), and the LLM will
# still confidently describe a "typography mismatch" between two unrelated
# things if asked to. sr.match_confidence is match_sections's own pairing
# score for this pair — below this floor, the pairing itself is suspect
# enough that findings get downgraded to needs_recheck (still generated and
# shown for manual review, just not reported as a confident defect) rather
# than silently trusted. Distinct from match_sections's own min_confidence=25
# "matched at all" floor — this is a stricter "trust the interpretation" gate.
_MIN_CONTENT_GATE_CONFIDENCE = 50.0
_SYSTEM_PROMPT = (
    "You are a senior QA engineer specializing in visual regression testing. "
    "You compare Figma design mockups against live Shopify storefronts and identify "
    "UI/UX discrepancies with precision. Be specific: name the element, describe what "
    "differs, and explain the user impact.\n\n"
    "Before describing a defect, rule out these two non-defect explanations — they are "
    "common and must NOT be reported as layout/typography/color bugs:\n"
    "1. CAPTURE FAILURE — one image (usually the live screenshot) shows a blank area, a "
    "solid black/gray box, a loading spinner, or a placeholder where the other image shows "
    "real content. This means the region simply failed to render in time for the "
    "screenshot (e.g. a video or lazy-loaded asset) — it is NOT a spacing, font, or color "
    "difference. Report it with issue_type \"capture_failure\" and description stating the "
    "region appears blank/unrendered, with no invented specifics (no font sizes, no hex "
    "colors, no pixel measurements).\n"
    "2. EXPECTED CONTENT VARIANCE — the region is a carousel, slideshow, or user-generated-"
    "content grid (e.g. customer photos, rotating product slides, testimonials) where the "
    "layout/structure matches but the specific images or slide shown differ because the "
    "live site is dynamic and the Figma mockup is a static snapshot of one arbitrary state. "
    "If the container/layout/styling matches and only the specific rotating content differs, "
    "this is NOT a defect — report issue_type \"expected_variance\" (or omit entirely if you "
    "are highly confident it's benign) rather than inventing a font/color/spacing complaint "
    "about it.\n\n"
    "Never state a specific font weight, exact hex/rgb color, or exact pixel size unless you "
    "can actually distinguish it by eye in the images — if you are guessing, say so in "
    "qualitative terms (e.g. \"appears bolder\") instead of fabricating precise numbers."
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
        _client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    return _client


def _encode_image(img_bytes: bytes, max_dim: int = _MAX_IMAGE_DIM) -> str:
    """Resize image and return base64 data-URL string."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if img.width > max_dim or img.height > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _crop_region(
    img_bytes: bytes, x: int, y: int, w: int, h: int,
    canonical_width: int = _CANONICAL_WIDTH,
) -> str:
    """
    Normalize image to canonical width, crop the diff region with padding,
    and return a plain base64 JPEG string (no data-URL prefix).
    Coordinates are in normalized (canonical_width px) space.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    scale = canonical_width / img.width
    new_h = int(img.height * scale)
    img = img.resize((canonical_width, new_h), Image.LANCZOS)

    x1 = max(0, x - _CROP_PADDING)
    y1 = max(0, y - _CROP_PADDING)
    x2 = min(img.width,  x + max(w, 1) + _CROP_PADDING)
    y2 = min(img.height, y + max(h, 1) + _CROP_PADDING)
    # Ensure valid box (can happen when region coords exceed normalised image size)
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)

    cropped = img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _save_crop(b64_str: str, out_dir: Path, filename: str) -> str:
    """Decode base64 JPEG and write to disk. Returns the file path as string."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_bytes(base64.b64decode(b64_str))
    return str(path)


def build_issue_crops(
    figma_bytes: bytes,
    live_bytes: bytes,
    x: int, y: int, w: int, h: int,
    canonical_width: int,
    job_id: str,
    filename: str,
    figma_x: int | None = None,
    figma_y: int | None = None,
    figma_width: int | None = None,
    figma_height: int | None = None,
) -> dict:
    """
    Crop the given region from the Figma frame + live screenshot, save to disk
    (when job_id is given), and return a dict of b64 strings + paths + URLs
    ready to merge into any issue dict — used by non-vision issue sources
    (e.g. typography_diff) so they get the same before/after thumbnails as
    vision-analyzed issues.

    x/y/w/h are the LIVE-side region. figma_x/figma_y/figma_width/figma_height
    are the matched FIGMA element's own region and default to the live-side
    values when omitted (preserves old callers' behavior) — but callers that
    have the real matched Figma box (typography_diff, geometry_diff) should
    always pass it explicitly. Reusing the live-side coordinates against the
    Figma image is exactly what produced blank/black or unrelated-content
    Figma-side crops: the live page's total height commonly diverges from the
    Figma frame's own height (e.g. after several repeating card rows), so an
    unscaled live y can land outside the Figma image's real pixel bounds
    (PIL pads that black) or on a genuinely different part of the design.
    """
    fx = figma_x if figma_x is not None else x
    fy = figma_y if figma_y is not None else y
    fw = figma_width if figma_width is not None else w
    fh = figma_height if figma_height is not None else h

    out: dict = {
        "expected_crop_b64": _crop_region(figma_bytes, fx, fy, fw, fh, canonical_width),
        "actual_crop_b64":   _crop_region(live_bytes,  x,  y,  w,  h,  canonical_width),
    }
    if job_id:
        output_root = Path(os.environ.get("OUTPUT_DIR", "artifacts")) / "screenshots"
        out["expected_screenshot_path"] = _save_crop(
            out["expected_crop_b64"], output_root / "expected" / job_id, filename
        )
        out["actual_screenshot_path"] = _save_crop(
            out["actual_crop_b64"], output_root / "current" / job_id, filename
        )
        out["expected_screenshot_url"] = f"/screenshots/expected/{job_id}/{filename}"
        out["actual_screenshot_url"] = f"/screenshots/current/{job_id}/{filename}"
    return out


def analyze(
    figma_bytes: bytes,
    live_bytes: bytes,
    compare_result: CompareResult,
    page_name: str = "page",
    job_id: str = "",
    figma_typography: dict | None = None,
    canonical_width: int = _CANONICAL_WIDTH,
) -> list[dict]:
    """
    Send Figma + live screenshots to Groq vision and get structured issue descriptions.
    Returns a list of issue dicts, one per diff region. Each issue carries:
      expected_crop_b64  — Figma region JPEG (base64)
      actual_crop_b64    — Live region JPEG (base64)
      diff_crop_b64      — Diff heatmap region JPEG (base64)
      expected_screenshot_path / actual_screenshot_path / diff_screenshot_path
        — saved to artifacts/screenshots/{expected|current|diff}/{job_id}/ when job_id is given
    """
    if not compare_result.regions:
        logger.info(f"No diff regions for {page_name} — skipping AI analysis")
        return []

    client = _get_client()

    figma_data_url = _encode_image(figma_bytes)
    live_data_url = _encode_image(live_bytes)

    region_lines = []
    for i, r in enumerate(compare_result.regions, 1):
        region_lines.append(
            f"  Region {i}: position ({r.x},{r.y}), size {r.width}x{r.height}px, "
            f"{r.diff_percent}% of pixels differ"
        )
    region_summary = "\n".join(region_lines)

    # Build typography context block from extracted Figma tokens
    typo = figma_typography or {}
    typo_lines = []
    if typo.get("fonts"):
        typo_lines.append(f"  - Font families: {', '.join(typo['fonts'])}")
    if typo.get("sizes"):
        typo_lines.append(f"  - Font sizes: {', '.join(str(s) + 'px' for s in typo['sizes'])}")
    if typo.get("weights"):
        weight_names = {100: "Thin", 200: "ExtraLight", 300: "Light", 400: "Regular",
                        500: "Medium", 600: "SemiBold", 700: "Bold", 800: "ExtraBold", 900: "Black"}
        wts = [f"{w} ({weight_names.get(w, '')})" for w in typo["weights"]]
        typo_lines.append(f"  - Font weights: {', '.join(wts)}")
    if typo.get("colors"):
        typo_lines.append(f"  - Text colors: {', '.join(typo['colors'][:8])}")

    typo_section = ""
    if typo_lines:
        typo_section = (
            "\n\nThe Figma design specifies these typography tokens:\n"
            + "\n".join(typo_lines)
            + "\n\nCompare the live site against these specifications."
        )

    # Build section context block so the AI knows which DOM section each region is in.
    # Section info is injected by visual_qa_agent._annotate_sections() after comparison.
    # Here we pre-compute it from DOM positions so the prompt is self-contained.

    prompt = f"""I'm comparing a Figma design mockup (Image 1) against the live Shopify store screenshot (Image 2) for the **{page_name}** page.

The automated pixel diff found {len(compare_result.regions)} changed region(s) with an overall {compare_result.diff_percent}% pixel difference:

{region_summary}{typo_section}

For each region, first check: is either image blank/black/a loading placeholder (capture_failure)? Is this a carousel/slideshow/UGC grid showing different rotating content rather than a real defect (expected_variance)? Only if neither applies, examine:
1. **Element** — what UI element is in that area (nav, hero, product card, button, heading, body text, etc.)
2. **Visual differences** — layout, colours, sizing, positioning
3. **Typography** — font family, font size, font weight, letter spacing, line height, text colour, text alignment
4. **User impact** — how this harms usability, brand trust, or conversions
5. **Suggested fix** — specific CSS property or design change to match the Figma spec

Return a JSON array with one object per region:
[
  {{
    "region_index": 1,
    "element": "element name",
    "description": "what specifically differs (include typography details if relevant)",
    "user_impact": "how this affects users",
    "suggested_fix": "specific change to match the Figma design",
    "issue_type": "typography | layout | color | spacing | image | capture_failure | expected_variance | other"
  }}
]

Return only the JSON array, no markdown fences."""

    logger.info(f"Sending {page_name} to Groq vision — {len(compare_result.regions)} regions")

    response = client.chat.completions.create(
        model=_VISION_MODEL,
        # Image tokens for this model are dominated by a large fixed per-image
        # cost, not resolution — shrinking images doesn't reduce request size.
        # The reserved max_tokens counts toward this org's 8000 TPM cap too, so
        # trimmed from 1500 to close the ~300-token gap that was tipping full-page
        # requests over the limit.
        max_tokens=1000,
        # qwen3.6-27b has a "thinking" mode that emits verbose reasoning text
        # before its answer — off, since we parse the response as a raw JSON
        # array and reasoning tokens would eat into max_tokens for no benefit.
        reasoning_effort="none",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 1 — Figma design mockup:"},
                    {"type": "image_url", "image_url": {"url": figma_data_url}},
                    {"type": "text", "text": "Image 2 — Live Shopify screenshot:"},
                    {"type": "image_url", "image_url": {"url": live_data_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    )

    raw = response.choices[0].message.content.strip()
    logger.info(f"Groq vision response for {page_name}: {len(raw)} chars")

    issues = _parse_response(raw, compare_result.regions)

    dropped = [iss for iss in issues if _is_empty_finding(iss.get("description", ""))]
    if dropped:
        logger.info(f"Dropped {len(dropped)} empty finding(s) for {page_name} — no concrete difference found")
    issues = [iss for iss in issues if not _is_empty_finding(iss.get("description", ""))]

    # Directories for disk-saving crops (only when job_id is provided)
    output_root = Path(os.environ.get("OUTPUT_DIR", "artifacts")) / "screenshots"
    current_dir  = output_root / "current"  / (job_id or "tmp")
    expected_dir = output_root / "expected" / (job_id or "tmp")
    diff_dir     = output_root / "diff"     / (job_id or "tmp")

    for i, issue in enumerate(issues):
        if not all(k in issue for k in ("x", "y", "width", "height")):
            continue
        x, y, w, h = issue["x"], issue["y"], issue["width"], issue["height"]
        safe_name = page_name.replace("/", "-").replace("\\", "-")
        slug = f"{safe_name}_i{i+1}.jpg"   # use position, not region_index, to avoid collisions
        try:
            issue["expected_crop_b64"] = _crop_region(figma_bytes, x, y, w, h, canonical_width)
            issue["actual_crop_b64"]   = _crop_region(live_bytes,  x, y, w, h, canonical_width)
            issue["diff_crop_b64"]     = _crop_region(compare_result.diff_mask, x, y, w, h, canonical_width)

            if job_id:
                issue["expected_screenshot_path"] = _save_crop(issue["expected_crop_b64"], expected_dir, slug)
                issue["actual_screenshot_path"]   = _save_crop(issue["actual_crop_b64"],   current_dir,  slug)
                issue["diff_screenshot_path"]     = _save_crop(issue["diff_crop_b64"],     diff_dir,     slug)
                # Browser-accessible URLs served by FastAPI /screenshots static mount
                issue["expected_screenshot_url"]  = f"/screenshots/expected/{job_id}/{slug}"
                issue["actual_screenshot_url"]    = f"/screenshots/current/{job_id}/{slug}"
                issue["diff_screenshot_url"]      = f"/screenshots/diff/{job_id}/{slug}"
                logger.info(f"Crops saved — job={job_id}, page={page_name}, issue={i+1}")
        except Exception as exc:
            logger.warning(f"Crop failed — page={page_name}, issue={i+1}: {exc}")

    return issues


_SECTION_PROMPT = """I'm comparing a Figma design section (Image 1) against the live Shopify store screenshot of the same section (Image 2) for the **{page_name}** page.

Figma section: **{figma_name}**
Live section:  **{live_name}**
Section match confidence: {confidence}%
Pixel diff: {diff_percent}% of pixels differ
SSIM similarity score: {ssim_score} (1.0 = identical, < 0.82 = significant difference)

First check: is either image blank/black/a loading placeholder (capture_failure)? Is this a carousel/slideshow/UGC grid showing different rotating content rather than a real defect (expected_variance)? Only if neither applies, examine ALL of the following:
1. **Element** — what UI element is shown (nav, hero, product card, button, heading, etc.)
2. **Visual differences** — layout, colours, sizing, positioning between Figma and live
3. **Typography** — font family, size, weight, letter spacing, line height, colour, alignment
4. **User impact** — how this harms usability, brand trust, or conversions
5. **Suggested fix** — specific CSS property or design change to match the Figma spec

Return a JSON array with one object per distinct issue found in this section:
[
  {{
    "region_index": 1,
    "element": "element name",
    "description": "what specifically differs",
    "user_impact": "how this affects users",
    "suggested_fix": "specific change to match the Figma design",
    "issue_type": "typography | layout | color | spacing | image | capture_failure | expected_variance | other"
  }}
]

Return only the JSON array, no markdown fences."""


def analyze_section_pairs(
    section_results: list,   # list[SectionCompareResult] from section_comparator
    page_name: str = "page",
    job_id: str = "",
) -> list[dict]:
    """
    Send each significant section pair to Groq vision for issue analysis.
    Returns a flat list of issue dicts enriched with section context.

    section_results: list of SectionCompareResult with has_significant_diff=True.
    """
    if not section_results:
        return []

    client = _get_client()
    all_issues: list[dict] = []

    output_root = Path(os.environ.get("OUTPUT_DIR", "artifacts")) / "screenshots"
    current_dir  = output_root / "current"  / (job_id or "tmp")
    expected_dir = output_root / "expected" / (job_id or "tmp")
    diff_dir     = output_root / "diff"     / (job_id or "tmp")

    for si, sr in enumerate(section_results):
        if not sr.figma_image_bytes or not sr.live_image_bytes:
            continue

        figma_url = _encode_image(sr.figma_image_bytes)
        live_url  = _encode_image(sr.live_image_bytes)

        prompt = _SECTION_PROMPT.format(
            page_name=page_name,
            figma_name=sr.figma_section_name,
            live_name=sr.live_section_name,
            confidence=sr.match_confidence,
            diff_percent=sr.diff_percent,
            ssim_score=sr.ssim_score,
        )

        try:
            response = client.chat.completions.create(
                model=_VISION_MODEL,
                max_tokens=1000,
                reasoning_effort="none",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",      "text": "Image 1 — Figma design section:"},
                            {"type": "image_url", "image_url": {"url": figma_url}},
                            {"type": "text",      "text": "Image 2 — Live Shopify section:"},
                            {"type": "image_url", "image_url": {"url": live_url}},
                            {"type": "text",      "text": prompt},
                        ],
                    },
                ],
            )
            raw = response.choices[0].message.content.strip()
            section_issues = _parse_section_response(raw, sr, si)
        except Exception as exc:
            logger.warning(f"Section AI analysis failed — '{sr.figma_section_name}': {exc}")
            section_issues = [_fallback_section_issue(sr, si)]

        if sr.match_confidence < _MIN_CONTENT_GATE_CONFIDENCE:
            logger.info(
                f"Section pair '{sr.figma_section_name}' <-> '{sr.live_section_name}' has "
                f"low match confidence ({sr.match_confidence:.0f} < {_MIN_CONTENT_GATE_CONFIDENCE}) "
                f"— downgrading {len(section_issues)} finding(s) to needs_recheck instead of "
                f"reporting as confident defects (the pairing itself is unverified)."
            )
            for issue in section_issues:
                issue["issue_type"] = "needs_recheck"
                issue["description"] = (
                    f"[Low section-match confidence: {sr.match_confidence:.0f}] "
                    + issue.get("description", "")
                )

        before = len(section_issues)
        section_issues = [iss for iss in section_issues if not _is_empty_finding(iss.get("description", ""))]
        if len(section_issues) < before:
            logger.info(
                f"Dropped {before - len(section_issues)} empty finding(s) for "
                f"'{sr.figma_section_name}' — no concrete difference found"
            )

        # Save section crops to disk when job_id is provided
        safe_name = page_name.replace("/", "-").replace("\\", "-")
        slug = f"{safe_name}_s{si + 1}.jpg"
        if job_id:
            try:
                _save_crop(
                    base64.standard_b64encode(sr.figma_image_bytes).decode(),
                    expected_dir, slug,
                )
                _save_crop(
                    base64.standard_b64encode(sr.live_image_bytes).decode(),
                    current_dir, slug,
                )
                if sr.diff_image_bytes:
                    diff_dir.mkdir(parents=True, exist_ok=True)
                    (diff_dir / slug).write_bytes(sr.diff_image_bytes)

                for issue in section_issues:
                    issue["expected_screenshot_url"] = f"/screenshots/expected/{job_id}/{slug}"
                    issue["actual_screenshot_url"]   = f"/screenshots/current/{job_id}/{slug}"
                    issue["diff_screenshot_url"]     = f"/screenshots/diff/{job_id}/{slug}"
                    issue["expected_screenshot_path"] = str(expected_dir / slug)
                    issue["actual_screenshot_path"]   = str(current_dir  / slug)
            except Exception as exc:
                logger.debug(f"Section crop save failed: {exc}")

        # Embed crops as base64 for the report
        b64_figma = base64.standard_b64encode(sr.figma_image_bytes).decode()
        b64_live  = base64.standard_b64encode(sr.live_image_bytes).decode()
        for issue in section_issues:
            issue["expected_crop_b64"] = b64_figma
            issue["actual_crop_b64"]   = b64_live
            issue["section_name"]      = sr.figma_section_name
            issue["match_confidence"]  = sr.match_confidence
            issue["ssim_score"]        = sr.ssim_score

        all_issues.extend(section_issues)

    logger.info(
        f"Section AI analysis complete — page={page_name}, "
        f"sections={len(section_results)}, issues={len(all_issues)}"
    )
    return all_issues


def _parse_section_response(raw: str, sr, section_idx: int) -> list[dict]:
    """Parse AI response for a single section pair."""
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            for item in parsed:
                item["diff_percent"]     = sr.diff_percent
                item["x"]               = 0
                item["y"]               = 0
                item["width"]           = 0
                item["height"]          = 0
            return parsed
    except Exception:
        pass
    return [_fallback_section_issue(sr, section_idx)]


def _fallback_section_issue(sr, section_idx: int) -> dict:
    return {
        "region_index": section_idx + 1,
        "element":      sr.figma_section_name or "section",
        "description":  f"Visual difference detected in section '{sr.figma_section_name}' "
                        f"(SSIM={sr.ssim_score}, diff={sr.diff_percent}%)",
        "user_impact":  "Visual discrepancy detected",
        "suggested_fix": "",
        "issue_type":   "other",
        "diff_percent": sr.diff_percent,
        "x": 0, "y": 0, "width": 0, "height": 0,
    }


def _parse_response(raw: str, regions: list[DiffRegion]) -> list[dict]:
    """Parse Groq's JSON response, fall back to basic issue list on failure."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            for i, item in enumerate(parsed):
                # Try to match by the AI-supplied region_index first
                ai_idx = item.get("region_index", 1) - 1
                r = regions[ai_idx] if 0 <= ai_idx < len(regions) else None
                # Positional fallback: if region_index is out of range or already
                # used (AI often returns 1 for every issue), fall back to position i
                if r is None or ("x" in item):
                    r = regions[i] if i < len(regions) else regions[-1]
                item["x"] = r.x
                item["y"] = r.y
                item["width"] = r.width
                item["height"] = r.height
                item["diff_percent"] = r.diff_percent
                # Normalise region_index so it matches the actual region used
                item["region_index"] = regions.index(r) + 1
            return parsed
    except Exception as e:
        logger.warning(f"Failed to parse Groq vision JSON: {e} — using fallback")

    return [
        {
            "region_index": i + 1,
            "element": "Unknown element",
            "description": raw[:300] if i == 0 else "See region 1 for full analysis",
            "user_impact": "Visual discrepancy detected",
            "suggested_fix": "",
            "issue_type": "other",
            "x": r.x, "y": r.y, "width": r.width, "height": r.height,
            "diff_percent": r.diff_percent,
        }
        for i, r in enumerate(regions)
    ]