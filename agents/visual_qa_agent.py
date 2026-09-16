import logging

from fastapi.concurrency import run_in_threadpool

from services.db import update_vqa_job
from services.bug_report_generator import build_full_report, build_page_report
from services.figma_extractor import extract_frames
from services.severity_classifier import classify_all
from services.shopify_scraper import capture_pages
from services.visual_ai_analyzer import analyze, analyze_section_pairs, build_issue_crops
from services.visual_comparator import compare, CompareResult
from services.typography_diff import compare_typography, match_text_nodes
from services.geometry_diff import compare_dimensions, compare_all_spacing
from qa.fix_recommendation_engine import generate_fix_recommendations
from visual.figma_section_extractor import crop_figma_sections
from visual.section_alignment_engine import (
    find_section_for_region,
    normalize_section_coords,
)
from visual.section_comparator import compare_section_pair, SectionCompareResult
from visual.section_matcher import match_sections, unmatched_figma_sections
from visual.section_targeting import (
    find_live_section,
    select_first_n_sections,
    derive_figma_region,
    filter_nodes_in_region,
    crop_frame_region,
)
from visual.section_exclusion import (
    normalize_exclude_types,
    excluded_regions,
    filter_excluded_issues,
    filter_out_sections,
    mask_excluded_regions,
)

logger = logging.getLogger(__name__)

_DEFAULT_PAGES = ["home", "product", "collection", "cart"]
_DEFAULT_CANONICAL_WIDTH = 1440   # fallback when a Figma frame has no usable width

# Section-based path is used when both Figma sections and live section screenshots
# are available AND the match produces at least this many pairs.
_MIN_SECTION_PAIRS = 2


async def run_visual_qa(
    job_id: str,
    shopify_url: str,
    figma_url: str,
    pages: list[str] | None = None,
    shopify_password: str | None = None,
    diff_threshold: float = 0.05,
    exclude_sections: list[str] | None = None,
) -> dict:
    """
    Full visual QA pipeline. Runs as a FastAPI BackgroundTask.
    Writes progress updates to MongoDB so the polling endpoint reflects live status.
    Returns the full report dict (also persisted to MongoDB on completion).

    exclude_sections: section types to skip entirely, e.g. ["header", "footer",
    "collection"] — any issue whose region substantially overlaps a section of
    that type (matched on either the Figma or live side) is dropped from the
    final report.
    """
    if not pages:
        pages = _DEFAULT_PAGES

    exclude_types = normalize_exclude_types(exclude_sections)

    try:
        # ── Step 1: Fetch Figma frames first — we need each frame's own width ──
        # before capturing the live site, so the browser viewport matches it.
        await _progress(job_id, "running", "Fetching Figma frames...")

        try:
            figma_frames, figma_typography = await extract_frames(figma_url)
        except Exception as e:
            await _fail(job_id, f"Figma fetch failed: {e}")
            raise

        if not figma_frames:
            await _fail(job_id, "No frames found in the Figma file.")
            raise ValueError("No frames found in the Figma file.")

        logger.info(
            f"Job {job_id}: {len(figma_frames)} Figma frame(s), "
            f"fonts={figma_typography.get('fonts', [])}"
        )

        # ── Step 2: Capture the live site at each page's matched frame width ──
        page_widths = {
            page_name: _frame_width(_best_frame_for(page_name, figma_frames))
            for page_name in pages
        }
        logger.info(f"Job {job_id}: viewport widths per page — {page_widths}")

        await _progress(job_id, "running", "Capturing Shopify screenshots...")
        try:
            shopify_pages = await run_in_threadpool(
                capture_pages, shopify_url, pages, shopify_password, 45000, job_id, page_widths
            )
        except Exception as e:
            await _fail(job_id, f"Shopify capture failed: {e}")
            raise

        # ── Step 3: Match each Shopify page to its Figma frame ───────────────
        page_frame_pairs = _match_pages_to_frames(shopify_pages, figma_frames)

        # ── Step 4: Compare + analyse each page ──────────────────────────────
        page_reports = []
        total = len(page_frame_pairs)

        for idx, (shopify_page, figma_frame) in enumerate(page_frame_pairs, 1):
            page_name = shopify_page["page"]

            if shopify_page["error"] or not shopify_page["screenshot"]:
                logger.warning(
                    f"Job {job_id}: skipping {page_name} — "
                    f"screenshot failed: {shopify_page['error']}"
                )
                continue

            await _progress(job_id, "running", f"Comparing {page_name} ({idx}/{total})...")

            # ── Normalize DOM sections to this frame's own canonical width ────
            frame_width = _frame_width(figma_frame)
            raw_sections = shopify_page.get("sections", [])
            # DOM coordinates come from getBoundingClientRect(), which is always in
            # logical CSS px — i.e. the browser viewport width, NOT the raw screenshot
            # image's pixel width (which is viewport_width * device_scale_factor).
            # Using the image width here would silently corrupt every DOM coordinate.
            live_original_width = shopify_page.get("viewport_width") or frame_width

            dom_sections = normalize_section_coords(
                raw_sections, live_original_width, frame_width
            )

            # ── Try section-based comparison first ───────────────────────────
            # Excluded-type sections (header/footer/collection/etc.) are dropped
            # before matching so the section pipeline never spends an AI vision
            # call analysing a region that will be filtered out of the report.
            await _progress(
                job_id, "running", f"Matching sections for {page_name} ({idx}/{total})..."
            )
            pipeline_dom_sections, pipeline_figma_sections = filter_out_sections(
                dom_sections, figma_frame.get("sections", []), exclude_types
            )
            pipeline_figma_frame = (
                {**figma_frame, "sections": pipeline_figma_sections}
                if exclude_types else figma_frame
            )
            issues = await _run_section_pipeline(
                figma_frame=pipeline_figma_frame,
                shopify_page=shopify_page,
                dom_sections=pipeline_dom_sections,
                page_name=page_name,
                job_id=job_id,
                diff_threshold=diff_threshold,
                figma_typography=figma_typography,
            )

            # ── Full-page fallback when section pipeline yields nothing ───────
            if issues is None:
                logger.info(
                    f"Job {job_id} / {page_name}: section pipeline skipped — "
                    "falling back to full-page comparison"
                )

                # Blank out excluded sections (header/footer/etc.) before
                # diffing or showing anything to the vision model — a single
                # merged diff region can span both an excluded and a kept
                # section, and filtering issues after the fact by bounding-box
                # overlap can miss that, letting excluded content leak into a
                # kept issue's crop and into the model's full-page view.
                fallback_figma_bytes = figma_frame["image_bytes"]
                fallback_live_bytes = shopify_page["screenshot"]
                if exclude_types:
                    fallback_exclude_boxes = excluded_regions(
                        dom_sections, figma_frame.get("sections", []), exclude_types
                    )
                    fallback_figma_bytes = mask_excluded_regions(
                        fallback_figma_bytes, fallback_exclude_boxes, frame_width
                    )
                    fallback_live_bytes = mask_excluded_regions(
                        fallback_live_bytes, fallback_exclude_boxes, frame_width
                    )

                compare_result = compare(
                    figma_bytes=fallback_figma_bytes,
                    live_bytes=fallback_live_bytes,
                    diff_threshold=diff_threshold,
                    canonical_width=frame_width,
                )
                logger.info(
                    f"Job {job_id} / {page_name}: {compare_result.diff_percent}% diff, "
                    f"{len(compare_result.regions)} region(s)"
                )

                if compare_result.regions:
                    await _progress(job_id, "running", f"Analysing {page_name} with AI...")
                    issues = await run_in_threadpool(
                        analyze,
                        fallback_figma_bytes,
                        fallback_live_bytes,
                        compare_result,
                        page_name,
                        job_id,
                        figma_typography,
                        frame_width,
                    )
                    issues = _annotate_sections(issues, dom_sections)
                else:
                    issues = []

                # Re-build a compare_result for the page report
                page_compare = compare_result
            else:
                # Section pipeline succeeded — build a synthetic CompareResult for the report
                page_compare = compare(
                    figma_bytes=figma_frame["image_bytes"],
                    live_bytes=shopify_page["screenshot"],
                    diff_threshold=diff_threshold,
                    canonical_width=frame_width,
                )

            # ── Typography diff: precise computed-style comparison ────────────
            # Independent of the vision model's pixel-based guesses — reads real
            # font-weight/size/color from Figma's API and the live DOM's computed
            # styles, so e.g. "font-weight 600 vs 400" is a measured fact, not a guess.
            try:
                live_text_nodes = normalize_section_coords(
                    shopify_page.get("text_nodes", []), live_original_width, frame_width
                )
                typography_issues = compare_typography(
                    figma_frame.get("text_nodes", []), live_text_nodes
                )
            except Exception as exc:
                logger.warning(f"Typography diff failed for {page_name}: {exc}")
                typography_issues = []

            if typography_issues:
                safe_page_name = page_name.replace("/", "-").replace("\\", "-")

                def _crop_typography_issues():
                    for ti, t_issue in enumerate(typography_issues, 1):
                        slug = f"{safe_page_name}_typo{ti}.jpg"
                        t_issue.update(build_issue_crops(
                            figma_frame["image_bytes"], shopify_page["screenshot"],
                            t_issue["x"], t_issue["y"], t_issue["width"], t_issue["height"],
                            frame_width, job_id, slug,
                            figma_x=t_issue.get("figma_x", t_issue["x"]),
                            figma_y=t_issue.get("figma_y", t_issue["y"]),
                            figma_width=t_issue.get("figma_width", t_issue["width"]),
                            figma_height=t_issue.get("figma_height", t_issue["height"]),
                        ))
                    return typography_issues

                typography_issues = await run_in_threadpool(_crop_typography_issues)
                logger.info(
                    f"Job {job_id} / {page_name}: {len(typography_issues)} typography issue(s)"
                )

            # ── Geometry diff: real element dimensions + rendered spacing ─────
            # Same measured-fact philosophy as typography — real Figma node
            # boxes vs real getBoundingClientRect() boxes, not a vision guess.
            try:
                live_image_elements = normalize_section_coords(
                    shopify_page.get("image_elements", []), live_original_width, frame_width
                )
                live_button_elements = normalize_section_coords(
                    shopify_page.get("button_elements", []), live_original_width, frame_width
                )
                figma_image_nodes = figma_frame.get("image_nodes", [])
                figma_button_nodes = figma_frame.get("button_nodes", [])

                dimension_issues = (
                    compare_dimensions(figma_image_nodes, live_image_elements, "label", "image")
                    + compare_dimensions(figma_button_nodes, live_button_elements, "label", "button")
                )

                # Section-to-section, text-to-text (title/subtitle etc.),
                # image-to-image, and button spacing — all in one pass, using
                # the exclusion-filtered sections so header/footer never
                # factor into a spacing check when excluded.
                spacing_issues = compare_all_spacing(
                    figma_frame.get("text_nodes", []), live_text_nodes,
                    figma_image_nodes, live_image_elements,
                    figma_button_nodes, live_button_elements,
                    pipeline_figma_sections, pipeline_dom_sections,
                )
            except Exception as exc:
                logger.warning(f"Geometry diff failed for {page_name}: {exc}")
                dimension_issues, spacing_issues = [], []

            geometry_issues = dimension_issues + spacing_issues
            if geometry_issues:
                safe_page_name = page_name.replace("/", "-").replace("\\", "-")

                def _crop_geometry_issues():
                    for gi, g_issue in enumerate(geometry_issues, 1):
                        slug = f"{safe_page_name}_geom{gi}.jpg"
                        g_issue.update(build_issue_crops(
                            figma_frame["image_bytes"], shopify_page["screenshot"],
                            g_issue["x"], g_issue["y"], g_issue["width"], g_issue["height"],
                            frame_width, job_id, slug,
                            figma_x=g_issue.get("figma_x", g_issue["x"]),
                            figma_y=g_issue.get("figma_y", g_issue["y"]),
                            figma_width=g_issue.get("figma_width", g_issue["width"]),
                            figma_height=g_issue.get("figma_height", g_issue["height"]),
                        ))
                    return geometry_issues

                geometry_issues = await run_in_threadpool(_crop_geometry_issues)
                logger.info(
                    f"Job {job_id} / {page_name}: {len(dimension_issues)} dimension, "
                    f"{len(spacing_issues)} spacing issue(s)"
                )

            issues = (issues or []) + typography_issues + geometry_issues

            # ── Drop issues in excluded sections (e.g. header, footer, collection) ─
            if exclude_types:
                exclude_boxes = excluded_regions(
                    dom_sections, figma_frame.get("sections", []), exclude_types
                )
                before = len(issues)
                issues = filter_excluded_issues(issues, exclude_boxes)
                if before != len(issues):
                    logger.info(
                        f"Job {job_id} / {page_name}: excluded {before - len(issues)} "
                        f"issue(s) in {sorted(exclude_types)} section(s)"
                    )

            if issues:
                issues = await classify_all(issues)
                await _progress(
                    job_id, "running", f"Generating fix recommendations for {page_name}..."
                )
                issues = await generate_fix_recommendations(issues, page_name, shopify_url)

            page_report = build_page_report(
                page_name=page_name,
                shopify_url=shopify_page["url"],
                figma_frame=figma_frame,
                live_screenshot=shopify_page["screenshot"],
                compare_result=page_compare,
                issues=issues,
            )
            page_reports.append(page_report)

        if not page_reports:
            await _fail(job_id, "All page screenshots failed — nothing to compare.")
            raise RuntimeError("All page screenshots failed.")

        # ── Step 5: Build + persist full report ──────────────────────────────
        await _progress(job_id, "running", "Building report...")

        report = build_full_report(
            job_id=job_id,
            shopify_url=shopify_url,
            figma_url=figma_url,
            page_reports=page_reports,
        )

        logger.info(
            f"Job {job_id} complete — overall={report['overall_severity']}, "
            f"issues={report['total_issues']}"
        )
        return report

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        await _fail(job_id, str(e))
        raise


async def run_single_section_qa(
    job_id: str,
    shopify_url: str,
    figma_url: str,
    page_name: str,
    target_section: str,
    shopify_password: str | None = None,
    include_typography: bool = True,
) -> dict:
    """
    Scoped QA run for exactly one named section (matched by free-text query
    against live headings/Figma text), instead of the whole page.

    Compares typography (unless include_typography=False), element
    dimensions, and spacing — full pixel/vision analysis scoped to a single
    section isn't
    wired up yet, since the vision model needs a stable crop boundary on both
    sides and that's a separate piece of work from section targeting itself.
    """
    try:
        await _progress(job_id, "running", "Fetching Figma frames...")
        try:
            figma_frames, figma_typography = await extract_frames(figma_url)
        except Exception as e:
            await _fail(job_id, f"Figma fetch failed: {e}")
            raise

        if not figma_frames:
            await _fail(job_id, "No frames found in the Figma file.")
            raise ValueError("No frames found in the Figma file.")

        figma_frame = _best_frame_for(page_name, figma_frames)
        frame_width = _frame_width(figma_frame)

        await _progress(job_id, "running", "Capturing Shopify screenshot...")
        try:
            shopify_pages = await run_in_threadpool(
                capture_pages, shopify_url, [page_name], shopify_password, 45000, job_id,
                {page_name: frame_width},
            )
        except Exception as e:
            await _fail(job_id, f"Shopify capture failed: {e}")
            raise

        shopify_page = shopify_pages[0]
        if shopify_page["error"] or not shopify_page["screenshot"]:
            await _fail(job_id, f"Screenshot failed: {shopify_page['error']}")
            raise RuntimeError(shopify_page["error"])

        live_original_width = shopify_page.get("viewport_width") or frame_width
        dom_sections = normalize_section_coords(
            shopify_page.get("sections", []), live_original_width, frame_width
        )
        live_text_nodes = normalize_section_coords(
            shopify_page.get("text_nodes", []), live_original_width, frame_width
        )
        live_image_elements = normalize_section_coords(
            shopify_page.get("image_elements", []), live_original_width, frame_width
        )
        live_button_elements = normalize_section_coords(
            shopify_page.get("button_elements", []), live_original_width, frame_width
        )

        await _progress(job_id, "running", f"Locating section '{target_section}'...")
        live_section = find_live_section(target_section, dom_sections)
        if not live_section:
            await _fail(
                job_id,
                f"Couldn't find a section matching '{target_section}' on the live page.",
            )
            raise ValueError(f"Section not found: {target_section}")

        logger.info(
            f"Job {job_id}: target section matched — heading={live_section.get('heading')!r}, "
            f"y={live_section.get('y')}, height={live_section.get('height')}"
        )

        # Live page's own logical height (bottom edge of its last section) —
        # needed to scale positions onto the Figma frame's height, since the
        # two are rarely pixel-identical even for a faithful implementation.
        live_page_height = max(
            (s.get("y", 0) + s.get("height", 0) for s in dom_sections), default=0
        )
        figma_frame_height = figma_frame.get("height") or live_page_height or 1

        figma_region = derive_figma_region(
            live_section, figma_frame.get("text_nodes", []),
            live_page_height, figma_frame_height, frame_width,
        )

        live_y0 = live_section["y"]
        live_y1 = live_section["y"] + live_section["height"]
        figma_y0 = figma_region["y"]
        figma_y1 = figma_region["y"] + figma_region["height"]

        figma_nodes_in_section = filter_nodes_in_region(
            figma_frame.get("text_nodes", []), figma_y0, figma_y1
        )
        live_nodes_in_section = filter_nodes_in_region(live_text_nodes, live_y0, live_y1)

        logger.info(
            f"Job {job_id}: scoped to section — {len(figma_nodes_in_section)} Figma "
            f"text node(s), {len(live_nodes_in_section)} live text node(s) in range"
        )

        typography_issues: list[dict] = []
        if include_typography:
            await _progress(job_id, "running", f"Comparing typography for '{target_section}'...")
            typography_issues = compare_typography(figma_nodes_in_section, live_nodes_in_section)

        if typography_issues:
            safe_page_name = page_name.replace("/", "-").replace("\\", "-")

            def _crop_typography_issues():
                for ti, t_issue in enumerate(typography_issues, 1):
                    slug = f"{safe_page_name}_section-typo{ti}.jpg"
                    t_issue.update(build_issue_crops(
                        figma_frame["image_bytes"], shopify_page["screenshot"],
                        t_issue["x"], t_issue["y"], t_issue["width"], t_issue["height"],
                        frame_width, job_id, slug,
                        figma_x=t_issue.get("figma_x", t_issue["x"]),
                        figma_y=t_issue.get("figma_y", t_issue["y"]),
                        figma_width=t_issue.get("figma_width", t_issue["width"]),
                        figma_height=t_issue.get("figma_height", t_issue["height"]),
                    ))
                return typography_issues

            typography_issues = await run_in_threadpool(_crop_typography_issues)
            logger.info(
                f"Job {job_id}: {len(typography_issues)} typography issue(s) in "
                f"'{target_section}'"
            )

        # ── Geometry diff, scoped to the same section boundaries ─────────────
        await _progress(job_id, "running", f"Comparing dimensions/spacing for '{target_section}'...")
        try:
            figma_image_nodes = filter_nodes_in_region(
                figma_frame.get("image_nodes", []), figma_y0, figma_y1
            )
            figma_button_nodes = filter_nodes_in_region(
                figma_frame.get("button_nodes", []), figma_y0, figma_y1
            )
            live_image_in_section = filter_nodes_in_region(live_image_elements, live_y0, live_y1)
            live_button_in_section = filter_nodes_in_region(live_button_elements, live_y0, live_y1)

            dimension_issues = (
                compare_dimensions(figma_image_nodes, live_image_in_section, "label", "image")
                + compare_dimensions(figma_button_nodes, live_button_in_section, "label", "button")
            )

            # Text-to-text (title/subtitle etc.), image-to-image, and button
            # spacing within the target section. No section-to-section pass
            # here — a single-section run has exactly one target section by
            # definition, so there's nothing to measure it against.
            #
            # Neighbor pool is the FULL page's boxes, not the section-clipped
            # subset above — an element near this section's own top/bottom
            # edge needs its true nearest neighbor to be searchable even when
            # that neighbor sits just outside the section boundary, or the
            # search falls through to a much farther, wrong node and reports
            # a bogus inflated gap for an element that isn't actually
            # misplaced.
            spacing_issues = compare_all_spacing(
                figma_nodes_in_section, live_nodes_in_section,
                figma_image_nodes, live_image_in_section,
                figma_button_nodes, live_button_in_section,
                neighbor_figma_boxes=(
                    figma_frame.get("text_nodes", []) + figma_frame.get("image_nodes", [])
                    + figma_frame.get("button_nodes", [])
                ),
                neighbor_live_boxes=live_text_nodes + live_image_elements + live_button_elements,
            )
        except Exception as exc:
            logger.warning(f"Job {job_id}: geometry diff failed for '{target_section}': {exc}")
            dimension_issues, spacing_issues = [], []

        if dimension_issues or spacing_issues:
            await _progress(
                job_id, "running",
                f"Cross-checking geometry findings for '{target_section}' at a reference viewport width...",
            )
            spacing_issues, dimension_issues = await _crossvalidate_geometry_issues(
                job_id, shopify_url, shopify_password, page_name, target_section,
                frame_width, figma_nodes_in_section, figma_image_nodes, figma_button_nodes,
                spacing_issues, dimension_issues,
                neighbor_figma_boxes=(
                    figma_frame.get("text_nodes", []) + figma_frame.get("image_nodes", [])
                    + figma_frame.get("button_nodes", [])
                ),
            )

        geometry_issues = dimension_issues + spacing_issues
        if geometry_issues:
            safe_page_name = page_name.replace("/", "-").replace("\\", "-")

            def _crop_geometry_issues():
                for gi, g_issue in enumerate(geometry_issues, 1):
                    slug = f"{safe_page_name}_section-geom{gi}.jpg"
                    g_issue.update(build_issue_crops(
                        figma_frame["image_bytes"], shopify_page["screenshot"],
                        g_issue["x"], g_issue["y"], g_issue["width"], g_issue["height"],
                        frame_width, job_id, slug,
                        figma_x=g_issue.get("figma_x", g_issue["x"]),
                        figma_y=g_issue.get("figma_y", g_issue["y"]),
                        figma_width=g_issue.get("figma_width", g_issue["width"]),
                        figma_height=g_issue.get("figma_height", g_issue["height"]),
                    ))
                return geometry_issues

            geometry_issues = await run_in_threadpool(_crop_geometry_issues)
            logger.info(
                f"Job {job_id}: {len(dimension_issues)} dimension, {len(spacing_issues)} "
                f"spacing issue(s) in '{target_section}'"
            )

        all_issues = typography_issues + geometry_issues
        issues = await classify_all(all_issues) if all_issues else []
        if issues:
            await _progress(
                job_id, "running", f"Generating fix recommendations for '{target_section}'..."
            )
            issues = await generate_fix_recommendations(issues, page_name, shopify_url)

        # Section-scoped pixel compare, purely for the report's diff-image/
        # percentage display — no issues are generated from this, since vision
        # analysis isn't scoped to a single section yet.
        compare_result = None
        live_section_crop = live_section.get("screenshot")
        if live_section_crop:
            try:
                figma_crop_bytes = await run_in_threadpool(
                    crop_frame_region, figma_frame["image_bytes"], figma_region, frame_width
                )
                compare_result = await run_in_threadpool(
                    compare, figma_crop_bytes, live_section_crop, 0.05, frame_width
                )
            except Exception as exc:
                logger.warning(f"Job {job_id}: section-level pixel compare failed: {exc}")

        if compare_result is None:
            compare_result = CompareResult(diff_percent=0.0, regions=[], diff_image=b"", diff_mask=b"")

        page_report = build_page_report(
            page_name=f"{page_name} → {target_section}",
            shopify_url=shopify_page["url"],
            figma_frame=figma_frame,
            live_screenshot=live_section_crop or shopify_page["screenshot"],
            compare_result=compare_result,
            issues=issues,
        )

        report = build_full_report(
            job_id=job_id,
            shopify_url=shopify_url,
            figma_url=figma_url,
            page_reports=[page_report],
        )
        logger.info(
            f"Job {job_id} complete (single-section) — overall={report['overall_severity']}, "
            f"issues={report['total_issues']}"
        )
        return report

    except Exception as e:
        logger.error(f"Job {job_id} (single-section) failed: {e}")
        await _fail(job_id, str(e))
        raise


async def run_multi_section_qa(
    job_id: str,
    shopify_url: str,
    figma_url: str,
    page_name: str,
    section_limit: int,
    exclude_sections: list[str] | None = None,
    shopify_password: str | None = None,
    include_typography: bool = True,
) -> dict:
    """
    Scoped QA run for the first N real-content sections of a page (e.g.
    "first 3 sections"), in page order, after dropping any excluded section
    types (typically header/footer) — so "first 3" means the first 3 sections
    a visitor would actually scroll through, not the first 3 raw DOM entries.

    Each section is scoped independently (own Figma region, own text/image/
    button subset) using the same targeting technique as run_single_section_qa,
    then all sections' issues are combined into one report. Typography can be
    turned off (include_typography=False) when it's already been separately
    verified and only geometry (dimension/spacing) checks are wanted.
    """
    try:
        exclude_types = normalize_exclude_types(exclude_sections)

        await _progress(job_id, "running", "Fetching Figma frames...")
        try:
            figma_frames, figma_typography = await extract_frames(figma_url)
        except Exception as e:
            await _fail(job_id, f"Figma fetch failed: {e}")
            raise

        if not figma_frames:
            await _fail(job_id, "No frames found in the Figma file.")
            raise ValueError("No frames found in the Figma file.")

        figma_frame = _best_frame_for(page_name, figma_frames)
        frame_width = _frame_width(figma_frame)

        await _progress(job_id, "running", "Capturing Shopify screenshot...")
        try:
            shopify_pages = await run_in_threadpool(
                capture_pages, shopify_url, [page_name], shopify_password, 45000, job_id,
                {page_name: frame_width},
            )
        except Exception as e:
            await _fail(job_id, f"Shopify capture failed: {e}")
            raise

        shopify_page = shopify_pages[0]
        if shopify_page["error"] or not shopify_page["screenshot"]:
            await _fail(job_id, f"Screenshot failed: {shopify_page['error']}")
            raise RuntimeError(shopify_page["error"])

        live_original_width = shopify_page.get("viewport_width") or frame_width
        dom_sections = normalize_section_coords(
            shopify_page.get("sections", []), live_original_width, frame_width
        )
        live_text_nodes = normalize_section_coords(
            shopify_page.get("text_nodes", []), live_original_width, frame_width
        )
        live_image_elements = normalize_section_coords(
            shopify_page.get("image_elements", []), live_original_width, frame_width
        )
        live_button_elements = normalize_section_coords(
            shopify_page.get("button_elements", []), live_original_width, frame_width
        )

        await _progress(job_id, "running", f"Selecting first {section_limit} section(s)...")
        selected_sections = select_first_n_sections(dom_sections, section_limit, exclude_types)
        if not selected_sections:
            await _fail(job_id, "No non-excluded sections found on this page.")
            raise ValueError("No sections selected.")

        logger.info(
            f"Job {job_id}: selected {len(selected_sections)} section(s) — "
            f"{[s.get('heading') or s.get('id') for s in selected_sections]}"
        )

        live_page_height = max(
            (s.get("y", 0) + s.get("height", 0) for s in dom_sections), default=0
        )
        figma_frame_height = figma_frame.get("height") or live_page_height or 1

        # Full-page neighbor pool for spacing checks, built once — an element
        # near a section's own top/bottom edge needs its true nearest
        # neighbor searchable even when that neighbor sits just outside this
        # loop's per-section boundary, or the search falls through to a much
        # farther, wrong node and reports a bogus inflated gap for an element
        # that isn't actually misplaced. Confirmed against a real report: a
        # target whose genuine 10px neighbor sat just past its section's
        # clipped boundary came back as "72px off" instead.
        neighbor_figma_boxes = (
            figma_frame.get("text_nodes", []) + figma_frame.get("image_nodes", [])
            + figma_frame.get("button_nodes", [])
        )
        neighbor_live_boxes = live_text_nodes + live_image_elements + live_button_elements

        safe_page_name = page_name.replace("/", "-").replace("\\", "-")
        all_issues: list[dict] = []
        # Shared across every section in this loop so a Figma text node
        # already claimed as one section's anchor can't also become a later
        # section's anchor (e.g. two "Luxury SUVs" cards independently
        # resolving to the same Figma badge) — mutated in place by
        # derive_figma_region on each call.
        used_anchor_indices: set[int] = set()

        for sec_idx, live_section in enumerate(selected_sections, 1):
            label = live_section.get("heading") or live_section.get("id") or f"section {sec_idx}"
            await _progress(job_id, "running", f"Comparing section {sec_idx}/{len(selected_sections)}: '{label}'...")

            figma_region = derive_figma_region(
                live_section, figma_frame.get("text_nodes", []),
                live_page_height, figma_frame_height, frame_width,
                used_anchor_indices=used_anchor_indices,
            )
            live_y0 = live_section["y"]
            live_y1 = live_section["y"] + live_section["height"]
            figma_y0 = figma_region["y"]
            figma_y1 = figma_region["y"] + figma_region["height"]

            figma_nodes_in_section = filter_nodes_in_region(figma_frame.get("text_nodes", []), figma_y0, figma_y1)
            live_nodes_in_section = filter_nodes_in_region(live_text_nodes, live_y0, live_y1)
            figma_image_nodes = filter_nodes_in_region(figma_frame.get("image_nodes", []), figma_y0, figma_y1)
            figma_button_nodes = filter_nodes_in_region(figma_frame.get("button_nodes", []), figma_y0, figma_y1)
            live_image_in_section = filter_nodes_in_region(live_image_elements, live_y0, live_y1)
            live_button_in_section = filter_nodes_in_region(live_button_elements, live_y0, live_y1)

            section_issues: list[dict] = []

            if include_typography:
                try:
                    section_issues += compare_typography(figma_nodes_in_section, live_nodes_in_section)
                except Exception as exc:
                    logger.warning(f"Job {job_id}: typography diff failed for '{label}': {exc}")

            try:
                section_issues += compare_dimensions(figma_image_nodes, live_image_in_section, "label", "image")
                section_issues += compare_dimensions(figma_button_nodes, live_button_in_section, "label", "button")
                section_issues += compare_all_spacing(
                    figma_nodes_in_section, live_nodes_in_section,
                    figma_image_nodes, live_image_in_section,
                    figma_button_nodes, live_button_in_section,
                    neighbor_figma_boxes=neighbor_figma_boxes,
                    neighbor_live_boxes=neighbor_live_boxes,
                )
            except Exception as exc:
                logger.warning(f"Job {job_id}: geometry diff failed for '{label}': {exc}")

            if section_issues:
                def _crop_section_issues(issues=section_issues, idx=sec_idx):
                    for ii, issue in enumerate(issues, 1):
                        slug = f"{safe_page_name}_sec{idx}_{ii}.jpg"
                        issue.update(build_issue_crops(
                            figma_frame["image_bytes"], shopify_page["screenshot"],
                            issue["x"], issue["y"], issue["width"], issue["height"],
                            frame_width, job_id, slug,
                            figma_x=issue.get("figma_x", issue["x"]),
                            figma_y=issue.get("figma_y", issue["y"]),
                            figma_width=issue.get("figma_width", issue["width"]),
                            figma_height=issue.get("figma_height", issue["height"]),
                        ))
                        issue["element"] = f"[{label}] {issue.get('element', '')}"[:100]
                    return issues

                section_issues = await run_in_threadpool(_crop_section_issues)
                logger.info(f"Job {job_id}: {len(section_issues)} issue(s) in section '{label}'")

            all_issues.extend(section_issues)

        issues = await classify_all(all_issues) if all_issues else []
        if issues:
            await _progress(job_id, "running", "Generating fix recommendations...")
            issues = await generate_fix_recommendations(issues, page_name, shopify_url)

        # Combined pixel compare spanning all selected sections, purely for
        # the report's diff-image/percentage display.
        combined_top = min(s["y"] for s in selected_sections)
        combined_bottom = max(s["y"] + s["height"] for s in selected_sections)
        combined_region = {"x": 0, "y": combined_top, "width": frame_width, "height": combined_bottom - combined_top}
        figma_scale = figma_frame_height / live_page_height if live_page_height else 1.0
        figma_combined_region = {
            "x": 0,
            "y": combined_top * figma_scale,
            "width": frame_width,
            "height": (combined_bottom - combined_top) * figma_scale,
        }

        compare_result = None
        live_crop_bytes = None
        try:
            figma_crop_bytes = await run_in_threadpool(
                crop_frame_region, figma_frame["image_bytes"], figma_combined_region, frame_width
            )
            live_crop_bytes = await run_in_threadpool(
                crop_frame_region, shopify_page["screenshot"], combined_region, frame_width
            )
            compare_result = await run_in_threadpool(
                compare, figma_crop_bytes, live_crop_bytes, 0.05, frame_width
            )
        except Exception as exc:
            logger.warning(f"Job {job_id}: combined pixel compare failed: {exc}")

        if compare_result is None:
            compare_result = CompareResult(diff_percent=0.0, regions=[], diff_image=b"", diff_mask=b"")

        section_labels = ", ".join(s.get("heading") or s.get("id") or "?" for s in selected_sections)
        page_report = build_page_report(
            page_name=f"{page_name} → first {len(selected_sections)} section(s) ({section_labels})",
            shopify_url=shopify_page["url"],
            figma_frame=figma_frame,
            live_screenshot=live_crop_bytes or shopify_page["screenshot"],
            compare_result=compare_result,
            issues=issues,
        )

        report = build_full_report(
            job_id=job_id,
            shopify_url=shopify_url,
            figma_url=figma_url,
            page_reports=[page_report],
        )
        logger.info(
            f"Job {job_id} complete (multi-section) — overall={report['overall_severity']}, "
            f"issues={report['total_issues']}"
        )
        return report

    except Exception as e:
        logger.error(f"Job {job_id} (multi-section) failed: {e}")
        await _fail(job_id, str(e))
        raise


async def _run_section_pipeline(
    figma_frame:     dict,
    shopify_page:    dict,
    dom_sections:    list[dict],
    page_name:       str,
    job_id:          str,
    diff_threshold:  float,
    figma_typography: dict,
) -> list[dict] | None:
    """
    Section-based comparison pipeline.

    Returns a list of issues if successful, or None to signal the caller
    should fall back to full-page comparison.

    The pipeline is skipped (returns None) when:
    - The Figma frame has no section children
    - The live page has no DOM sections with screenshots
    - Fewer than _MIN_SECTION_PAIRS matched pairs are found
    """
    # Crop Figma sections from the full frame PNG
    figma_sections = await run_in_threadpool(crop_figma_sections, figma_frame)
    if not figma_sections:
        logger.info(f"No Figma sections for {page_name} — skipping section pipeline")
        return None

    # Only use live sections that have a screenshot captured
    live_with_screenshots = [s for s in dom_sections if s.get("screenshot")]
    if not live_with_screenshots:
        logger.info(f"No live section screenshots for {page_name} — skipping section pipeline")
        return None

    # Match Figma sections to live DOM sections
    matched_pairs = await run_in_threadpool(
        match_sections, figma_sections, live_with_screenshots
    )
    if len(matched_pairs) < _MIN_SECTION_PAIRS:
        logger.info(
            f"Only {len(matched_pairs)} section pairs for {page_name} "
            f"(need {_MIN_SECTION_PAIRS}) — skipping section pipeline"
        )
        return None

    # Figma sections the optimal matcher couldn't confidently pair with any
    # live section — surface for manual review rather than silently dropping
    # them (a Figma section with no live counterpart may mean it's genuinely
    # missing from the live page, or that the matcher just couldn't locate
    # it — either way worth a human look, not a confident finding either way).
    unmatched_sections = unmatched_figma_sections(figma_sections, matched_pairs)
    unmatched_issues = [
        {
            "element": fsec.get("name", "Figma section"),
            "description": (
                f"Figma section '{fsec.get('name', '?')}' had no confident match "
                "on the live page — needs manual review."
            ),
            "user_impact": "Cannot verify this section was implemented without manual review.",
            "suggested_fix": "Manually confirm whether this section exists on the live page.",
            "issue_type": "needs_recheck",
            "x": fsec.get("rel_x", 0), "y": fsec.get("rel_y", 0),
            "width": fsec.get("width", 0), "height": fsec.get("height", 0),
            "diff_percent": 0.0,
        }
        for fsec in unmatched_sections
    ]
    if unmatched_issues:
        logger.info(
            f"Job {job_id} / {page_name}: {len(unmatched_issues)} Figma section(s) "
            "unmatched — flagged as needs_recheck"
        )

    logger.info(
        f"Job {job_id} / {page_name}: section pipeline — "
        f"{len(figma_sections)} Figma sections, {len(live_with_screenshots)} live sections, "
        f"{len(matched_pairs)} pairs"
    )

    # Compare each matched pair
    compare_results: list[SectionCompareResult] = []
    for fsec, lsec, conf in matched_pairs:
        result = await run_in_threadpool(
            compare_section_pair, fsec, lsec, conf, diff_threshold
        )
        compare_results.append(result)

    significant = [r for r in compare_results if r.has_significant_diff]
    logger.info(
        f"Job {job_id} / {page_name}: {len(significant)}/{len(compare_results)} "
        "section pairs have significant differences"
    )

    if not significant:
        logger.info(f"No significant section diffs for {page_name} — returning empty issues")
        return unmatched_issues

    # AI analysis on significant section pairs
    issues = await run_in_threadpool(
        analyze_section_pairs, significant, page_name, job_id
    )

    # Annotate issues with their DOM section for the fix engine
    issues = _annotate_sections(issues, dom_sections)
    return issues + unmatched_issues


def _annotate_sections(issues: list[dict], dom_sections: list[dict]) -> list[dict]:
    """
    For each issue, add 'dom_section' key with the matching DOM section metadata.
    Issues from the section pipeline have x=y=0, so fall back to best heading match.
    """
    for issue in issues:
        region_y = issue.get("y", 0)
        region_h = issue.get("height", 0)

        # If section_name was set by analyze_section_pairs, find the matching DOM section
        section_name = issue.get("section_name", "")
        sec = None
        if section_name and dom_sections:
            for ds in dom_sections:
                label = " ".join(filter(None, [
                    ds.get("heading", ""), ds.get("classes", ""), ds.get("id", ""),
                ]))
                if section_name.lower() in label.lower() or label.lower() in section_name.lower():
                    sec = ds
                    break

        # Fall back to positional overlap
        if sec is None and region_h > 0:
            sec = find_section_for_region(region_y, region_h, dom_sections)

        if sec:
            issue["dom_section"] = {
                "tag":         sec.get("tag", ""),
                "id":          sec.get("id", ""),
                "classes":     sec.get("classes", ""),
                "heading":     sec.get("heading", ""),
                "sectionType": sec.get("sectionType", ""),
            }
    return issues


# A Figma frame's own absoluteBoundingBox width is often a design max-width
# content container, not the real device viewport it's meant to be viewed in —
# testing the live site at that exact width can land right on a CSS breakpoint
# edge the design never intended to be tested at, producing a spacing/dimension
# "diff" that's really just a breakpoint artifact (confirmed against a real
# report: a 36px "spacing" miss at a 1439px frame width that fully disappeared
# when checked at a normal 1920px desktop width). Cross-validating flagged
# geometry issues against a second, standard desktop width catches these before
# they reach the report as false positives.
_REFERENCE_DESKTOP_WIDTH = 1920
_REFERENCE_WIDTH_FALLBACK = 1440  # used when the frame itself is already ~1920


async def _crossvalidate_geometry_issues(
    job_id: str,
    shopify_url: str,
    shopify_password: str | None,
    page_name: str,
    target_section: str,
    frame_width: int,
    figma_nodes_in_section: list[dict],
    figma_image_nodes: list[dict],
    figma_button_nodes: list[dict],
    spacing_issues: list[dict],
    dimension_issues: list[dict],
    neighbor_figma_boxes: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Re-capture the live page at a standard reference desktop width and re-run
    the same spacing/dimension checks. A flagged gap that only reproduces at
    the Figma frame's own (possibly non-representative) width, and vanishes at
    a normal desktop width, gets downgraded to a non-defect "expected_variance"
    finding — still visible in the report's needs-recheck bucket, but not
    counted as a real design mismatch.
    """

    if not (spacing_issues or dimension_issues):
        return spacing_issues, dimension_issues

    reference_width = (
        _REFERENCE_WIDTH_FALLBACK if abs(frame_width - _REFERENCE_DESKTOP_WIDTH) < 40
        else _REFERENCE_DESKTOP_WIDTH
    )
    if reference_width == frame_width:
        return spacing_issues, dimension_issues

    try:
        ref_pages = await run_in_threadpool(
            capture_pages, shopify_url, [page_name], shopify_password, 45000, job_id,
            {page_name: reference_width},
        )
        ref_page = ref_pages[0]
        if ref_page.get("error") or not ref_page.get("screenshot"):
            return spacing_issues, dimension_issues

        # Normalize into the SAME coordinate space as figma_nodes_in_section
        # (the Figma frame's own width) — not the reference capture's own
        # width. Both captures must land in one shared coordinate space or
        # every gap comparison below is comparing apples to oranges.
        ref_original_width = ref_page.get("viewport_width") or reference_width
        ref_dom_sections = normalize_section_coords(ref_page.get("sections", []), ref_original_width, frame_width)
        ref_text_nodes = normalize_section_coords(ref_page.get("text_nodes", []), ref_original_width, frame_width)
        ref_image_elements = normalize_section_coords(ref_page.get("image_elements", []), ref_original_width, frame_width)
        ref_button_elements = normalize_section_coords(ref_page.get("button_elements", []), ref_original_width, frame_width)

        ref_section = find_live_section(target_section, ref_dom_sections)
        if not ref_section:
            return spacing_issues, dimension_issues

        ref_y0 = ref_section["y"]
        ref_y1 = ref_section["y"] + ref_section["height"]
        ref_text_in_section = filter_nodes_in_region(ref_text_nodes, ref_y0, ref_y1)
        ref_image_in_section = filter_nodes_in_region(ref_image_elements, ref_y0, ref_y1)
        ref_button_in_section = filter_nodes_in_region(ref_button_elements, ref_y0, ref_y1)

        ref_dimension_issues = (
            compare_dimensions(figma_image_nodes, ref_image_in_section, "label", "image")
            + compare_dimensions(figma_button_nodes, ref_button_in_section, "label", "button")
        )
        # Same full-page-neighbor-pool fix as the primary comparison — without
        # it, this cross-check has its own section-boundary clipping bug and
        # can wrongly "confirm" a bogus gap simply because both runs clipped
        # out the same true neighbor.
        ref_spacing_issues = compare_all_spacing(
            figma_nodes_in_section, ref_text_in_section,
            figma_image_nodes, ref_image_in_section,
            figma_button_nodes, ref_button_in_section,
            neighbor_figma_boxes=neighbor_figma_boxes,
            neighbor_live_boxes=ref_text_nodes + ref_image_elements + ref_button_elements,
        )
        ref_elements = {i["element"] for i in ref_dimension_issues + ref_spacing_issues}

        def _demote_unreproduced(issues: list[dict]) -> list[dict]:
            out = []
            for issue in issues:
                if issue["element"] in ref_elements:
                    out.append(issue)
                    continue
                demoted = dict(issue)
                demoted["issue_type"] = "expected_variance"
                demoted["description"] = (
                    issue["description"] + f" (Only observed at the Figma frame's own "
                    f"{frame_width}px width — did not reproduce at a standard "
                    f"{reference_width}px desktop viewport, so this is likely a CSS "
                    f"breakpoint artifact rather than a real design mismatch; worth a "
                    f"manual check rather than an automatic fix.)"
                )
                out.append(demoted)
            return out

        return _demote_unreproduced(spacing_issues), _demote_unreproduced(dimension_issues)

    except Exception as exc:
        logger.warning(f"Job {job_id}: reference-width cross-validation failed: {exc}")
        return spacing_issues, dimension_issues


def _best_frame_for(page_name: str, figma_frames: list[dict]) -> dict:
    """
    Find the Figma frame whose name best matches a page name.
    Falls back to the first frame when no name match is found.
    """
    name = page_name.lower()
    for frame in figma_frames:
        frame_name = frame["name"].lower()
        if name in frame_name or frame_name in name:
            return frame
    return figma_frames[0]


def _frame_width(frame: dict) -> int:
    """Frame's own logical width, rounded to an int. Falls back when unavailable."""
    w = frame.get("width")
    if not w or w <= 0:
        return _DEFAULT_CANONICAL_WIDTH
    return int(round(w))


def _match_pages_to_frames(
    shopify_pages: list[dict],
    figma_frames: list[dict],
) -> list[tuple[dict, dict]]:
    """Pair each Shopify page to the best-matching Figma frame by name."""
    return [(page, _best_frame_for(page["page"], figma_frames)) for page in shopify_pages]


async def _progress(job_id: str, status: str, message: str) -> None:
    update_vqa_job(job_id, status=status, progress=message)
    logger.info(f"Job {job_id} [{status}]: {message}")


async def _fail(job_id: str, reason: str) -> None:
    update_vqa_job(job_id, status="failed", error=reason, progress="Failed")
    logger.error(f"Job {job_id} failed: {reason}")