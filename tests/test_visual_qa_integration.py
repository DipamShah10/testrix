"""
Integration test for the Visual QA pipeline.

Uses synthetic images and calls the real Groq API.
MongoDB is mocked so no Atlas connection is needed.

Run:  pytest tests/test_visual_qa_integration.py -v -s
"""
import asyncio
import io
import logging
import os
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Image helpers ─────────────────────────────────────────────────────────────

def _make_figma_image(width=1440, height=2400) -> bytes:
    """Simulate a Figma frame: white background, hero, product grid, footer."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Hero section — dark blue
    draw.rectangle([0, 0, width, 600], fill=(15, 23, 42))
    draw.rectangle([120, 180, 720, 280], fill=(255, 255, 255))  # headline block
    draw.rectangle([120, 310, 420, 370], fill=(37, 99, 235))    # CTA button

    # Product grid — 3 cards
    for i in range(3):
        x = 120 + i * 420
        draw.rectangle([x, 700, x + 380, 1100], fill=(249, 250, 251), outline=(229, 231, 235), width=2)
        draw.rectangle([x, 700, x + 380, 920], fill=(209, 213, 219))   # product image
        draw.rectangle([x + 20, 940, x + 250, 970], fill=(17, 24, 39)) # title
        draw.rectangle([x + 20, 990, x + 120, 1010], fill=(37, 99, 235)) # price

    # Footer
    draw.rectangle([0, 2200, width, 2400], fill=(15, 23, 42))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_live_image_with_diffs(width=1440, height=2400) -> bytes:
    """
    Live screenshot that differs from Figma in two areas:
    - Hero CTA button is green instead of blue
    - First product card title is missing (white instead of dark)
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Hero — same layout
    draw.rectangle([0, 0, width, 600], fill=(15, 23, 42))
    draw.rectangle([120, 180, 720, 280], fill=(255, 255, 255))
    # DIFF: CTA button is green instead of blue
    draw.rectangle([120, 310, 420, 370], fill=(16, 185, 129))

    # Product grid — 3 cards, first card has missing title
    for i in range(3):
        x = 120 + i * 420
        draw.rectangle([x, 700, x + 380, 1100], fill=(249, 250, 251), outline=(229, 231, 235), width=2)
        draw.rectangle([x, 700, x + 380, 920], fill=(209, 213, 219))
        if i == 0:
            # DIFF: first card title is missing (white on white)
            draw.rectangle([x + 20, 940, x + 250, 970], fill=(255, 255, 255))
        else:
            draw.rectangle([x + 20, 940, x + 250, 970], fill=(17, 24, 39))
        draw.rectangle([x + 20, 990, x + 120, 1010], fill=(37, 99, 235))

    # Footer
    draw.rectangle([0, 2200, width, 2400], fill=(15, 23, 42))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Step 1: Visual Comparator ─────────────────────────────────────────────────

class TestVisualComparatorStep:
    def test_detects_diffs_between_figma_and_live(self):
        from services.visual_comparator import compare

        figma = _make_figma_image()
        live  = _make_live_image_with_diffs()
        result = compare(figma, live, diff_threshold=0.02)

        logger.info(f"[comparator] diff={result.diff_percent}%, regions={len(result.regions)}")
        assert result.diff_percent > 0, "Should detect pixel differences"
        assert len(result.regions) >= 1, "Should find at least 1 diff region"
        for r in result.regions:
            assert r.width > 0 and r.height > 0
            assert 0.0 <= r.diff_percent <= 100.0
        logger.info(f"[comparator] regions: {[(r.x, r.y, r.width, r.height) for r in result.regions]}")


# ── Step 2: Visual AI Analyzer ────────────────────────────────────────────────

@pytest.mark.live_api
class TestVisualAIAnalyzerStep:
    def test_ai_analyzes_diff_regions(self):
        from services.visual_comparator import compare
        from services.visual_ai_analyzer import analyze

        figma = _make_figma_image()
        live  = _make_live_image_with_diffs()
        compare_result = compare(figma, live, diff_threshold=0.02)

        if not compare_result.regions:
            pytest.skip("No diff regions found — comparator step failed")

        figma_typo = {
            "fonts": ["Inter", "Helvetica Neue"],
            "sizes": [14, 16, 24, 48],
            "weights": [400, 600, 700],
            "colors": ["rgb(15,23,42)", "rgb(37,99,235)", "rgb(255,255,255)"],
        }

        logger.info(f"[AI analyzer] sending {len(compare_result.regions)} region(s) to Groq vision...")
        issues = analyze(
            figma_bytes=figma,
            live_bytes=live,
            compare_result=compare_result,
            page_name="home",
            job_id="",
            figma_typography=figma_typo,
        )

        logger.info(f"[AI analyzer] got {len(issues)} issue(s)")
        assert isinstance(issues, list)
        assert len(issues) >= 1, "AI should return at least 1 issue"

        for issue in issues:
            logger.info(f"  - [{issue.get('issue_type','?')}] {issue.get('element','?')}: {issue.get('description','')[:80]}")
            assert "element" in issue
            assert "description" in issue
            assert "issue_type" in issue
            # Full valid set per the vision model's own prompt (visual_ai_analyzer.py) —
            # capture_failure/expected_variance are legitimate, expected outputs (a
            # blank/loading-placeholder region, or rotating carousel/UGC content),
            # not error states. Omitting them here was a stale assertion that made
            # this test flaky: which issue_type comes back depends on the live
            # Groq vision call's non-deterministic read of the synthetic test image.
            assert issue["issue_type"] in (
                "typography", "layout", "color", "spacing", "image",
                "capture_failure", "expected_variance", "other",
            )


# ── Step 3: Severity Classifier ───────────────────────────────────────────────

@pytest.mark.live_api
class TestSeverityClassifierStep:
    @pytest.mark.asyncio
    async def test_classifies_issues_from_ai(self):
        from services.visual_comparator import compare
        from services.visual_ai_analyzer import analyze
        from services.severity_classifier import classify_all

        figma = _make_figma_image()
        live  = _make_live_image_with_diffs()
        compare_result = compare(figma, live, diff_threshold=0.02)

        if not compare_result.regions:
            pytest.skip("No diff regions found")

        issues = analyze(
            figma_bytes=figma,
            live_bytes=live,
            compare_result=compare_result,
            page_name="home",
            job_id="",
        )

        if not issues:
            pytest.skip("AI analyzer returned no issues")

        logger.info(f"[severity] classifying {len(issues)} issue(s)...")
        classified = await classify_all(issues)

        VALID = {"Critical", "High", "Medium", "Low"}
        for issue in classified:
            logger.info(f"  - {issue.get('severity','?')} — {issue.get('element','?')}")
            assert issue["severity"] in VALID
            assert issue.get("rule_severity") in VALID

        # Should be sorted Critical first
        severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        severities = [i["severity"] for i in classified]
        assert severities == sorted(severities, key=lambda s: severity_order[s])

        return classified


# ── Step 4: Fix Recommendation Engine ────────────────────────────────────────

@pytest.mark.live_api
class TestFixRecommendationStep:
    @pytest.mark.asyncio
    async def test_generates_fix_recommendations(self):
        from services.visual_comparator import compare
        from services.visual_ai_analyzer import analyze
        from services.severity_classifier import classify_all
        from qa.fix_recommendation_engine import generate_fix_recommendations

        figma = _make_figma_image()
        live  = _make_live_image_with_diffs()
        compare_result = compare(figma, live, diff_threshold=0.02)

        if not compare_result.regions:
            pytest.skip("No diff regions found")

        issues = analyze(
            figma_bytes=figma,
            live_bytes=live,
            compare_result=compare_result,
            page_name="home",
            job_id="",
        )

        if not issues:
            pytest.skip("AI analyzer returned no issues")

        classified = await classify_all(issues)

        logger.info(f"[fix engine] generating recommendations for {len(classified)} issue(s)...")
        enriched = await generate_fix_recommendations(classified, "home", "https://test-store.myshopify.com")

        has_fix = [i for i in enriched if "fix_recommendation" in i]
        logger.info(f"[fix engine] {len(has_fix)}/{len(enriched)} issues got fix recommendations")
        assert len(has_fix) >= 1, "At least one issue should have a fix recommendation"

        for issue in has_fix:
            fr = issue["fix_recommendation"]
            logger.info(f"  - [{fr.get('fix_complexity','?')}] {fr.get('estimated_effort','?')} — {issue.get('element','?')}")
            assert "root_cause" in fr
            assert "frontend_fix" in fr
            assert isinstance(fr["frontend_fix"], list)
            assert fr.get("fix_complexity") in ("Easy", "Medium", "Complex")


# ── Step 5: Full pipeline (run_visual_qa) with mocked Figma + Shopify + MongoDB

@pytest.mark.live_api
class TestFullVisualQAPipeline:
    @pytest.mark.asyncio
    async def test_run_visual_qa_end_to_end(self):
        """
        Runs the complete run_visual_qa() pipeline with:
        - Real Groq API calls (vision + text LLM)
        - Mocked Figma extractor (returns synthetic image)
        - Mocked Shopify scraper (returns synthetic screenshot + DOM sections)
        - Mocked MongoDB (in-memory dict)
        """
        from agents.visual_qa_agent import run_visual_qa

        figma_bytes = _make_figma_image()
        live_bytes  = _make_live_image_with_diffs()

        mock_figma_frames = [{
            "name": "Home",
            "node_id": "1:1",
            "image_bytes": figma_bytes,
            "width": 1440,
            "height": 2400,
            "sections": [
                {"name": "Hero", "type": "FRAME", "rel_x": 0, "rel_y": 0,   "width": 1440, "height": 600},
                {"name": "Products", "type": "FRAME", "rel_x": 0, "rel_y": 600, "width": 1440, "height": 500},
                {"name": "Footer", "type": "FRAME", "rel_x": 0, "rel_y": 2200, "width": 1440, "height": 200},
            ],
        }]
        mock_typography = {
            "fonts": ["Inter"],
            "sizes": [14, 16, 24, 48],
            "weights": [400, 600, 700],
            "colors": ["rgb(15,23,42)"],
        }
        mock_shopify_pages = [{
            "page": "home",
            "url": "https://test-store.myshopify.com/",
            "screenshot": live_bytes,
            "sections": [
                {
                    "index": 0, "tag": "div", "id": "shopify-section-hero",
                    "classes": "shopify-section hero-section", "sectionType": "hero",
                    "heading": "Welcome to our store", "textSnippet": "Shop now",
                    "x": 0, "y": 0, "width": 1440, "height": 600,
                    "screenshot": _crop_bytes(live_bytes, 0, 0, 1440, 600),
                },
                {
                    "index": 1, "tag": "div", "id": "shopify-section-products",
                    "classes": "shopify-section product-grid", "sectionType": "collection",
                    "heading": "Featured Products", "textSnippet": "Browse our products",
                    "x": 0, "y": 600, "width": 1440, "height": 500,
                    "screenshot": _crop_bytes(live_bytes, 0, 600, 1440, 500),
                },
            ],
            "error": None,
        }]

        job_state = {"status": "pending", "progress": "", "result": None, "error": None}

        def mock_update_vqa_job(job_id, **fields):
            job_state.update(fields)
            logger.info(f"  [db] {fields.get('status','?')}: {fields.get('progress','')}")

        # Patch where visual_qa_agent imported from (not the source module)
        with patch("agents.visual_qa_agent.extract_frames",
                   return_value=(mock_figma_frames, mock_typography)), \
             patch("agents.visual_qa_agent.capture_pages",
                   return_value=mock_shopify_pages), \
             patch("agents.visual_qa_agent.update_vqa_job", side_effect=mock_update_vqa_job), \
             patch("services.bug_report_generator.update_vqa_job", side_effect=mock_update_vqa_job):

            logger.info("[pipeline] starting run_visual_qa()...")
            report = await run_visual_qa(
                job_id="test-job-001",
                shopify_url="https://test-store.myshopify.com",
                figma_url="https://www.figma.com/design/TESTKEY/Test",
                pages=["home"],
                diff_threshold=0.02,
            )

        logger.info(f"[pipeline] overall_severity={report['overall_severity']}, total_issues={report['total_issues']}")

        # Report structure
        assert "overall_severity" in report
        assert "total_issues" in report
        assert "page_reports" in report
        assert len(report["page_reports"]) == 1
        assert report["overall_severity"] in ("Critical", "High", "Medium", "Low", "Pass")
        assert report["total_issues"] >= 0

        # Page report structure
        page = report["page_reports"][0]
        assert page["page"] == "home"
        assert "issues" in page
        assert "diff_percent" in page
        assert "overall_severity" in page

        # Issues (if any) have required fields
        for issue in page["issues"]:
            assert "element" in issue
            assert "description" in issue
            assert "severity" in issue
            logger.info(f"  [{issue['severity']}] {issue['element']}: {issue['description'][:60]}")

        logger.info(f"[pipeline] PASSED — {report['total_issues']} issue(s) found, severity={report['overall_severity']}")
        assert job_state["status"] == "complete"


def _crop_bytes(img_bytes: bytes, x: int, y: int, w: int, h: int) -> bytes:
    """Crop a region from PNG bytes and return as PNG bytes."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    cropped = img.crop((x, y, x + w, y + h))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()
