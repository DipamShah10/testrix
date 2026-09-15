import argparse
import asyncio
import json
import logging
from dataclasses import asdict

from agents.ai_reviewer import AIReviewer
from agents.llm_client import LLMClient
from agents.requirement_analyzer import RequirementAnalyzer
from browser.browser_agent import BrowserAgent
from config.settings import get_settings
from figma.figma_analyzer import FigmaAnalyzer
from qa.qa_engine import QAEngine
from reports.report_generator import ReportGenerator
from utils.logger import configure_logging


async def run_pipeline(
    requirements_path: str,
    target_url: str,
    figma_url: str | None,
    pages: list[str],
    visual_diff_threshold: float | None = None,
    strict_accessibility: bool | None = None,
    visual_diff_page_thresholds: dict[str, float] | None = None,
) -> dict:
    settings = get_settings()
    llm_client = LLMClient(settings)

    requirement_analyzer = RequirementAnalyzer(llm_client)
    figma_analyzer = FigmaAnalyzer(settings.figma_api_token) if figma_url else None
    browser_agent = BrowserAgent(
        headless=settings.browser_headless,
        timeout_ms=settings.browser_timeout_ms,
        screenshot_dir=settings.screenshot_dir,
    )
    qa_engine = QAEngine()
    ai_reviewer = AIReviewer(llm_client)
    report_generator = ReportGenerator(settings.report_dir)

    requirements = await requirement_analyzer.analyze(requirements_path)
    figma = await figma_analyzer.analyze(figma_url) if figma_analyzer and figma_url else None
    observations = await browser_agent.inspect(target_url, pages)
    qa_result = qa_engine.evaluate(
        requirements,
        figma,
        observations,
        visual_diff_threshold=visual_diff_threshold if visual_diff_threshold is not None else settings.visual_diff_threshold,
        page_threshold_overrides=visual_diff_page_thresholds or settings.visual_diff_page_thresholds,
    )
    strict_mode = strict_accessibility if strict_accessibility is not None else settings.strict_accessibility
    ai_review = await ai_reviewer.review(requirements, figma, qa_result, strict_accessibility=strict_mode)
    reports = report_generator.generate_all(requirements, figma, observations, qa_result, ai_review)

    return {
        "requirements": asdict(requirements),
        "figma": asdict(figma) if figma else None,
        "observations": [asdict(item) for item in observations],
        "qa_result": {
            "requirement_coverage": qa_result.requirement_coverage,
            "missing_features": qa_result.missing_features,
            "findings": [asdict(f) for f in qa_result.findings],
            "accessibility_gaps": qa_result.accessibility_gaps,
            "responsive_issues": qa_result.responsive_issues,
            "visual_mismatch_scores": qa_result.visual_mismatch_scores,
            "accessibility_blocker_count": qa_result.accessibility_blocker_count,
        },
        "ai_review": ai_review,
        "reports": reports,
    }


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Autonomous AI Frontend QA Agent")
    parser.add_argument("--requirements", required=False, default=settings.requirements_path)
    parser.add_argument("--url", required=False, default=settings.target_url)
    parser.add_argument("--figma-url", required=False, default=settings.figma_url)
    parser.add_argument("--pages", nargs="+", default=["/", "/products", "/collections/all", "/cart"])
    parser.add_argument(
        "--discover-pages",
        action="store_true",
        help="Crawl from --url (sitemap + same-host BFS) and use discovered paths instead of --pages",
    )
    parser.add_argument("--crawl-max-pages", type=int, default=50, help="With --discover-pages: max URLs to collect")
    parser.add_argument("--crawl-max-depth", type=int, default=2, help="With --discover-pages: max link depth from seed/sitemap pages")
    parser.add_argument(
        "--crawl-no-persist",
        action="store_true",
        help="With --discover-pages: skip writing artifacts/crawl/latest.json",
    )
    parser.add_argument("--visual-diff-threshold", type=float, default=settings.visual_diff_threshold)
    parser.add_argument(
        "--page-visual-threshold",
        action="append",
        default=[],
        help="Override per page as route=threshold (repeatable), e.g. /cart=0.15",
    )
    parser.add_argument("--strict-accessibility", action="store_true", default=settings.strict_accessibility)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _parse_page_threshold_args(items: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            result[key] = float(raw.strip())
        except ValueError:
            continue
    return result


def main() -> None:
    args = parse_args()
    if not args.requirements:
        raise ValueError("Requirements path is required. Pass --requirements or set REQUIREMENTS_PATH.")
    if not args.url:
        raise ValueError("Target URL is required. Pass --url or set TARGET_URL.")
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

    configure_logging(args.log_level)
    log = logging.getLogger(__name__)

    pages: list[str] = list(args.pages)
    if args.discover_pages:
        from services.site_crawler import discover_site

        graph = discover_site(
            args.url,
            max_pages=args.crawl_max_pages,
            max_depth=args.crawl_max_depth,
            persist=not args.crawl_no_persist,
        )
        pages = graph["routes_for_pipeline"]
        log.info("Discovered %d route(s) for pipeline (persist=%s)", len(pages), graph.get("persist_path", "no"))

    result = asyncio.run(
        run_pipeline(
            requirements_path=args.requirements,
            target_url=args.url,
            figma_url=args.figma_url,
            pages=pages,
            visual_diff_threshold=args.visual_diff_threshold,
            strict_accessibility=args.strict_accessibility,
            visual_diff_page_thresholds=_parse_page_threshold_args(args.page_visual_threshold),
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
