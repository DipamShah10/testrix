import asyncio
import logging

from ai_engine.llm import ask_ai
from ai_engine.utils import extract_json_object

logger = logging.getLogger(__name__)

_FIX_PROMPT = """You are a senior frontend engineer and Shopify theme developer reviewing a visual QA issue found by comparing a Figma design against a live Shopify storefront.

Issue details:
- Page: {page_name}
- Element: {element}
- Issue type: {issue_type}
- Severity: {severity}
- Description: {description}
- User impact: {user_impact}
- Pixel diff: {diff_percent}% of region differs from design
- Initial fix hint: {suggested_fix}
- Live URL: {shopify_url}

Your job is to produce a developer-ready fix guide that a frontend developer can apply immediately.

Assume the storefront may be using:
- Shopify Dawn theme (Liquid + CSS custom properties)
- Custom Shopify sections with schema JSON
- Tailwind CSS utility classes
- Vanilla CSS with BEM or flat class names
- Possibly a headless Shopify with React/Next.js

Rules:
- Name REAL CSS properties and VALUES (e.g. "change padding-top from 32px to 64px" not "update padding")
- Name REAL Shopify section or component names that are likely involved (e.g. main-hero__content, product-card__image)
- For typography issues: specify exact font-size, font-weight, letter-spacing, line-height values from the Figma spec
- For layout issues: specify container max-width, flex/grid properties, gap values
- For color issues: provide the hex value from Figma
- For responsive issues: specify the @media breakpoint and the property override
- fix_complexity: Easy = CSS-only change < 30 min; Medium = multiple files or theme overrides 30 min–2 h; Complex = structural change > 2 h
- estimated_effort: give a human time range like "~20 minutes", "~1 hour", "~half a day"
- affected_devices: list from ["desktop", "tablet", "mobile"]
- impacted_components: list 2–4 Shopify section names or CSS class names (e.g. ["main-banner", ".hero__heading", "sections/image-banner.liquid"])

Output ONLY valid JSON — no markdown, no code fences, no explanation:

{{
  "root_cause": "<one technical paragraph: why did this mismatch occur in implementation terms>",
  "probable_causes": [
    "<specific developer mistake or omission — e.g. 'CSS custom property --color-base-text overridden by theme settings'>",
    "<second specific cause>",
    "<third specific cause>"
  ],
  "frontend_fix": [
    "<step 1: concrete action with actual values — e.g. 'Set .hero__heading font-size to 48px (was 36px)'>",
    "<step 2>",
    "<step 3>"
  ],
  "css_snippet": "<ready-to-paste CSS block — real selectors and values>",
  "responsive_fix": "<@media breakpoint and specific overrides for mobile/tablet — real values>",
  "severity_reason": "<explain why this severity level was chosen and what real user/business impact it has>",
  "fix_complexity": "<Easy | Medium | Complex>",
  "estimated_effort": "<human time estimate>",
  "affected_devices": ["desktop", "mobile"],
  "impacted_components": ["<section or class name 1>", "<section or class name 2>"]
}}
"""


async def generate_fix_recommendation(
    issue: dict,
    page_name: str,
    shopify_url: str = "",
) -> dict:
    """Enrich a single issue dict with 'fix_recommendation' from AI."""
    # Include DOM section context when available (injected by visual_qa_agent)
    dom_sec = issue.get("dom_section", {})
    dom_context = ""
    if dom_sec:
        parts = []
        if dom_sec.get("sectionType"): parts.append(f"section-type={dom_sec['sectionType']}")
        if dom_sec.get("heading"):     parts.append(f"heading='{dom_sec['heading']}'")
        if dom_sec.get("classes"):     parts.append(f"classes='{dom_sec['classes'][:80]}'")
        if dom_sec.get("id"):          parts.append(f"id='{dom_sec['id']}'")
        if parts:
            dom_context = "\n- DOM section context: " + ", ".join(parts)

    prompt = _FIX_PROMPT.format(
        page_name=page_name,
        element=issue.get("element", "Unknown element"),
        issue_type=issue.get("issue_type", "other"),
        severity=issue.get("severity", "Medium"),
        description=(issue.get("description", "")[:400] + dom_context),
        user_impact=issue.get("user_impact", "")[:200],
        diff_percent=issue.get("diff_percent", 0),
        suggested_fix=issue.get("suggested_fix", "No hint available")[:300],
        shopify_url=shopify_url or "Shopify storefront",
    )

    try:
        raw = await ask_ai(prompt)
        fix_data = extract_json_object(raw)
        if fix_data:
            issue["fix_recommendation"] = fix_data
            logger.info(
                f"Fix recommendation generated — element='{issue.get('element')}', "
                f"complexity={fix_data.get('fix_complexity', '?')}"
            )
        else:
            logger.warning(
                f"Fix recommendation parse failed for '{issue.get('element')}' — skipping"
            )
    except Exception as e:
        logger.warning(f"Fix recommendation failed for '{issue.get('element')}': {e}")

    return issue


# Cap concurrent fix-recommendation calls — firing one per issue at once (a
# page can easily have 20-30) bursts straight through Groq's per-minute token
# cap, and ask_ai's retry-with-backoff can only smooth over occasional 429s,
# not a stampede of them all landing in the same instant.
_MAX_CONCURRENT_FIXES = 4


async def generate_fix_recommendations(
    issues: list[dict],
    page_name: str,
    shopify_url: str = "",
) -> list[dict]:
    """Generate fix recommendations for all issues on a page, throttled to
    avoid tripping the LLM's rate limit."""
    if not issues:
        return issues
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_FIXES)

    async def _bounded(issue: dict) -> dict:
        async with semaphore:
            return await generate_fix_recommendation(issue, page_name, shopify_url)

    results = await asyncio.gather(
        *[_bounded(issue) for issue in issues],
        return_exceptions=False,
    )
    return list(results)