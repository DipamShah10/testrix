"""Tests for qa/fix_recommendation_engine.py — AI fix generation with mocked LLM."""
import pytest
from unittest.mock import AsyncMock, patch

from qa.fix_recommendation_engine import generate_fix_recommendation, generate_fix_recommendations


_MOCK_FIX_JSON = """{
  "root_cause": "CSS custom property override missing",
  "probable_causes": ["Theme default overrides custom CSS", "Missing media query"],
  "frontend_fix": ["Set font-size to 48px on .hero__heading", "Add line-height 1.2"],
  "css_snippet": ".hero__heading { font-size: 48px; }",
  "responsive_fix": "@media (max-width: 768px) { .hero__heading { font-size: 32px; } }",
  "severity_reason": "Typography mismatch affects brand identity above the fold",
  "fix_complexity": "Easy",
  "estimated_effort": "~15 minutes",
  "affected_devices": ["desktop", "mobile"],
  "impacted_components": ["main-hero__content", ".hero__heading"]
}"""


class TestGenerateFixRecommendation:
    def _base_issue(self, **overrides):
        issue = {
            "element": "Hero heading",
            "issue_type": "typography",
            "severity": "High",
            "description": "Font size differs from Figma spec",
            "user_impact": "Brand inconsistency",
            "diff_percent": 12.5,
            "suggested_fix": "Match font-size to Figma token",
        }
        issue.update(overrides)
        return issue

    @pytest.mark.asyncio
    async def test_adds_fix_recommendation_key(self):
        with patch("qa.fix_recommendation_engine.ask_ai", new_callable=AsyncMock, return_value=_MOCK_FIX_JSON):
            result = await generate_fix_recommendation(self._base_issue(), "home", "https://store.myshopify.com")
        assert "fix_recommendation" in result
        fr = result["fix_recommendation"]
        assert fr["fix_complexity"] == "Easy"
        assert isinstance(fr["frontend_fix"], list)
        assert len(fr["frontend_fix"]) > 0

    @pytest.mark.asyncio
    async def test_fix_recommendation_fields_present(self):
        with patch("qa.fix_recommendation_engine.ask_ai", new_callable=AsyncMock, return_value=_MOCK_FIX_JSON):
            result = await generate_fix_recommendation(self._base_issue(), "home")
        fr = result["fix_recommendation"]
        for field in ("root_cause", "probable_causes", "frontend_fix", "css_snippet",
                      "fix_complexity", "estimated_effort", "affected_devices", "impacted_components"):
            assert field in fr, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_llm_failure_does_not_crash(self):
        with patch("qa.fix_recommendation_engine.ask_ai", new_callable=AsyncMock, side_effect=Exception("API down")):
            result = await generate_fix_recommendation(self._base_issue(), "home")
        # Issue returned unchanged (no fix_recommendation key, no exception)
        assert "fix_recommendation" not in result
        assert result["element"] == "Hero heading"

    @pytest.mark.asyncio
    async def test_dom_section_context_included_in_prompt(self):
        issue = self._base_issue(dom_section={
            "sectionType": "hero",
            "heading": "Welcome to our store",
            "classes": "main-hero shopify-section",
            "id": "hero-section",
        })
        captured_prompts = []

        async def capture(prompt):
            captured_prompts.append(prompt)
            return _MOCK_FIX_JSON

        with patch("qa.fix_recommendation_engine.ask_ai", new_callable=AsyncMock, side_effect=capture):
            await generate_fix_recommendation(issue, "home")

        assert captured_prompts
        assert "hero" in captured_prompts[0].lower() or "Welcome" in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_invalid_json_response_gracefully_handled(self):
        with patch("qa.fix_recommendation_engine.ask_ai", new_callable=AsyncMock, return_value="not json at all"):
            result = await generate_fix_recommendation(self._base_issue(), "home")
        assert "fix_recommendation" not in result


class TestGenerateFixRecommendations:
    @pytest.mark.asyncio
    async def test_processes_all_issues(self):
        issues = [
            {"element": f"Element {i}", "issue_type": "layout", "severity": "Medium",
             "description": "", "user_impact": "", "diff_percent": 5.0, "suggested_fix": ""}
            for i in range(3)
        ]
        with patch("qa.fix_recommendation_engine.ask_ai", new_callable=AsyncMock, return_value=_MOCK_FIX_JSON):
            results = await generate_fix_recommendations(issues, "home")
        assert len(results) == 3
        assert all("fix_recommendation" in r for r in results)

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        results = await generate_fix_recommendations([], "home")
        assert results == []
