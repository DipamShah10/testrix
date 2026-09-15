"""Tests for services/severity_classifier.py — rule-based and LLM paths."""
import pytest
from unittest.mock import AsyncMock, patch

from services.severity_classifier import _rule_based, _parse_severity, classify_issue, classify_all, SEVERITIES


class TestRuleBased:
    def test_critical_nav_element(self):
        assert _rule_based("navigation bar", 1.0) == "Critical"

    def test_critical_checkout(self):
        assert _rule_based("checkout button", 50.0) == "Critical"

    def test_critical_cart(self):
        assert _rule_based("cart icon", 5.0) == "Critical"

    def test_high_hero_with_large_diff(self):
        assert _rule_based("hero banner", 10.0) == "High"

    def test_medium_hero_with_small_diff(self):
        assert _rule_based("hero section", 2.0) == "Medium"

    def test_medium_typography(self):
        assert _rule_based("body text paragraph", 3.0) == "Medium"

    def test_low_typography_tiny_diff(self):
        assert _rule_based("caption text", 0.5) == "Low"

    def test_low_footer(self):
        assert _rule_based("footer links", 20.0) == "Low"

    def test_low_shadow(self):
        assert _rule_based("drop shadow", 5.0) == "Low"

    def test_fallback_critical_by_diff(self):
        assert _rule_based("unknown widget", 35.0) == "Critical"

    def test_fallback_high_by_diff(self):
        assert _rule_based("unknown widget", 20.0) == "High"

    def test_fallback_medium_by_diff(self):
        assert _rule_based("unknown widget", 7.0) == "Medium"

    def test_fallback_low_by_diff(self):
        assert _rule_based("unknown widget", 1.0) == "Low"


class TestParseSeverity:
    def test_exact_match(self):
        assert _parse_severity("Critical", "Low") == "Critical"

    def test_extra_whitespace(self):
        assert _parse_severity("  High  ", "Low") == "High"

    def test_lowercase_in_text(self):
        assert _parse_severity("the severity is medium for this issue", "Low") == "Medium"

    def test_unknown_returns_fallback(self):
        assert _parse_severity("not sure", "Medium") == "Medium"

    def test_empty_returns_fallback(self):
        assert _parse_severity("", "High") == "High"


class TestClassifyIssue:
    @pytest.mark.asyncio
    async def test_critical_skips_llm(self):
        issue = {
            "element": "checkout button",
            "issue_type": "layout",
            "description": "checkout misaligned",
            "user_impact": "users cannot checkout",
            "diff_percent": 20.0,
        }
        with patch("services.severity_classifier.ask_ai") as mock_ai:
            result = await classify_issue(issue)
        mock_ai.assert_not_called()
        assert result["severity"] == "Critical"

    @pytest.mark.asyncio
    async def test_llm_called_for_medium(self):
        # "heading text" matches _MEDIUM_KEYWORDS ("text") + diff 3% >= 2% → rule = Medium
        # Medium is not in the LLM-skip set (Critical, Low), so ask_ai must be called.
        issue = {
            "element": "heading text",
            "issue_type": "typography",
            "description": "font size differs",
            "user_impact": "readability",
            "diff_percent": 3.0,
        }
        with patch("services.severity_classifier.ask_ai", new_callable=AsyncMock, return_value="High") as mock_ai:
            result = await classify_issue(issue)
        mock_ai.assert_called_once()
        assert result["severity"] == "High"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_rule(self):
        issue = {
            "element": "hero image",
            "issue_type": "image",
            "description": "image missing",
            "user_impact": "brand impact",
            "diff_percent": 8.0,
        }
        with patch("services.severity_classifier.ask_ai", new_callable=AsyncMock, side_effect=Exception("API error")):
            result = await classify_issue(issue)
        assert result["severity"] in SEVERITIES


class TestClassifyAll:
    @pytest.mark.asyncio
    async def test_sorted_critical_first(self):
        issues = [
            {"element": "footer", "issue_type": "layout", "description": "", "user_impact": "", "diff_percent": 2.0},
            {"element": "checkout", "issue_type": "layout", "description": "", "user_impact": "", "diff_percent": 5.0},
            {"element": "hero banner", "issue_type": "layout", "description": "", "user_impact": "", "diff_percent": 10.0},
        ]
        with patch("services.severity_classifier.ask_ai", new_callable=AsyncMock, return_value="Medium"):
            result = await classify_all(issues)

        order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        severities = [r["severity"] for r in result]
        assert severities == sorted(severities, key=lambda s: order.get(s, 99))
