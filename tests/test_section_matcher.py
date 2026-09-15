"""Tests for visual/section_matcher.py — Figma↔DOM section pairing."""
import pytest

from visual.section_matcher import match_sections, _detect_type, _similarity, _score_pair


class TestDetectType:
    def test_hero(self):
        assert _detect_type("hero-banner slideshow") == "hero"

    def test_nav(self):
        assert _detect_type("main-navigation header") == "nav"

    def test_footer(self):
        assert _detect_type("site-footer copyright") == "footer"

    def test_product(self):
        assert _detect_type("product-detail add-to-cart") == "product"

    def test_unknown(self):
        assert _detect_type("some-random-class-name") == "unknown"

    def test_case_insensitive(self):
        assert _detect_type("HERO Section") == "hero"


class TestSimilarity:
    def test_identical(self):
        assert _similarity("hero", "hero") == 1.0

    def test_empty_strings(self):
        assert _similarity("", "anything") == 0.0
        assert _similarity("anything", "") == 0.0

    def test_partial_match(self):
        score = _similarity("hero banner", "hero section")
        assert 0.0 < score < 1.0

    def test_completely_different(self):
        score = _similarity("aaaa", "bbbb")
        assert score == 0.0


class TestMatchSections:
    def _figma(self, name):
        return {"name": name, "rel_x": 0, "rel_y": 0, "width": 1440, "height": 300,
                "image_bytes": b"\xff" * 100}

    def _live(self, heading="", classes="", section_type="", screenshot=b"\xff" * 100):
        return {
            "heading": heading, "classes": classes, "sectionType": section_type,
            "id": "", "screenshot": screenshot,
            "index": 0, "tag": "div", "textSnippet": "",
            "x": 0, "y": 0, "width": 1440, "height": 300,
        }

    def test_empty_figma_returns_empty(self):
        live = [self._live("Hero", "hero-section")]
        assert match_sections([], live) == []

    def test_empty_live_returns_empty(self):
        figma = [self._figma("Hero")]
        assert match_sections(figma, []) == []

    def test_single_pair_matched(self):
        figma = [self._figma("Hero Banner")]
        live = [self._live("Welcome", "hero-banner", "hero")]
        result = match_sections(figma, live)
        assert len(result) == 1
        fsec, lsec, conf = result[0]
        assert fsec["name"] == "Hero Banner"
        assert conf > 0

    def test_each_live_section_used_at_most_once(self):
        figma = [self._figma("Hero"), self._figma("Footer")]
        live = [self._live("Hero", "hero", "hero")]
        result = match_sections(figma, live)
        live_names = [r[1]["heading"] for r in result]
        assert len(live_names) == len(set(live_names))

    def test_confidence_between_0_and_100(self):
        figma = [self._figma("Nav Menu")]
        live = [self._live("Navigation", "nav-bar", "nav")]
        result = match_sections(figma, live)
        if result:
            _, _, conf = result[0]
            assert 0 <= conf <= 100

    def test_below_min_confidence_excluded(self):
        # A Figma section with a completely unrelated name vs a live section
        figma = [self._figma("ZZZZZ_NOMATCH_12345")]
        live = [self._live("Hero", "hero", "hero")]
        # With very high min_confidence they won't match
        result = match_sections(figma, live, min_confidence=99.0)
        assert result == []
