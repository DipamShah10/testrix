"""Tests for visual/section_exclusion.py."""
from visual.section_exclusion import (
    normalize_exclude_types,
    excluded_regions,
    filter_out_sections,
    filter_excluded_issues,
    _overlap_fraction,
)


class TestNormalizeExcludeTypes:
    def test_maps_friendly_aliases(self):
        assert normalize_exclude_types(["header", "footer", "collection"]) == {
            "nav", "footer", "collection",
        }

    def test_case_and_whitespace_insensitive(self):
        assert normalize_exclude_types([" Header ", "FOOTER"]) == {"nav", "footer"}

    def test_unknown_type_passes_through_lowercased(self):
        assert normalize_exclude_types(["Testimonial"]) == {"testimonial"}

    def test_empty_or_none_returns_empty_set(self):
        assert normalize_exclude_types(None) == set()
        assert normalize_exclude_types([]) == set()

    def test_ignores_blank_entries(self):
        assert normalize_exclude_types(["", "  ", "footer"]) == {"footer"}


class TestExcludedRegions:
    def _dom_section(self, heading, x, y, w, h):
        return {"heading": heading, "classes": "", "id": "", "sectionType": "",
                "textSnippet": "", "x": x, "y": y, "width": w, "height": h}

    def _figma_section(self, name, rel_x, rel_y, w, h):
        return {"name": name, "rel_x": rel_x, "rel_y": rel_y, "width": w, "height": h}

    def test_no_exclude_types_returns_empty(self):
        dom = [self._dom_section("Header", 0, 0, 1440, 100)]
        assert excluded_regions(dom, [], set()) == []

    def test_matches_dom_section_by_type(self):
        dom = [
            self._dom_section("Main Header", 0, 0, 1440, 100),
            self._dom_section("Hero", 0, 100, 1440, 600),
        ]
        boxes = excluded_regions(dom, [], {"nav"})
        assert boxes == [(0, 0, 1440, 100)]

    def test_matches_figma_section_by_name(self):
        figma = [self._figma_section("Footer Links", 0, 5000, 1440, 300)]
        boxes = excluded_regions([], figma, {"footer"})
        assert boxes == [(0, 5000, 1440, 300)]

    def test_unions_dom_and_figma_matches(self):
        dom = [self._dom_section("Collection Grid", 0, 800, 1440, 900)]
        figma = [self._figma_section("Collection - Featured", 0, 850, 1440, 850)]
        boxes = excluded_regions(dom, figma, {"collection"})
        assert len(boxes) == 2

    def test_non_matching_sections_excluded_from_result(self):
        dom = [self._dom_section("About Us", 0, 0, 1440, 400)]
        assert excluded_regions(dom, [], {"footer"}) == []


class TestOverlapFraction:
    def test_full_overlap_is_one(self):
        assert _overlap_fraction((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0

    def test_no_overlap_is_zero(self):
        assert _overlap_fraction((0, 0, 100, 100), (500, 500, 100, 100)) == 0.0

    def test_partial_overlap(self):
        # issue box 100x100 at (0,0); excl box covers right half (50-150, 0-100)
        frac = _overlap_fraction((0, 0, 100, 100), (50, 0, 100, 100))
        assert frac == 0.5

    def test_zero_area_issue_box_is_zero(self):
        assert _overlap_fraction((0, 0, 0, 0), (0, 0, 100, 100)) == 0.0


class TestFilterOutSections:
    def _dom(self, heading):
        return {"heading": heading, "classes": "", "id": "", "sectionType": "", "textSnippet": ""}

    def _figma(self, name):
        return {"name": name}

    def test_no_exclude_types_returns_originals_unchanged(self):
        dom = [self._dom("Header")]
        figma = [self._figma("Header")]
        result_dom, result_figma = filter_out_sections(dom, figma, set())
        assert result_dom == dom
        assert result_figma == figma

    def test_drops_matching_sections_from_both_lists(self):
        dom = [self._dom("Main Header"), self._dom("Hero Banner")]
        figma = [self._figma("Header"), self._figma("Hero")]
        result_dom, result_figma = filter_out_sections(dom, figma, {"nav"})
        assert len(result_dom) == 1
        assert result_dom[0]["heading"] == "Hero Banner"
        assert len(result_figma) == 1
        assert result_figma[0]["name"] == "Hero"


class TestFilterExcludedIssues:
    def _issue(self, x, y, w, h):
        return {"x": x, "y": y, "width": w, "height": h, "element": "test"}

    def test_no_boxes_returns_all_issues(self):
        issues = [self._issue(0, 0, 100, 100)]
        assert filter_excluded_issues(issues, []) == issues

    def test_drops_issue_fully_inside_excluded_box(self):
        issues = [self._issue(10, 10, 50, 50)]
        boxes = [(0, 0, 200, 200)]
        assert filter_excluded_issues(issues, boxes) == []

    def test_keeps_issue_outside_excluded_box(self):
        issues = [self._issue(1000, 1000, 50, 50)]
        boxes = [(0, 0, 200, 200)]
        assert filter_excluded_issues(issues, boxes) == issues

    def test_respects_min_overlap_threshold(self):
        # Only 20% overlap — below default 0.4 threshold, should be kept
        issue = self._issue(0, 0, 100, 100)
        boxes = [(80, 0, 100, 100)]  # overlaps x=80-100 → 20% of issue area
        assert filter_excluded_issues([issue], boxes) == [issue]

    def test_drops_issue_above_overlap_threshold(self):
        issue = self._issue(0, 0, 100, 100)
        boxes = [(50, 0, 100, 100)]  # 50% overlap — above default 0.4 threshold
        assert filter_excluded_issues([issue], boxes) == []
