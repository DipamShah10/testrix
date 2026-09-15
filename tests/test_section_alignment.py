"""Tests for visual/section_alignment_engine.py."""
import pytest

from visual.section_alignment_engine import (
    normalize_section_coords,
    find_section_for_region,
    match_sections_to_frames,
)


class TestNormalizeSectionCoords:
    def test_no_scaling_needed(self):
        sections = [{"x": 0, "y": 100, "width": 1440, "height": 300}]
        result = normalize_section_coords(sections, original_width=1440)
        assert result[0] == sections[0]

    def test_scale_down(self):
        sections = [{"x": 0, "y": 200, "width": 2880, "height": 600}]
        result = normalize_section_coords(sections, original_width=2880, canonical_width=1440)
        assert result[0]["width"] == 1440
        assert result[0]["height"] == 300
        assert result[0]["y"] == 100

    def test_scale_up(self):
        sections = [{"x": 10, "y": 50, "width": 720, "height": 150}]
        result = normalize_section_coords(sections, original_width=720, canonical_width=1440)
        assert result[0]["width"] == 1440
        assert result[0]["height"] == 300
        assert result[0]["y"] == 100

    def test_zero_original_width_no_crash(self):
        sections = [{"x": 0, "y": 0, "width": 100, "height": 100}]
        result = normalize_section_coords(sections, original_width=0)
        assert result == sections

    def test_preserves_extra_fields(self):
        sections = [{"x": 0, "y": 0, "width": 1440, "height": 100, "heading": "Hero", "tag": "div"}]
        result = normalize_section_coords(sections, original_width=1440)
        assert result[0]["heading"] == "Hero"
        assert result[0]["tag"] == "div"

    def test_empty_list(self):
        assert normalize_section_coords([], original_width=1440) == []


class TestFindSectionForRegion:
    def _make_section(self, y, height, heading=""):
        return {"y": y, "height": height, "heading": heading}

    def test_exact_overlap(self):
        sections = [
            self._make_section(0, 300, "Hero"),
            self._make_section(300, 400, "Products"),
            self._make_section(700, 200, "Footer"),
        ]
        result = find_section_for_region(320, 100, sections)
        assert result["heading"] == "Products"

    def test_partial_overlap_returns_best(self):
        sections = [
            self._make_section(0, 200, "Nav"),
            self._make_section(200, 600, "Main"),
        ]
        # Region mostly in Main
        result = find_section_for_region(180, 200, sections)
        assert result["heading"] == "Main"

    def test_no_sections_returns_none(self):
        assert find_section_for_region(100, 50, []) is None

    def test_region_below_all_sections(self):
        sections = [self._make_section(0, 100, "Top")]
        # Still returns best overlap even if it's small
        result = find_section_for_region(500, 50, sections)
        # No overlap at all — returns None or best
        # The function returns None only when no sections, otherwise best overlap
        # With 0 overlap best_overlap=0, so returns None since best stays None
        assert result is None


class TestMatchSectionsToFrames:
    def test_empty_sections_returns_empty(self):
        frames = [{"name": "Home"}]
        assert match_sections_to_frames([], frames) == []

    def test_empty_frames_returns_empty(self):
        sections = [{"heading": "Hero", "classes": "hero", "id": "", "sectionType": ""}]
        assert match_sections_to_frames(sections, []) == []

    def test_single_pair(self):
        sections = [{"heading": "Hero", "classes": "hero-section", "id": "", "sectionType": "", "textSnippet": ""}]
        frames = [{"name": "Hero Frame"}]
        result = match_sections_to_frames(sections, frames)
        assert len(result) == 1
        assert result[0][1]["name"] == "Hero Frame"

    def test_each_frame_used_at_most_once(self):
        sections = [
            {"heading": "Nav", "classes": "nav", "id": "", "sectionType": "nav", "textSnippet": ""},
            {"heading": "Hero", "classes": "hero", "id": "", "sectionType": "hero", "textSnippet": ""},
        ]
        frames = [{"name": "Nav"}, {"name": "Hero"}]
        result = match_sections_to_frames(sections, frames)
        used_frames = [r[1]["name"] for r in result]
        assert len(used_frames) == len(set(used_frames))
