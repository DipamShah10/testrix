"""Tests for visual/section_targeting.py — previously had zero coverage.

Focused on the fixes added to kill a real, confirmed bug: two different live
sections/cards in a "fleet" of near-identical repeating cards independently
resolving to the SAME Figma anchor, producing two report entries that quote
the same Figma text with reversed, contradictory numbers.
"""
from visual.section_targeting import (
    find_live_section,
    find_figma_anchor_node,
    derive_figma_region,
    select_first_n_sections,
    filter_nodes_in_region,
)


def _text_node(text, x=0, y=0, w=150, h=30):
    return {"text": text, "x": x, "y": y, "width": w, "height": h}


def _dom_section(heading="", y=0, height=200, snippet="", sec_id=""):
    return {"heading": heading, "y": y, "height": height, "textSnippet": snippet, "id": sec_id}


class TestFindFigmaAnchorNode:
    def test_finds_best_text_match_full_search(self):
        nodes = [_text_node("Contact Us", y=100), _text_node("Luxury SUVs", y=500)]
        result = find_figma_anchor_node("Luxury SUVs", nodes)
        assert result is not None
        node, idx = result
        assert node["text"] == "Luxury SUVs"
        assert idx == 1

    def test_below_floor_returns_none(self):
        nodes = [_text_node("Completely unrelated paragraph about something else entirely")]
        assert find_figma_anchor_node("Luxury SUVs", nodes) is None

    def test_windowed_search_prefers_nearby_candidate_over_global_best(self):
        # Two nodes both named "Luxury SUVs" (repeating card labels) — one
        # near the expected position, one far away. Without a positional
        # prior, a purely global search could pick either (tie). With a
        # window, the nearby one must win.
        nodes = [
            _text_node("Luxury SUVs", y=2000),   # far from expected_y
            _text_node("Luxury SUVs", y=505),    # near expected_y=500
        ]
        result = find_figma_anchor_node(
            "Luxury SUVs", nodes, expected_y=500, search_radius=200,
        )
        assert result is not None
        node, idx = result
        assert idx == 1
        assert node["y"] == 505

    def test_windowed_search_falls_back_to_full_page_when_window_empty(self):
        nodes = [_text_node("Luxury SUVs", y=2000)]
        result = find_figma_anchor_node(
            "Luxury SUVs", nodes, expected_y=500, search_radius=100,
        )
        assert result is not None
        node, idx = result
        assert node["y"] == 2000

    def test_used_indices_excluded_forces_different_anchor(self):
        # Two identical "Luxury SUVs" candidates both within the search
        # window — if index 0 is already used, the search must land on
        # index 1 rather than returning the same node twice.
        nodes = [_text_node("Luxury SUVs", y=490), _text_node("Luxury SUVs", y=510)]
        used = {0}
        result = find_figma_anchor_node(
            "Luxury SUVs", nodes, expected_y=500, search_radius=200, used_indices=used,
        )
        assert result is not None
        node, idx = result
        assert idx == 1

    def test_used_indices_exhausted_returns_none(self):
        nodes = [_text_node("Luxury SUVs", y=500)]
        used = {0}
        assert find_figma_anchor_node("Luxury SUVs", nodes, used_indices=used) is None


class TestDeriveFigmaRegion:
    def test_used_anchor_indices_accumulates_across_calls(self):
        # Simulates run_multi_section_qa's per-section loop: two live
        # sections, both with a heading that matches one of two identical
        # Figma "Luxury SUVs" text nodes. The SAME used_anchor_indices set
        # must be passed to both calls, and each call must land on a
        # DIFFERENT Figma node — this is the direct regression test for the
        # confirmed duplicate/reversed-numbers bug.
        figma_nodes = [
            _text_node("Luxury SUVs", y=500, h=30),
            _text_node("Luxury SUVs", y=1500, h=30),
        ]
        live_section_1 = _dom_section(heading="Luxury SUVs", y=900, height=100)
        live_section_2 = _dom_section(heading="Luxury SUVs", y=2700, height=100)

        used: set[int] = set()
        region_1 = derive_figma_region(
            live_section_1, figma_nodes,
            live_page_height=3000, figma_frame_height=2000, frame_width=1440,
            used_anchor_indices=used,
        )
        region_2 = derive_figma_region(
            live_section_2, figma_nodes,
            live_page_height=3000, figma_frame_height=2000, frame_width=1440,
            used_anchor_indices=used,
        )

        assert len(used) == 2
        assert region_1["y"] != region_2["y"]

    def test_without_used_anchor_indices_still_works(self):
        # run_single_section_qa's call site — no used_anchor_indices passed.
        figma_nodes = [_text_node("Luxury SUVs", y=500, h=30)]
        live_section = _dom_section(heading="Luxury SUVs", y=900, height=100)
        region = derive_figma_region(
            live_section, figma_nodes,
            live_page_height=3000, figma_frame_height=2000, frame_width=1440,
        )
        assert region["width"] == 1440

    def test_no_anchor_falls_back_to_proportional_position(self):
        live_section = _dom_section(heading="Nothing Like This", y=1500, height=100)
        region = derive_figma_region(
            live_section, [],
            live_page_height=3000, figma_frame_height=2000, frame_width=1440,
        )
        assert region["y"] == (1500 / 3000) * 2000


class TestSelectFirstNSections:
    def test_orders_by_y_and_limits(self):
        sections = [_dom_section(y=300), _dom_section(y=100), _dom_section(y=200)]
        result = select_first_n_sections(sections, 2)
        assert len(result) == 2
        assert result[0]["y"] == 100
        assert result[1]["y"] == 200

    def test_from_end_picks_last_n_in_page_order(self):
        sections = [_dom_section(y=100), _dom_section(y=200), _dom_section(y=300), _dom_section(y=400)]
        result = select_first_n_sections(sections, 2, from_end=True)
        assert len(result) == 2
        # Last 2 by y (300, 400), but still returned top-to-bottom, not reversed.
        assert result[0]["y"] == 300
        assert result[1]["y"] == 400

    def test_from_end_with_fewer_sections_than_n_returns_all(self):
        sections = [_dom_section(y=100), _dom_section(y=200)]
        result = select_first_n_sections(sections, 5, from_end=True)
        assert len(result) == 2
        assert result[0]["y"] == 100
        assert result[1]["y"] == 200


class TestFilterNodesInRegion:
    def test_keeps_only_nodes_with_center_in_range(self):
        nodes = [_text_node("a", y=0, h=20), _text_node("b", y=100, h=20), _text_node("c", y=300, h=20)]
        result = filter_nodes_in_region(nodes, y_start=50, y_end=150)
        assert len(result) == 1
        assert result[0]["text"] == "b"
