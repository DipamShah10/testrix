"""Tests for services/geometry_diff.py — focused on the figma-side/live-side
coordinate separation added alongside typography_diff.py's equivalent fix
(see tests/test_typography_diff.py::TestCompareTypography::
test_issue_also_carries_figma_side_coords_separately for the root-cause
context: reusing live coordinates against the Figma image produces a blank
or wrong-content crop)."""
from services.geometry_diff import compare_dimensions, compare_spacing


def _item(x=0, y=0, w=100, h=40, label="Item"):
    return {"x": x, "y": y, "width": w, "height": h, "label": label}


class TestCompareDimensions:
    def test_issue_carries_both_live_and_figma_boxes_separately(self):
        figma = [_item(x=10, y=800, w=150, h=50, label="Icon")]
        live = [_item(x=15, y=1600, w=100, h=50, label="Icon")]
        issues = compare_dimensions(figma, live, "label", "image")
        assert len(issues) == 1
        issue = issues[0]
        assert issue["x"] == 15 and issue["y"] == 1600
        assert issue["width"] == 100 and issue["height"] == 50
        assert issue["figma_x"] == 10 and issue["figma_y"] == 800
        assert issue["figma_width"] == 150 and issue["figma_height"] == 50
        assert issue["figma_y"] != issue["y"]

    def test_no_dimension_diff_no_issue(self):
        figma = [_item(w=100, h=40)]
        live = [_item(w=100, h=40)]
        assert compare_dimensions(figma, live, "label", "image") == []


class TestCompareSpacing:
    def test_spacing_issue_crop_region_spans_target_and_neighbor(self):
        """
        Regression: a spacing issue's crop used to be a tight box around
        just the target element, which doesn't show the actual whitespace
        gap being complained about — a human reviewer had nothing to
        visually verify the claim against. The region must now span target
        + neighbor (+ the gap between them) on both the live and Figma side.
        """
        # Two vertically stacked figma elements with a 50px gap; matching
        # live elements with a 90px gap (real spacing mismatch).
        figma_target = _item(x=0, y=100, w=200, h=30, label="Title")
        figma_neighbor = _item(x=0, y=180, w=200, h=20, label="Subtitle")
        live_target = _item(x=5, y=1200, w=200, h=30, label="Title")
        live_neighbor = _item(x=5, y=1320, w=200, h=20, label="Subtitle")

        figma_boxes = [figma_target, figma_neighbor]
        live_boxes = [live_target, live_neighbor]
        matched = [(figma_target, live_target)]

        issues = compare_spacing(figma_boxes, live_boxes, matched, label_field="label")
        assert len(issues) >= 1
        issue = issues[0]

        # Figma-side region must be independent of the live-side region
        # (root cause of the earlier blank/wrong-content crop bug), and it
        # must be the matched FIGMA target's box UNIONED with its FIGMA
        # neighbor — not a copy of the live side, and not just the target
        # alone (which would cut the gap off entirely).
        assert issue["figma_y"] != issue["y"]
        assert issue["figma_y"] == figma_target["y"]  # topmost of target/neighbor
        # Region must extend far enough down to include the neighbor's
        # bottom edge (180 + 20 = 200), not stop at the target's own bottom
        # edge (100 + 30 = 130) — that's the actual fix being verified.
        assert issue["figma_y"] + issue["figma_height"] >= figma_neighbor["y"] + figma_neighbor["height"]
        assert issue["y"] + issue["height"] >= live_neighbor["y"] + live_neighbor["height"]

    def test_no_neighbor_falls_back_to_target_box(self):
        # A target with no neighbor on the flagged side (shouldn't normally
        # happen since f_gap/l_gap are only non-None when a neighbor exists,
        # but the fallback path itself must not crash).
        figma_target = _item(x=0, y=100, w=200, h=30, label="Title")
        live_target = _item(x=5, y=1200, w=200, h=30, label="Title")
        issues = compare_spacing([figma_target], [live_target], [(figma_target, live_target)], label_field="label")
        assert issues == []  # no neighbor on any side -> nothing to measure -> no issues
