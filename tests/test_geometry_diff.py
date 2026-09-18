"""Tests for services/geometry_diff.py — focused on the figma-side/live-side
coordinate separation added alongside typography_diff.py's equivalent fix
(see tests/test_typography_diff.py::TestCompareTypography::
test_issue_also_carries_figma_side_coords_separately for the root-cause
context: reusing live coordinates against the Figma image produces a blank
or wrong-content crop)."""
from services.geometry_diff import (
    compare_dimensions,
    compare_spacing,
    _find_nearest_above,
    _find_nearest_below,
    _find_nearest_left,
    _find_nearest_right,
)


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


class TestCompareSpacingFigmaScale:
    """
    Regression coverage for a real, confirmed bug: comparing a Figma-space
    vertical gap directly against a raw live-space vertical gap, with no
    correction for the two sides having different total page heights,
    produces bogus large diffs even when the underlying design is
    implemented correctly and proportionally. Confirmed against a real
    report where 8 unrelated elements all measured live_gap_px=12 (correct,
    internally consistent within the live page) while their Figma-side
    counterparts for the same "above" measurement scattered from 104 to
    2284 (each internally consistent within the Figma frame, but not
    directly comparable to the live number without a scale correction).
    """

    def test_proportionally_correct_gap_not_flagged_once_scaled(self):
        # Figma frame is 4000px tall; live page renders at 8000px (2x taller
        # — common when the live page has more real content than the
        # static mock). A gap that is proportionally IDENTICAL on both
        # sides (3.5% of each page's total height) must NOT be flagged once
        # the correct figma_scale is supplied.
        figma_frame_height = 4000
        live_page_height = 8000
        figma_scale = figma_frame_height / live_page_height  # 0.5

        figma_target = _item(x=0, y=2800, w=200, h=30, label="Title")
        figma_neighbor = _item(x=0, y=2640, w=200, h=20, label="Above")  # figma gap = 2800-2660 = 140
        live_target = _item(x=0, y=5600, w=200, h=30, label="Title")
        live_neighbor = _item(x=0, y=5300, w=200, h=20, label="Above")  # live gap = 5600-5320 = 280 (exactly 2x)

        figma_boxes = [figma_target, figma_neighbor]
        live_boxes = [live_target, live_neighbor]
        matched = [(figma_target, live_target)]

        f_gap = figma_target["y"] - (figma_neighbor["y"] + figma_neighbor["height"])
        l_gap = live_target["y"] - (live_neighbor["y"] + live_neighbor["height"])
        assert f_gap == 140
        assert l_gap == 280  # exactly 2x figma's gap — proportionally perfect match

        # Without the fix (figma_scale=1.0, the default): this WOULD be
        # flagged, since 140 vs 280 raw looks like a real 140px difference.
        issues_unscaled = compare_spacing(figma_boxes, live_boxes, matched, label_field="label")
        assert len(issues_unscaled) >= 1

        # With the real scale factor supplied: 280 * 0.5 = 140 == figma's
        # 140 -> zero diff -> not flagged, correctly reflecting a properly
        # scaled implementation.
        issues_scaled = compare_spacing(
            figma_boxes, live_boxes, matched, label_field="label", figma_scale=figma_scale,
        )
        assert issues_scaled == []

    def test_genuine_mismatch_still_caught_after_scaling(self):
        # Same 2x page-height scenario, but this time the live gap is
        # genuinely wrong even after accounting for the scale — must still
        # be flagged, proving the fix doesn't just suppress everything.
        figma_scale = 4000 / 8000  # 0.5

        figma_target = _item(x=0, y=2800, w=200, h=30, label="Title")
        figma_neighbor = _item(x=0, y=2640, w=200, h=20, label="Above")  # figma gap = 140
        live_target = _item(x=0, y=5600, w=200, h=30, label="Title")
        # A genuinely broken implementation: almost no gap at all, instead
        # of the proportionally-correct ~280px.
        live_neighbor = _item(x=0, y=5580, w=200, h=20, label="Above")  # live gap = 5600-5600=0

        figma_boxes = [figma_target, figma_neighbor]
        live_boxes = [live_target, live_neighbor]
        matched = [(figma_target, live_target)]

        issues = compare_spacing(
            figma_boxes, live_boxes, matched, label_field="label", figma_scale=figma_scale,
        )
        assert len(issues) == 1
        assert issues[0]["gap_direction"] == "above"

    def test_scale_only_applied_to_vertical_directions(self):
        # Horizontal (left/right) gaps must NOT be scaled — the live page is
        # captured at the same WIDTH as the Figma frame, so horizontal units
        # already match; only the vertical (page-height) axis diverges.
        figma_scale = 0.5

        figma_target = _item(x=200, y=100, w=100, h=30, label="Btn")
        figma_neighbor = _item(x=50, y=100, w=100, h=30, label="Left")  # figma gap-left = 200-150=50
        live_target = _item(x=200, y=100, w=100, h=30, label="Btn")
        live_neighbor = _item(x=50, y=100, w=100, h=30, label="Left")  # live gap-left = 50 too (same, correct)

        figma_boxes = [figma_target, figma_neighbor]
        live_boxes = [live_target, live_neighbor]
        matched = [(figma_target, live_target)]

        # Identical horizontal gaps on both sides — must NOT be flagged
        # regardless of figma_scale, since horizontal isn't scaled.
        issues = compare_spacing(
            figma_boxes, live_boxes, matched, label_field="label", figma_scale=figma_scale,
        )
        assert issues == []

    def test_default_scale_of_one_is_backward_compatible(self):
        # Omitting figma_scale entirely must behave exactly as before this
        # fix (no correction applied) — existing callers that don't pass it
        # yet must see unchanged behavior.
        figma_target = _item(x=0, y=100, w=200, h=30, label="Title")
        figma_neighbor = _item(x=0, y=180, w=200, h=20, label="Subtitle")
        live_target = _item(x=5, y=1200, w=200, h=30, label="Title")
        live_neighbor = _item(x=5, y=1320, w=200, h=20, label="Subtitle")

        matched = [(figma_target, live_target)]
        boxes_figma = [figma_target, figma_neighbor]
        boxes_live = [live_target, live_neighbor]

        default_result = compare_spacing(boxes_figma, boxes_live, matched, label_field="label")
        explicit_result = compare_spacing(boxes_figma, boxes_live, matched, label_field="label", figma_scale=1.0)
        assert len(default_result) == len(explicit_result)
        if default_result:
            assert default_result[0]["figma_gap_px"] == explicit_result[0]["figma_gap_px"]


class TestAdjacencyOverlapTolerance:
    """
    Regression coverage for a second, separate confirmed bug found while
    manually verifying the figma_scale fix against a real run (With-Cleo,
    job 6aab8f42ae06533bc1767dbf): browser-extracted DOM text boxes
    (getBoundingClientRect) routinely extend a few px past their visible
    glyphs because of CSS line-height, so two block-level siblings that
    render with ~zero visual gap between them can still have bounding boxes
    that overlap by a handful of pixels. The real case: a heading
    (y=3846, height=88 -> bottom=3934) immediately followed by a paragraph
    (y=3930) overlapped by 4px. The original strict adjacency check
    (`bottom <= target.y`) excluded the heading entirely as a candidate
    neighbor for the paragraph's "above" search, and fell through to a
    kicker label sitting 118px further up (behind the whole heading) —
    reporting a 118px gap for something that was visually touching.
    """

    def test_nearest_above_tolerates_small_overlap_from_line_height(self):
        # Exact shape of the real confirmed case: caption's true nearest
        # neighbor above (the heading) overlaps it by 4px; a decoy sits far
        # above the heading and must NOT be picked.
        caption = _item(x=112, y=3930, w=1696, h=29, label="Caption")
        heading = _item(x=112, y=3846, w=1696, h=88, label="Heading")  # bottom=3934, overlaps caption by 4px
        decoy_far_above = _item(x=112, y=3796, w=1696, h=16, label="Kicker")  # bottom=3812, gap=118 to caption

        best, gap = _find_nearest_above(caption, [caption, heading, decoy_far_above])
        assert best is heading
        assert gap == 0.0  # overlap is floored at 0, not left unmeasured

    def test_nearest_below_tolerates_small_overlap_from_line_height(self):
        heading = _item(x=112, y=3846, w=1696, h=88, label="Heading")
        caption = _item(x=112, y=3930, w=1696, h=29, label="Caption")  # overlaps heading's bottom by 4px
        decoy_far_below = _item(x=112, y=4200, w=1696, h=16, label="Far")

        best, gap = _find_nearest_below(heading, [heading, caption, decoy_far_below])
        assert best is caption
        assert gap == 0.0

    def test_overlap_beyond_tolerance_still_excluded(self):
        # A box that genuinely overlaps by a lot (not line-height noise, but
        # real stacking/positioning) must NOT be treated as "above" — the
        # tolerance is a small fixed slop, not an invitation to match anything.
        target = _item(x=0, y=100, w=200, h=50, label="Target")
        mostly_overlapping = _item(x=0, y=80, w=200, h=100, label="Overlaps a lot")  # bottom=180, overlaps by 80px
        genuinely_above = _item(x=0, y=0, w=200, h=50, label="Above")  # bottom=50, gap=50

        best, gap = _find_nearest_above(target, [target, mostly_overlapping, genuinely_above])
        assert best is genuinely_above
        assert gap == 50.0

    def test_nearest_left_and_right_tolerate_small_overlap(self):
        target = _item(x=200, y=0, w=100, h=40, label="Target")
        left_neighbor = _item(x=95, y=0, w=110, h=40, label="Left")  # right edge=205, overlaps target by 5px
        right_neighbor = _item(x=295, y=0, w=100, h=40, label="Right")  # left edge=295, overlaps target by 5px

        best_left, gap_left = _find_nearest_left(target, [target, left_neighbor, right_neighbor])
        assert best_left is left_neighbor
        assert gap_left == 0.0

        best_right, gap_right = _find_nearest_right(target, [target, left_neighbor, right_neighbor])
        assert best_right is right_neighbor
        assert gap_right == 0.0

    def test_end_to_end_spacing_issue_uses_tolerant_gap_not_inflated_one(self):
        """
        Full compare_spacing() integration check mirroring the real bug:
        without the tolerance fix, the reported live gap would jump from
        ~0px to 118px (picking the decoy), producing a bogus large-magnitude
        finding. With the fix, the live gap correctly reflects the tiny
        real-world figma/live difference instead.
        """
        figma_target = _item(x=0, y=200, w=200, h=30, label="Caption")
        figma_heading = _item(x=0, y=150, w=200, h=50, label="Heading")  # figma gap = 200-200=0

        live_target = _item(x=0, y=3930, w=1696, h=29, label="Caption")
        live_heading = _item(x=0, y=3846, w=1696, h=88, label="Heading")  # bottom=3934, overlaps live_target by 4
        live_decoy = _item(x=0, y=3796, w=1696, h=16, label="Kicker")  # would give gap=118 if heading excluded

        figma_boxes = [figma_target, figma_heading]
        live_boxes = [live_target, live_heading, live_decoy]
        matched = [(figma_target, live_target)]

        issues = compare_spacing(figma_boxes, live_boxes, matched, label_field="label")
        above_issues = [i for i in issues if i["gap_direction"] == "above"]
        # Both sides measure ~0px gap once the heading is correctly found as
        # the nearest neighbor on both sides -> no issue, not a bogus 118px one.
        assert above_issues == []
