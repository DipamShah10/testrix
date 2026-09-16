"""Tests for services/typography_diff.py."""
from services.typography_diff import (
    _normalize_family,
    _parse_css_color,
    _color_distance,
    match_text_nodes,
    compare_typography,
)


def _figma_node(text, x=0, y=0, w=200, h=30, font_family="Inter", font_weight=600,
                 font_size=16, color=(36, 28, 24), name=""):
    return {
        "name": name, "text": text, "x": x, "y": y, "width": w, "height": h,
        "font_family": font_family, "font_weight": font_weight,
        "font_size": font_size, "color": color,
    }


def _live_node(text, x=0, y=0, w=200, h=30, font_family="Inter", font_weight=600,
               font_size=16, color=(36, 28, 24)):
    return {
        "text": text, "x": x, "y": y, "width": w, "height": h,
        "font_family": font_family, "font_weight": font_weight,
        "font_size": font_size, "color": f"rgb({color[0]}, {color[1]}, {color[2]})",
    }


class TestNormalizeFamily:
    def test_strips_weight_suffix(self):
        assert _normalize_family("Inter Bold") == "inter"
        assert _normalize_family("Inter SemiBold") == "inter"

    def test_strips_css_fallback_stack(self):
        assert _normalize_family('"Inter", sans-serif') == "inter"

    def test_none_returns_empty(self):
        assert _normalize_family(None) == ""

    def test_case_insensitive(self):
        assert _normalize_family("INTER") == _normalize_family("inter regular")

    def test_strips_hyphenated_weight_suffix(self):
        # Custom webfonts commonly bake weight into the family name with a
        # hyphen (e.g. self-hosted @font-face families) rather than a space.
        assert _normalize_family("Raw-Bold") == "raw"
        assert _normalize_family("Quasimoda-Medium") == "quasimoda"
        assert _normalize_family("Quasimoda-Medium, sans-serif") == "quasimoda"

    def test_multiword_family_space_vs_concatenated_alias(self):
        # Regression: a two-word Figma display name vs a self-hosted webfont
        # loader's concatenated/hyphenated name for the SAME typeface must
        # normalize equal — this was the actual root cause of a real false
        # positive (Figma "Helvetica Neue" vs live "HelveticaNeueRegular").
        assert _normalize_family("Helvetica Neue") == _normalize_family("HelveticaNeueRegular")
        assert _normalize_family("Helvetica Neue") == _normalize_family("Helvetica-Neue-Medium")
        assert _normalize_family("Proxima Nova") == _normalize_family("ProximaNova")
        assert _normalize_family("Open Sans") == _normalize_family("OpenSans-Regular")

    def test_multiword_family_distinct_variant_still_differs(self):
        # A real distinct font variant (not a formatting artifact) must NOT
        # be collapsed — "Display" isn't a weight/style suffix, so there's
        # nothing legitimate to strip here.
        assert _normalize_family("Inter") != _normalize_family("InterDisplay")


class TestParseCssColor:
    def test_parses_rgb(self):
        assert _parse_css_color("rgb(36, 28, 24)") == (36, 28, 24)

    def test_parses_rgba(self):
        assert _parse_css_color("rgba(36, 28, 24, 0.5)") == (36, 28, 24)

    def test_invalid_returns_none(self):
        assert _parse_css_color("transparent") is None
        assert _parse_css_color(None) is None


class TestColorDistance:
    def test_identical_is_zero(self):
        assert _color_distance((10, 10, 10), (10, 10, 10)) == 0

    def test_far_colors_large_distance(self):
        assert _color_distance((0, 0, 0), (255, 255, 255)) > 400


class TestMatchTextNodes:
    def test_matches_by_text_similarity(self):
        figma = [_figma_node("Sign up to our studio", x=100, y=200)]
        live = [_live_node("Sign up to our studio", x=105, y=202)]
        pairs = match_text_nodes(figma, live)
        assert len(pairs) == 1
        assert pairs[0][0]["text"] == "Sign up to our studio"

    def test_no_match_below_similarity_threshold(self):
        figma = [_figma_node("Shop the new collection")]
        live = [_live_node("Contact our support team")]
        assert match_text_nodes(figma, live) == []

    def test_distant_identical_text_still_matches(self):
        # Exact text match should win even far apart (e.g. repeated nav item)
        figma = [_figma_node("Shop Now", x=50, y=50)]
        live = [_live_node("Shop Now", x=900, y=3000)]
        pairs = match_text_nodes(figma, live)
        assert len(pairs) == 1

    def test_each_node_used_at_most_once(self):
        figma = [_figma_node("Shop Now", x=0, y=0), _figma_node("Shop Now", x=0, y=500)]
        live = [_live_node("Shop Now", x=0, y=0)]
        pairs = match_text_nodes(figma, live)
        assert len(pairs) == 1

    def test_empty_inputs(self):
        assert match_text_nodes([], []) == []
        assert match_text_nodes([_figma_node("x")], []) == []
        assert match_text_nodes([], [_live_node("x")]) == []


class TestCompareTypography:
    def test_identical_styles_produce_no_issues(self):
        figma = [_figma_node("Menu Item", name="Menu Item")]
        live = [_live_node("Menu Item")]
        assert compare_typography(figma, live) == []

    def test_font_weight_mismatch_detected(self):
        figma = [_figma_node("Menu Item", font_weight=600, name="Menu Item")]
        live = [_live_node("Menu Item", font_weight=400)]
        issues = compare_typography(figma, live)
        assert len(issues) == 1
        fields = [m["field"] for m in issues[0]["typography_mismatches"]]
        assert "font weight" in fields
        assert "600" in issues[0]["typography_mismatches"][0]["figma"]
        assert issues[0]["issue_type"] == "typography"

    def test_font_size_mismatch_detected(self):
        figma = [_figma_node("Body copy", font_size=16)]
        live = [_live_node("Body copy", font_size=13)]
        issues = compare_typography(figma, live)
        assert len(issues) == 1
        assert issues[0]["typography_mismatches"][0]["field"] == "font size"

    def test_small_size_difference_is_ignored(self):
        # Sub-pixel rounding noise should not trigger a false positive
        figma = [_figma_node("Body copy", font_size=16.0)]
        live = [_live_node("Body copy", font_size=16.9)]
        assert compare_typography(figma, live) == []

    def test_color_mismatch_detected(self):
        figma = [_figma_node("Copyright", color=(20, 20, 20))]
        live = [_live_node("Copyright", color=(180, 180, 180))]
        issues = compare_typography(figma, live)
        assert len(issues) == 1
        assert issues[0]["typography_mismatches"][0]["field"] == "color"

    def test_multiple_mismatches_all_reported(self):
        figma = [_figma_node("Heading", font_weight=700, font_size=32, color=(0, 0, 0))]
        live = [_live_node("Heading", font_weight=400, font_size=24, color=(120, 120, 120))]
        issues = compare_typography(figma, live)
        assert len(issues) == 1
        fields = {m["field"] for m in issues[0]["typography_mismatches"]}
        assert fields == {"font weight", "font size", "color"}
        # Severity signal should scale with number of mismatches
        assert issues[0]["diff_percent"] == 60.0

    def test_font_family_alias_not_flagged(self):
        # "Inter Bold" (Figma naming) vs "Inter" (live) should NOT be a family mismatch
        figma = [_figma_node("Title", font_family="Inter Bold")]
        live = [_live_node("Title", font_family="Inter")]
        assert compare_typography(figma, live) == []

    def test_hyphenated_webfont_alias_not_flagged(self):
        # Regression: self-hosted webfont families are often named
        # "Quasimoda-Medium" in live CSS vs base "Quasimoda" + weight in Figma.
        figma = [_figma_node("Shop Now", font_family="Quasimoda", font_weight=500)]
        live = [_live_node("Shop Now", font_family="Quasimoda-Medium", font_weight=500)]
        assert compare_typography(figma, live) == []

    def test_multiword_font_family_alias_not_flagged(self):
        # Regression for the real false positive found in production: Figma's
        # "Helvetica Neue" vs a self-hosted webfont loader's concatenated
        # "HelveticaNeueRegular" is the SAME typeface, not a mismatch.
        figma = [_figma_node("Luxury SUVs", font_family="Helvetica Neue", font_weight=400)]
        live = [_live_node("Luxury SUVs", font_family="HelveticaNeueRegular", font_weight=400)]
        assert compare_typography(figma, live) == []

    def test_multiword_font_family_distinct_variant_still_flagged(self):
        figma = [_figma_node("Title", font_family="Inter")]
        live = [_live_node("Title", font_family="InterDisplay")]
        issues = compare_typography(figma, live)
        assert len(issues) == 1
        assert issues[0]["typography_mismatches"][0]["field"] == "font family"

    def test_font_family_real_mismatch_detected(self):
        figma = [_figma_node("Title", font_family="Inter")]
        live = [_live_node("Title", font_family="Arial")]
        issues = compare_typography(figma, live)
        assert len(issues) == 1
        assert issues[0]["typography_mismatches"][0]["field"] == "font family"

    def test_bare_font_family_mismatch_scores_low_not_high(self):
        # Regression: a real production report showed "HelveticaNeueRegular"
        # vs "Inter Display" — genuinely different font-family metadata, but
        # visually indistinguishable side-by-side in the actual rendered
        # crops (confirmed by direct human review). A font-family mismatch
        # with NOTHING else different about the text is often not something
        # a human would actually notice, so it must score low (Low severity
        # territory), not the same flat-40 weight as before.
        figma = [_figma_node("Title", font_family="Inter", font_weight=400, font_size=16)]
        live = [_live_node("Title", font_family="Arial", font_weight=400, font_size=16)]
        issues = compare_typography(figma, live)
        assert len(issues) == 1
        assert issues[0]["magnitude_score"] < 15.0  # lands at Low per severity_classifier

    def test_font_family_mismatch_with_other_difference_still_scores_high(self):
        # A font swap that ALSO changes weight/size/color is a genuinely
        # different, visibly different typeface treatment — keep the full
        # weight so this still lands at High/Critical, unlike the bare case.
        figma = [_figma_node("Title", font_family="Inter", font_weight=700, font_size=32)]
        live = [_live_node("Title", font_family="Arial", font_weight=400, font_size=32)]
        issues = compare_typography(figma, live)
        assert len(issues) == 1
        family_mismatch = next(m for m in issues[0]["typography_mismatches"] if m["field"] == "font family")
        assert issues[0]["magnitude_score"] >= 30.0  # lands at High+ per severity_classifier

    def test_issue_carries_region_coords_from_live_node(self):
        figma = [_figma_node("Title", x=10, y=10, font_size=20)]
        live = [_live_node("Title", x=15, y=12, w=180, h=28, font_size=14)]
        issues = compare_typography(figma, live)
        assert issues[0]["x"] == 15
        assert issues[0]["y"] == 12
        assert issues[0]["width"] == 180
        assert issues[0]["height"] == 28

    def test_issue_also_carries_figma_side_coords_separately(self):
        # Regression: the Figma-side crop was being sliced using the LIVE
        # element's coordinates reused verbatim against the Figma image,
        # producing a blank/black or unrelated-content crop. The matched
        # Figma element's own box must be carried through independently.
        figma = [_figma_node("Title", x=10, y=800, w=150, h=25, font_size=20)]
        live = [_live_node("Title", x=15, y=1600, w=180, h=28, font_size=14)]
        issues = compare_typography(figma, live)
        assert issues[0]["figma_x"] == 10
        assert issues[0]["figma_y"] == 800
        assert issues[0]["figma_width"] == 150
        assert issues[0]["figma_height"] == 25
        # And the figma-side box must be genuinely independent of the live one
        assert issues[0]["figma_y"] != issues[0]["y"]

    def test_empty_inputs_produce_no_issues(self):
        assert compare_typography([], []) == []


class TestMatchTextNodesReadingOrderTiebreak:
    """
    Regression coverage for a real bug: a page with several near-identical
    repeating cards (e.g. a fleet of SUV listings, each with a "Luxury SUVs"
    badge) produced two findings for the "same" element with reversed,
    contradictory numbers — card 1's live label got cross-matched to card 2's
    Figma label and vice versa. Reading order must break the tie correctly
    when genuine ambiguity (2+ near-identical candidates) exists on both sides.
    """

    def test_two_vs_two_near_identical_labels_pair_in_order_not_crossed(self):
        figma = [
            _figma_node("Luxury SUVs", x=100, y=500),
            _figma_node("Luxury SUVs", x=100, y=1500),
        ]
        live = [
            _live_node("Luxury SUVs", x=100, y=900),
            _live_node("Luxury SUVs", x=100, y=2700),
        ]
        pairs = match_text_nodes(figma, live)
        assert len(pairs) == 2
        # Card-1 (topmost on both sides) must pair with card-1, not card-2 —
        # verified by object identity, not just text (text is identical here).
        first = next(p for p in pairs if p[0] is figma[0])
        second = next(p for p in pairs if p[0] is figma[1])
        assert first[1] is live[0]
        assert second[1] is live[1]

    def test_asymmetric_twin_groups_still_pair_gracefully(self):
        # Figma mock shows 3 example cards; live inventory only renders 2 —
        # a common real-world Figma-vs-live density mismatch.
        figma = [
            _figma_node("Luxury SUVs", x=100, y=500),
            _figma_node("Luxury SUVs", x=100, y=1500),
            _figma_node("Luxury SUVs", x=100, y=2500),
        ]
        live = [
            _live_node("Luxury SUVs", x=100, y=900),
            _live_node("Luxury SUVs", x=100, y=2700),
        ]
        pairs = match_text_nodes(figma, live)
        assert len(pairs) == 2
        first = next(p for p in pairs if p[0] is figma[0])
        assert first[1] is live[0]

    def test_single_instance_labels_unaffected_by_twin_grouping(self):
        # Common case: no repeated text anywhere — must behave exactly as
        # before (no ambiguity, no ordering machinery kicks in).
        figma = [_figma_node("Contact Us", x=50, y=50)]
        live = [_live_node("Contact Us", x=55, y=52)]
        pairs = match_text_nodes(figma, live)
        assert len(pairs) == 1
        assert pairs[0][0] is figma[0]
        assert pairs[0][1] is live[0]
