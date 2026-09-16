"""Tests for services/bug_report_generator.py — previously had zero coverage.

Also documents a tried-and-reverted experiment: grouping spacing issues by
matching numeric gap values (regardless of element) looked right on a small,
hand-picked real example (3 pill-row elements all reporting "34px vs 10px"),
but a full real-page run showed it collapsing 39 unrelated issues spanning
all 5 page sections — including a horizontal "left of" gap grouped with
vertical "above"/"below" gaps — into one entry. That risks silently hiding
real, distinct defects, which is worse than the handful of near-duplicate
reports it was meant to fix, so _same_spacing_pattern is disabled
(unconditionally returns False) until a safer version exists. The tests
below guard against it being silently re-enabled without addressing that.
"""
from services.bug_report_generator import _group_similar_issues, _same_spacing_pattern


def _spacing_issue(element, direction, figma_gap, live_gap):
    diff = abs(figma_gap - live_gap)
    return {
        "element": f"spacing {direction} {element}",
        "description": (
            f'Spacing {direction} "{element}" is {live_gap:.0f}px on the live site '
            f'vs {figma_gap:.0f}px in Figma ({diff:.0f}px off).'
        ),
        "issue_type": "spacing",
        "severity": "Medium",
        "gap_direction": direction,
        "figma_gap_px": figma_gap,
        "live_gap_px": live_gap,
    }


class TestSameSpacingPatternDisabled:
    """
    _same_spacing_pattern is intentionally disabled (always False) — see
    module docstring. These tests lock that in: even the exact case that
    originally motivated the feature (identical gap values, different
    elements) must NOT be grouped by number alone until a safer version
    (e.g. requiring same direction AND a shared-cause signal, not just
    coincidence of magnitude) replaces this.
    """

    def test_identical_gap_values_do_not_match(self):
        a = _spacing_issue("Optional chauffeur", "above", 10.0, 34.0)
        b = _spacing_issue("Support through your booking", "above", 10.0, 34.0)
        assert _same_spacing_pattern(a, b) is False

    def test_non_spacing_issue_types_never_match(self):
        a = {"issue_type": "typography", "gap_direction": "above", "figma_gap_px": 10.0, "live_gap_px": 34.0}
        b = {"issue_type": "typography", "gap_direction": "above", "figma_gap_px": 10.0, "live_gap_px": 34.0}
        assert _same_spacing_pattern(a, b) is False

    def test_missing_gap_fields_never_match(self):
        a = {"issue_type": "spacing", "description": "x"}
        b = {"issue_type": "spacing", "description": "y"}
        assert _same_spacing_pattern(a, b) is False


class TestGroupSimilarIssuesTextSimilarityOnly:
    """
    With numeric spacing-pattern matching disabled, _group_similar_issues
    falls back to its original, proven behavior: group only by description
    text similarity, within the same issue_type.
    """

    def test_three_pill_rows_with_identical_gaps_stay_separate(self):
        # This is the case that motivated (and then broke) the numeric
        # experiment above — three different elements, identical gap
        # numbers, but different quoted element names in the description.
        # With the experiment disabled, they correctly stay as 3 distinct
        # findings rather than risking a false merge.
        issues = [
            _spacing_issue("Optional chauffeur", "above", 10.0, 34.0),
            _spacing_issue("Support through your booking", "above", 10.0, 34.0),
            _spacing_issue("Events & special occasions", "below", 10.0, 34.0),
        ]
        grouped = _group_similar_issues(issues)
        assert len(grouped) == 3
        assert all("occurrence_count" not in g for g in grouped)

    def test_near_identical_descriptions_still_group(self):
        # Text-similarity grouping (the original, unmodified mechanism)
        # must still work for genuinely near-identical descriptions, e.g.
        # the same vision-LLM observation repeated with minor wording
        # differences across occurrences of one real duplicate defect.
        issues = [
            {"element": "Shop Now button", "description": "Font weight is bold on live vs regular in Figma for the Shop Now button", "issue_type": "typography", "severity": "Low"},
            {"element": "Shop Now button", "description": "Font weight is bold on live vs regular in Figma for the Shop Now button.", "issue_type": "typography", "severity": "High"},
        ]
        grouped = _group_similar_issues(issues)
        assert len(grouped) == 1
        assert grouped[0]["occurrence_count"] == 2
        # Worst severity among duplicates wins.
        assert grouped[0]["severity"] == "High"

    def test_genuinely_different_issues_stay_separate(self):
        issues = [
            {"element": "Title", "description": "Font family differs on Title", "issue_type": "typography", "severity": "Low"},
            {"element": "Subtitle", "description": "Color differs on Subtitle badge", "issue_type": "typography", "severity": "Low"},
        ]
        grouped = _group_similar_issues(issues)
        assert len(grouped) == 2
        assert all("occurrence_count" not in g for g in grouped)

    def test_different_issue_types_never_merge_regardless_of_text(self):
        issues = [
            {"element": "Title", "description": "Something differs here", "issue_type": "typography", "severity": "Low"},
            {"element": "Title", "description": "Something differs here", "issue_type": "spacing", "severity": "Low"},
        ]
        grouped = _group_similar_issues(issues)
        assert len(grouped) == 2
