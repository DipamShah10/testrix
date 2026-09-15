"""Tests for services/visual_comparator.py — pixel diff with synthetic images."""
import io
import pytest
from PIL import Image

from services.visual_comparator import compare, CompareResult, _normalize, _match_heights, _find_regions


def _make_png(width: int, height: int, color=(255, 255, 255)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_png_with_patch(width: int, height: int, patch_color=(255, 0, 0), patch_rect=(100, 100, 200, 200)) -> bytes:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    x1, y1, x2, y2 = patch_rect
    for x in range(x1, x2):
        for y in range(y1, y2):
            img.putpixel((x, y), patch_color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestNormalize:
    def test_resizes_to_canonical_width(self):
        from services.visual_comparator import _CANONICAL_WIDTH, _load_image, _normalize
        img = _load_image(_make_png(2880, 5760), "test")
        result = _normalize(img, "test")
        assert result.width == _CANONICAL_WIDTH

    def test_preserves_aspect_ratio(self):
        from services.visual_comparator import _load_image, _normalize
        img = _load_image(_make_png(2880, 4320), "test")  # 2:3 ratio
        result = _normalize(img, "test")
        expected_h = int(4320 * (1440 / 2880))
        assert abs(result.height - expected_h) <= 1


class TestMatchHeights:
    def test_same_height_unchanged(self):
        a = Image.new("RGB", (100, 200))
        b = Image.new("RGB", (100, 200))
        ra, rb = _match_heights(a, b)
        assert ra.height == rb.height == 200

    def test_shorter_padded(self):
        a = Image.new("RGB", (100, 300))
        b = Image.new("RGB", (100, 200))
        ra, rb = _match_heights(a, b)
        assert ra.height == rb.height == 300


class TestCompare:
    def test_identical_images_zero_diff(self):
        img = _make_png(1440, 900)
        result = compare(img, img)
        assert isinstance(result, CompareResult)
        assert result.diff_percent == 0.0
        assert result.regions == []

    def test_completely_different_images_high_diff(self):
        white = _make_png(1440, 900, color=(255, 255, 255))
        black = _make_png(1440, 900, color=(0, 0, 0))
        result = compare(white, black)
        assert result.diff_percent > 50.0

    def test_diff_image_is_png_bytes(self):
        a = _make_png(1440, 900)
        b = _make_png(1440, 900, color=(200, 200, 200))
        result = compare(a, b)
        assert result.diff_image[:4] == b"\x89PNG"

    def test_diff_mask_is_png_bytes(self):
        a = _make_png(1440, 900)
        b = _make_png(1440, 900, color=(0, 0, 0))
        result = compare(a, b)
        assert result.diff_mask[:4] == b"\x89PNG"

    def test_localized_diff_produces_region(self):
        # One image is white, other has a red patch → should find at least 1 region
        white = _make_png(1440, 900)
        patched = _make_png_with_patch(1440, 900, patch_color=(255, 0, 0), patch_rect=(200, 200, 400, 400))
        result = compare(white, patched, diff_threshold=0.01)
        assert len(result.regions) >= 1

    def test_region_has_correct_fields(self):
        white = _make_png(1440, 900)
        patched = _make_png_with_patch(1440, 900, patch_color=(0, 0, 255), patch_rect=(100, 100, 300, 300))
        result = compare(white, patched, diff_threshold=0.01)
        if result.regions:
            r = result.regions[0]
            assert r.x >= 0
            assert r.y >= 0
            assert r.width > 0
            assert r.height > 0
            assert 0.0 <= r.diff_percent <= 100.0

    def test_high_threshold_suppresses_small_diff(self):
        a = _make_png(1440, 900, color=(200, 200, 200))
        b = _make_png(1440, 900, color=(210, 210, 210))
        result = compare(a, b, diff_threshold=0.5)
        assert result.diff_percent == 0.0

    def test_raises_on_empty_figma_bytes(self):
        with pytest.raises(Exception):
            compare(b"", _make_png(1440, 900))

    def test_raises_on_empty_live_bytes(self):
        with pytest.raises(Exception):
            compare(_make_png(1440, 900), b"")
