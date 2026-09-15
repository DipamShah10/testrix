"""Tests for services/figma_extractor.py — URL parsing (pure, no API calls)."""
import pytest

from services.figma_extractor import parse_figma_url


class TestParseFigmaUrl:
    def test_design_url_no_node(self):
        url = "https://www.figma.com/design/AbCdEfGhIjKl/My-Design-File"
        file_key, node_id = parse_figma_url(url)
        assert file_key == "AbCdEfGhIjKl"
        assert node_id is None

    def test_design_url_with_node_id(self):
        url = "https://www.figma.com/design/AbCdEfGhIjKl/My-File?node-id=1-234"
        file_key, node_id = parse_figma_url(url)
        assert file_key == "AbCdEfGhIjKl"
        assert node_id == "1:234"

    def test_file_url_legacy(self):
        url = "https://www.figma.com/file/XyZKey123/Title"
        file_key, node_id = parse_figma_url(url)
        assert file_key == "XyZKey123"
        assert node_id is None

    def test_node_id_dash_converted_to_colon(self):
        url = "https://www.figma.com/design/KEY/Name?node-id=5-100"
        _, node_id = parse_figma_url(url)
        assert node_id == "5:100"

    def test_proto_url(self):
        url = "https://www.figma.com/proto/KEY123/Prototype?node-id=10-20"
        file_key, node_id = parse_figma_url(url)
        assert file_key == "KEY123"
        assert node_id == "10:20"

    def test_slides_url(self):
        url = "https://www.figma.com/slides/SlidesKey/My-Slides"
        file_key, _ = parse_figma_url(url)
        assert file_key == "SlidesKey"

    def test_branch_url(self):
        # Branch URL format: /design/:fileKey/branch/:branchKey/...
        # The extra title segment is NOT present — branch immediately follows the file key.
        url = "https://www.figma.com/design/MAINKEY/branch/BRANCHKEY/SubPage"
        file_key, _ = parse_figma_url(url)
        assert file_key == "BRANCHKEY"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Could not extract file key"):
            parse_figma_url("https://www.figma.com/invalid/path/here")

    def test_no_node_id_query_param(self):
        url = "https://www.figma.com/design/KEY/Name?other-param=123"
        _, node_id = parse_figma_url(url)
        assert node_id is None
