"""Tests for ai_engine/utils.py — JSON extraction helpers."""
import pytest
from ai_engine.utils import extract_json_object, extract_json_array


class TestExtractJsonObject:
    def test_plain_json(self):
        raw = '{"key": "value", "num": 42}'
        result = extract_json_object(raw)
        assert result == {"key": "value", "num": 42}

    def test_markdown_fenced(self):
        raw = '```json\n{"a": 1}\n```'
        result = extract_json_object(raw)
        assert result == {"a": 1}

    def test_markdown_no_lang(self):
        raw = '```\n{"x": true}\n```'
        result = extract_json_object(raw)
        assert result == {"x": True}

    def test_json_with_surrounding_text(self):
        raw = 'Here is the result:\n{"severity": "High"}\nDone.'
        result = extract_json_object(raw)
        assert result == {"severity": "High"}

    def test_nested_object(self):
        raw = '{"outer": {"inner": [1, 2, 3]}}'
        result = extract_json_object(raw)
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_returns_none_on_empty(self):
        assert extract_json_object("") is None

    def test_returns_none_on_no_braces(self):
        assert extract_json_object("just plain text") is None

    def test_returns_none_on_invalid_json(self):
        assert extract_json_object("{invalid json}") is None


class TestExtractJsonArray:
    def test_plain_array(self):
        raw = '[{"a": 1}, {"b": 2}]'
        result = extract_json_array(raw)
        assert result == [{"a": 1}, {"b": 2}]

    def test_markdown_fenced_array(self):
        raw = '```json\n[1, 2, 3]\n```'
        result = extract_json_array(raw)
        assert result == [1, 2, 3]

    def test_array_with_surrounding_text(self):
        raw = 'Issues found:\n[{"issue": "x"}, {"issue": "y"}]\nEnd.'
        result = extract_json_array(raw)
        assert result == [{"issue": "x"}, {"issue": "y"}]

    def test_empty_array(self):
        result = extract_json_array("[]")
        assert result == []

    def test_returns_none_on_no_brackets(self):
        assert extract_json_array("no array here") is None

    def test_returns_none_on_invalid(self):
        assert extract_json_array("[broken") is None
