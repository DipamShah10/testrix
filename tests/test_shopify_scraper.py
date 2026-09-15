"""Tests for services/shopify_scraper.py — URL validation (pure, no browser)."""
import pytest
from unittest.mock import patch

from services.shopify_scraper import validate_url


class TestValidateUrl:
    def test_valid_https_url(self):
        with patch("socket.gethostbyname", return_value="104.21.0.1"):
            validate_url("https://my-store.myshopify.com")  # should not raise

    def test_http_rejected(self):
        with pytest.raises(ValueError, match="Only https"):
            validate_url("http://my-store.myshopify.com")

    def test_no_scheme_rejected(self):
        with pytest.raises(ValueError):
            validate_url("my-store.myshopify.com")

    def test_localhost_rejected(self):
        with patch("socket.gethostbyname", return_value="127.0.0.1"):
            with pytest.raises(ValueError, match="Private/internal"):
                validate_url("https://localhost")

    def test_private_10_network_rejected(self):
        with patch("socket.gethostbyname", return_value="10.0.0.1"):
            with pytest.raises(ValueError, match="Private/internal"):
                validate_url("https://internal.corp")

    def test_private_192_168_rejected(self):
        with patch("socket.gethostbyname", return_value="192.168.1.1"):
            with pytest.raises(ValueError, match="Private/internal"):
                validate_url("https://router.local")

    def test_link_local_169_rejected(self):
        with patch("socket.gethostbyname", return_value="169.254.0.1"):
            with pytest.raises(ValueError, match="Private/internal"):
                validate_url("https://metadata.internal")

    def test_unresolvable_host_rejected(self):
        import socket
        with patch("socket.gethostbyname", side_effect=socket.gaierror("not found")):
            with pytest.raises(ValueError, match="Cannot resolve"):
                validate_url("https://totally-fake-nonexistent-host.example")

    def test_no_hostname_rejected(self):
        with pytest.raises(ValueError, match="no hostname"):
            validate_url("https://")
