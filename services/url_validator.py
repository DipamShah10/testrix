"""URL validation utilities — no Playwright dependency."""
import ipaddress
import socket
from urllib.parse import urlparse

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def validate_url(url: str) -> None:
    """Validate that a URL is HTTPS and not on a private/internal network."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Only https:// URLs are allowed, got: {parsed.scheme}://")
    if not parsed.hostname:
        raise ValueError("URL has no hostname")
    try:
        ip_str = socket.gethostbyname(parsed.hostname)
        ip = ipaddress.ip_address(ip_str)
    except socket.gaierror:
        raise ValueError(f"Cannot resolve hostname: {parsed.hostname}")
    for network in _BLOCKED_NETWORKS:
        if ip in network:
            raise ValueError(f"Private/internal URLs are not allowed: {parsed.hostname}")
