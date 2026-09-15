from urllib.parse import parse_qs, urlparse

import httpx

from qa.models import FigmaBaseline
from utils.retry import with_retry


class FigmaAnalyzer:
    def __init__(self, figma_token: str) -> None:
        self.figma_token = figma_token

    async def analyze(self, figma_url: str) -> FigmaBaseline:
        if not self.figma_token:
            raise ValueError("FIGMA_API_TOKEN is required for Figma analysis.")
        file_key, node_id = self._parse_url(figma_url)
        headers = {"X-Figma-Token": self.figma_token}

        async def load_document() -> dict:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10, read=60, write=10, pool=10)
            ) as client:
                if node_id:
                    resp = await client.get(
                        f"https://api.figma.com/v1/files/{file_key}/nodes",
                        headers=headers,
                        params={"ids": node_id},
                    )
                else:
                    resp = await client.get(
                        f"https://api.figma.com/v1/files/{file_key}",
                        headers=headers,
                        params={"depth": 4},
                    )
                if resp.status_code == 401:
                    raise ValueError("Figma 401 — FIGMA_API_TOKEN is invalid or expired.")
                if resp.status_code == 403:
                    raise ValueError("Figma 403 — token does not have access to this file.")
                resp.raise_for_status()
                return resp.json()

        payload = await with_retry(load_document, retries=5, base_delay=15.0)
        if node_id:
            nodes_payload = payload.get("nodes", {})
            entry = nodes_payload.get(node_id) or next(iter(nodes_payload.values()), {})
            root = entry.get("document", {})
        else:
            root = payload.get("document", {})
        target_nodes = self._collect_nodes(root)

        return FigmaBaseline(
            source_url=figma_url,
            typography=sorted({n.get("style", {}).get("fontFamily", "") for n in target_nodes if n.get("style")}),
            colors=sorted({self._extract_color(n) for n in target_nodes if self._extract_color(n)}),
            components=sorted({n.get("name", "") for n in target_nodes if n.get("type") in {"COMPONENT", "INSTANCE", "COMPONENT_SET"}}),
            layout_structure=[n.get("name", "") for n in target_nodes if n.get("type") in {"FRAME", "SECTION"}][:100],
            spacing=[self._extract_spacing(n) for n in target_nodes if self._extract_spacing(n)],
            buttons=[n.get("name", "") for n in target_nodes if "button" in n.get("name", "").lower()],
            responsive_structure=[n.get("layoutMode", "") for n in target_nodes if n.get("layoutMode")],
        )

    def _parse_url(self, figma_url: str) -> tuple[str, str | None]:
        parsed = urlparse(figma_url)
        parts = parsed.path.strip("/").split("/")
        file_key = None
        for index, part in enumerate(parts):
            if part in {"design", "file"} and index + 1 < len(parts):
                file_key = parts[index + 1]
                break
        if not file_key:
            raise ValueError("Could not parse Figma file key from URL.")
        raw_node_id = parse_qs(parsed.query).get("node-id", [None])[0]
        return file_key, raw_node_id.replace("-", ":") if raw_node_id else None

    def _collect_nodes(self, root: dict) -> list[dict]:
        nodes: list[dict] = []

        def walk(node: dict) -> None:
            nodes.append(node)
            for child in node.get("children", []):
                walk(child)

        if root:
            walk(root)
        return nodes

    def _extract_color(self, node: dict) -> str:
        fills = node.get("fills", [])
        for fill in fills:
            color = fill.get("color")
            if color:
                r = int(color.get("r", 0) * 255)
                g = int(color.get("g", 0) * 255)
                b = int(color.get("b", 0) * 255)
                return f"rgb({r}, {g}, {b})"
        return ""

    def _extract_spacing(self, node: dict) -> str:
        padding = [node.get("paddingTop"), node.get("paddingRight"), node.get("paddingBottom"), node.get("paddingLeft")]
        if any(value is not None for value in padding):
            return f"padding={padding}"
        item_spacing = node.get("itemSpacing")
        return f"itemSpacing={item_spacing}" if item_spacing is not None else ""
