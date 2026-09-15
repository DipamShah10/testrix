import asyncio
import logging
import os
import time
from urllib.parse import urlparse, parse_qs

import httpx

logger = logging.getLogger(__name__)

_FIGMA_API = "https://api.figma.com/v1"

# ── Retry / backoff ──────────────────────────────────────────────────────────
_MAX_RETRIES = 5
_RETRY_BASE_S = 15          # first retry waits 15s, then 30s, 60s, 120s, 240s
_MAX_RETRY_WAIT_S = 120     # cap on any single wait — Figma's own Retry-After can be absurdly long
_EXPORT_BATCH_SIZE = 5      # max node IDs per /images request — avoids 400 on large files

# ── Cache ────────────────────────────────────────────────────────────────────
_CACHE_TTL_S = 300          # 5 min — reuse frames across back-to-back jobs

# ── Throttle ─────────────────────────────────────────────────────────────────
_MIN_INTERVAL_S = 0.5       # ≤ 2 req/s to api.figma.com (well inside 120/min limit)
_MAX_CONCURRENT = 2         # semaphore: at most 2 simultaneous Figma API calls

# ── Runtime state (all lazy-initialised on first use) ────────────────────────
_frame_cache: dict[str, tuple[float, list[dict]]] = {}  # cache_key → (ts, frames)
_fetch_locks: dict[str, asyncio.Lock] = {}               # per-file-key dedup lock
_throttle_lock: asyncio.Lock | None = None
_api_semaphore: asyncio.Semaphore | None = None
_last_api_call_at: float = 0.0

# ── Persistent HTTP clients (reuse TCP connections across jobs) ──────────────
_api_client: httpx.AsyncClient | None = None    # api.figma.com  — short read timeout
_cdn_client: httpx.AsyncClient | None = None    # CDN PNG download — long read timeout


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_api_client() -> httpx.AsyncClient:
    global _api_client
    if _api_client is None or _api_client.is_closed:
        _api_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=60, write=10, pool=10),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _api_client


def _get_cdn_client() -> httpx.AsyncClient:
    global _cdn_client
    if _cdn_client is None or _cdn_client.is_closed:
        _cdn_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
            limits=httpx.Limits(max_keepalive_connections=3, max_connections=5),
        )
    return _cdn_client


def _get_throttle_lock() -> asyncio.Lock:
    global _throttle_lock
    if _throttle_lock is None:
        _throttle_lock = asyncio.Lock()
    return _throttle_lock


def _get_semaphore() -> asyncio.Semaphore:
    global _api_semaphore
    if _api_semaphore is None:
        _api_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    return _api_semaphore


# ─────────────────────────────────────────────────────────────────────────────
# Throttle
# ─────────────────────────────────────────────────────────────────────────────

async def _throttle() -> None:
    """Enforce _MIN_INTERVAL_S between consecutive calls to api.figma.com."""
    global _last_api_call_at
    async with _get_throttle_lock():
        gap = _MIN_INTERVAL_S - (time.monotonic() - _last_api_call_at)
        if gap > 0:
            await asyncio.sleep(gap)
        _last_api_call_at = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# Token validation
# ─────────────────────────────────────────────────────────────────────────────

def _get_token() -> str:
    token = os.environ.get("FIGMA_API_TOKEN", "")
    if not token:
        raise ValueError("FIGMA_API_TOKEN is not set. Add it to your .env file.")
    if not token.startswith("figd_"):
        raise ValueError(
            "FIGMA_API_TOKEN looks invalid — Figma personal access tokens must start with 'figd_'."
        )
    return token


# ─────────────────────────────────────────────────────────────────────────────
# URL parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_figma_url(figma_url: str) -> tuple[str, str | None]:
    """Return (file_key, node_id | None) from any Figma URL format."""
    parsed = urlparse(figma_url)
    parts = parsed.path.strip("/").split("/")

    file_key = None
    for i, part in enumerate(parts):
        if part in ("design", "file", "board", "slides", "make", "proto") and i + 1 < len(parts):
            file_key = parts[i + 1]
            # branch URLs: /design/:fileKey/branch/:branchKey/... → use branchKey
            if i + 3 < len(parts) and parts[i + 2] == "branch":
                file_key = parts[i + 3]
            break

    if not file_key:
        raise ValueError(f"Could not extract file key from Figma URL: {figma_url}")

    node_id = None
    qs = parse_qs(parsed.query)
    raw_node = qs.get("node-id", [None])[0]
    if raw_node:
        node_id = raw_node.replace("-", ":")

    return file_key, node_id


# ─────────────────────────────────────────────────────────────────────────────
# Centralised API GET — throttle + semaphore + retry + structured logging
# ─────────────────────────────────────────────────────────────────────────────

async def _api_get(path: str, headers: dict, **kwargs) -> httpx.Response:
    """
    Single entry point for every call to api.figma.com.
    Applies throttling, concurrency cap, exponential backoff, and structured logging.
    """
    client = _get_api_client()
    semaphore = _get_semaphore()
    url = f"{_FIGMA_API}{path}"

    for attempt in range(_MAX_RETRIES):
        await _throttle()
        t0 = time.monotonic()

        try:
            async with semaphore:
                resp = await client.get(url, headers=headers, **kwargs)
        except httpx.TimeoutException as exc:
            elapsed = time.monotonic() - t0
            logger.warning(
                f"Figma timeout — path={path}, attempt={attempt + 1}/{_MAX_RETRIES}, "
                f"elapsed={elapsed:.2f}s: {exc}"
            )
            if attempt == _MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Figma API timed out after {_MAX_RETRIES} attempts ({path})"
                ) from exc
            await asyncio.sleep(_RETRY_BASE_S * (2 ** attempt))
            continue

        elapsed = time.monotonic() - t0

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = (
                int(retry_after)
                if retry_after and retry_after.isdigit()
                else _RETRY_BASE_S * (2 ** attempt)
            )
            if wait > _MAX_RETRY_WAIT_S:
                logger.warning(
                    f"Figma Retry-After={wait}s exceeds cap — clamping to {_MAX_RETRY_WAIT_S}s"
                )
                wait = _MAX_RETRY_WAIT_S
            logger.warning(
                f"Figma 429 — path={path}, attempt={attempt + 1}/{_MAX_RETRIES}, "
                f"retry_in={wait}s, elapsed={elapsed:.2f}s"
            )
            await asyncio.sleep(wait)
            continue

        if resp.status_code == 401:
            raise ValueError("Figma 401 — FIGMA_API_TOKEN is invalid or expired.")
        if resp.status_code == 403:
            raise ValueError("Figma 403 — token does not have access to this file.")
        if resp.status_code == 404:
            raise ValueError("Figma 404 — file not found. Check the URL and file permissions.")

        resp.raise_for_status()
        logger.info(
            f"Figma {resp.status_code} — path={path}, elapsed={elapsed:.2f}s, "
            f"response={len(resp.content)} bytes"
        )
        return resp

    raise RuntimeError(
        f"Figma API rate limit exceeded after {_MAX_RETRIES} retries ({path}). "
        "Wait a few minutes before submitting another job for the same file."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CDN download — separate client, no throttle needed (not api.figma.com)
# ─────────────────────────────────────────────────────────────────────────────

async def _download_png(img_url: str, frame_name: str) -> bytes:
    """Download an exported PNG from Figma's CDN with timeout retry."""
    client = _get_cdn_client()
    for attempt in range(3):
        t0 = time.monotonic()
        try:
            resp = await client.get(img_url)
            resp.raise_for_status()
            elapsed = time.monotonic() - t0
            logger.info(
                f"CDN download OK — frame={frame_name}, "
                f"size={len(resp.content)} bytes, elapsed={elapsed:.2f}s"
            )
            return resp.content
        except httpx.TimeoutException as exc:
            elapsed = time.monotonic() - t0
            logger.warning(
                f"CDN timeout — frame={frame_name}, attempt={attempt + 1}/3, "
                f"elapsed={elapsed:.2f}s: {exc}"
            )
            if attempt == 2:
                raise RuntimeError(
                    f"CDN download timed out after 3 attempts for frame '{frame_name}'"
                ) from exc
            await asyncio.sleep(5 * (attempt + 1))

    raise RuntimeError(f"CDN download failed for frame '{frame_name}'")


# ─────────────────────────────────────────────────────────────────────────────
# Public API — cache + dedup lock layer
# ─────────────────────────────────────────────────────────────────────────────

async def extract_frames(figma_url: str) -> tuple[list[dict], dict]:
    """
    Return (frames, typography) where:
      frames     — list of { name, node_id, image_bytes, width, height }
      typography — { fonts, sizes, weights, colors } extracted from TEXT nodes

    Caches results for _CACHE_TTL_S seconds so back-to-back jobs for the same
    Figma file make zero additional API calls — eliminating the 429 chain.
    A per-file lock ensures only one in-flight fetch per file key at a time.
    """
    token = _get_token()
    file_key, node_id = parse_figma_url(figma_url)
    cache_key = f"{file_key}:{node_id or ''}"

    # ── Fast path: serve from cache ──────────────────────────────────────────
    cached = _frame_cache.get(cache_key)
    if cached:
        fetched_at, results, typography = cached
        age = time.monotonic() - fetched_at
        if age < _CACHE_TTL_S:
            logger.info(
                f"Figma cache hit — key={cache_key}, age={age:.0f}s, "
                f"ttl_remaining={_CACHE_TTL_S - age:.0f}s, frames={len(results)}"
            )
            return results, typography
        logger.info(f"Figma cache expired — key={cache_key}, age={age:.0f}s, re-fetching")

    # ── Dedup lock: one fetch per file at a time ─────────────────────────────
    if cache_key not in _fetch_locks:
        _fetch_locks[cache_key] = asyncio.Lock()

    async with _fetch_locks[cache_key]:
        # Another waiter may have populated the cache while we were waiting
        cached = _frame_cache.get(cache_key)
        if cached:
            fetched_at, results, typography = cached
            if time.monotonic() - fetched_at < _CACHE_TTL_S:
                logger.info(f"Figma cache hit (post-lock) — key={cache_key}, frames={len(results)}")
                return results, typography

        results, typography = await _fetch_frames(token, file_key, node_id)
        _frame_cache[cache_key] = (time.monotonic(), results, typography)
        logger.info(
            f"Figma frames cached — key={cache_key}, frames={len(results)}, "
            f"fonts={typography['fonts']}, ttl={_CACHE_TTL_S}s"
        )
        return results, typography


# ─────────────────────────────────────────────────────────────────────────────
# Internal fetch — called only through extract_frames
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_frames(token: str, file_key: str, node_id: str | None) -> tuple[list[dict], dict]:
    """Hit the Figma API and download frame PNGs. No caching — use extract_frames."""
    headers = {"X-Figma-Token": token}

    # Step 1: file structure (depth 3 to reach TEXT nodes inside frames for typography)
    logger.info(f"Figma fetch start — key={file_key}, node_id={node_id or 'all'}")
    file_resp = await _api_get(f"/files/{file_key}", headers, params={"depth": 3})
    file_data = file_resp.json()

    frames = _collect_frames(file_data, node_id)
    if not frames:
        raise ValueError(
            "No frames found in the Figma file. "
            "Make sure the URL points to a file with top-level frames."
        )
    logger.info(f"Figma frames found — {len(frames)}: {[f['name'] for f in frames]}")

    # The depth=3 fetch above is only deep enough to *locate* each frame — its
    # own children are frequently truncated by that same depth cap, which is
    # why `sections` (and text/image/button content) can come back empty or
    # truncated for any frame with real nesting. Re-fetch each frame's full
    # (uncapped) subtree via /nodes ONCE, and derive sections, text nodes,
    # image nodes, and button nodes all from that single fetched document —
    # previously this same subtree was fetched twice (once here, once again
    # inside the old _fetch_frame_text_nodes), doubling Figma API usage for
    # no reason.
    for frame in frames:
        full_doc = None
        try:
            full_doc = await _fetch_frame_full_node(token, file_key, frame["node_id"])
        except Exception as exc:
            logger.warning(
                f"Full-depth fetch failed for frame '{frame['name']}' "
                f"({frame['node_id']}): {exc} — keeping depth-limited data"
            )
        frame["_full_doc"] = full_doc
        if full_doc:
            frame["sections"] = _extract_frame_sections(full_doc)

    # Step 2: export PNG URLs (batched to avoid 400 from URL-length limits)
    logger.info(f"Figma export request — {len(frames)} frame(s) at @2x PNG")
    image_urls: dict = {}
    for i in range(0, len(frames), _EXPORT_BATCH_SIZE):
        batch = frames[i:i + _EXPORT_BATCH_SIZE]
        node_ids = ",".join(f["node_id"] for f in batch)
        logger.info(f"Figma export batch {i // _EXPORT_BATCH_SIZE + 1}/{-(-len(frames) // _EXPORT_BATCH_SIZE)}: {len(batch)} frame(s)")
        img_resp = await _api_get(
            f"/images/{file_key}", headers,
            params={"ids": node_ids, "format": "png", "scale": 2},
        )
        img_data = img_resp.json()
        if img_data.get("err"):
            raise RuntimeError(f"Figma image export error: {img_data['err']}")
        image_urls.update(img_data.get("images", {}))

    # Step 3: download each PNG from CDN + fetch full-depth text nodes per frame.
    # depth=3 above is enough for sections, but text runs are often nested 3+
    # levels inside a frame (Frame > Group > Group > Text) and get truncated —
    # so each frame's text specs are fetched separately via /nodes (no depth cap).
    results = []
    for frame in frames:
        nid = frame["node_id"]
        img_url = image_urls.get(nid)
        if not img_url:
            logger.warning(f"No CDN URL for frame '{frame['name']}' ({nid}) — skipping")
            continue
        image_bytes = await _download_png(img_url, frame["name"])

        full_doc = frame.get("_full_doc")
        text_nodes = _extract_text_nodes(full_doc) if full_doc else []
        image_nodes = _extract_image_nodes(full_doc) if full_doc else []
        button_nodes = _extract_button_nodes(full_doc) if full_doc else []
        logger.info(
            f"Figma elements — frame={nid}, text={len(text_nodes)}, "
            f"images={len(image_nodes)}, buttons={len(button_nodes)}"
        )

        results.append({
            "name":         frame["name"],
            "node_id":      nid,
            "image_bytes":  image_bytes,
            "width":        frame.get("width"),
            "height":       frame.get("height"),
            "sections":     frame.get("sections", []),
            "text_nodes":   text_nodes,
            "image_nodes":  image_nodes,
            "button_nodes": button_nodes,
        })

    typography = _collect_typography(file_data)
    logger.info(
        f"Figma fetch complete — key={file_key}, {len(results)} frame(s) downloaded, "
        f"fonts={typography['fonts']}"
    )
    return results, typography


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame text node fetch — full depth, no truncation
# ─────────────────────────────────────────────────────────────────────────────

_MAX_TEXT_NODES_PER_FRAME = 200   # cap to keep matching/comparison cost bounded


async def _fetch_frame_full_node(token: str, file_key: str, node_id: str) -> dict | None:
    """
    Fetch one frame's full (untruncated) document subtree via /files/{key}/nodes.
    Unlike the depth-limited /files call used for frame discovery, this reaches
    every descendant regardless of nesting depth — needed for section extraction
    on frames whose real content sits deeper than the discovery depth cap.
    """
    headers = {"X-Figma-Token": token}
    resp = await _api_get(f"/files/{file_key}/nodes", headers, params={"ids": node_id})
    data = resp.json()
    entry = data.get("nodes", {}).get(node_id)
    return entry.get("document") if entry else None


# A short/icon-like text node (a lone "!", a single glyph in a badge) has a
# tight glyph-only bounding box that's much smaller than the padded visual
# container it actually sits inside (a callout, a chip, a labeled icon). Using
# that tight box as the neighbor-search anchor for spacing measurements
# measures from the wrong edge — the padding inside the container reads as
# part of the "gap" to whatever's next to it, wildly inflating the number.
# Confirmed against a real case: a "!" glyph (26px tall) inside a 66px-tall
# icon+message row — spacing measured from the glyph's own box overstated the
# real gap by exactly the container's extra padding.
#
# _CONTAINER_PAD_MIN_PX / _MAX_PX bound how much taller the immediate parent
# must be to count as "this text's padded container" rather than an unrelated
# larger layout wrapper: some padding is expected (min), but a huge jump (past
# max) means the parent is a whole section, not a snug wrapper around this text.
_CONTAINER_PAD_MIN_PX = 4
_CONTAINER_PAD_MAX_PX = 150

# A parent only qualifies as a text node's padded "spacing box" if its OTHER
# text children sit on roughly the same visual row (an icon glyph next to a
# one-line label) — not stacked below/above it (a heading with its own body
# paragraph inside one bordered callout). Confirmed as a real regression: a
# heading + paragraph sharing one bordered "note" box both have modest height
# padding individually, but treating the whole box as the HEADING's own
# spacing box silently absorbed the paragraph's space into the heading's
# effective bottom edge — corrupting every gap measured against that heading.
# A same-row icon+label pair has near-identical child y (within a few px of
# line-height/baseline variance); stacked content differs by tens of px.
_SAME_ROW_Y_TOLERANCE_PX = 20


def _extract_text_nodes(frame_doc: dict) -> list[dict]:
    """
    Collect every TEXT node's spec from an already-fetched full-depth frame
    document: text content, position (frame-relative), and style (font
    family/weight/size/color). Pure function — no API call, so it can be
    reused for section/image/button extraction off the same fetched tree
    instead of each needing its own /nodes round-trip.
    """
    bb = frame_doc.get("absoluteBoundingBox") or {}
    frame_x = bb.get("x", 0) or 0
    frame_y = bb.get("y", 0) or 0

    nodes: list[dict] = []

    def walk(node: dict, parent: dict | None) -> None:
        if len(nodes) >= _MAX_TEXT_NODES_PER_FRAME:
            return
        # An invisible node's children are invisible too, even if a child's own
        # "visible" flag is individually true (Figma doesn't rewrite descendant
        # flags when a parent is hidden) — e.g. a hidden/duplicate component
        # variant left in the tree. Skipping the whole subtree here, not just
        # the node itself, stops these ghost nodes from being picked as a real
        # spacing/typography neighbor.
        if node.get("visible", True) is False:
            return
        if node.get("type") == "TEXT":
            text = (node.get("characters") or "").strip()
            if text:
                node_bb = node.get("absoluteBoundingBox") or {}
                style = node.get("style", {}) or {}
                own_y = node_bb.get("y", 0) or 0
                own_height = node_bb.get("height", 0) or 0
                own_width = node_bb.get("width", 0) or 0

                color = None
                for fill in node.get("fills", []) or []:
                    if fill.get("visible", True) is False:
                        continue
                    c = fill.get("color")
                    if c:
                        color = (
                            round(c.get("r", 0) * 255),
                            round(c.get("g", 0) * 255),
                            round(c.get("b", 0) * 255),
                        )
                        break

                entry = {
                    "name":        node.get("name", ""),
                    "text":        text[:120],
                    "x":           max(0, round((node_bb.get("x", 0) or 0) - frame_x)),
                    "y":           max(0, round(own_y - frame_y)),
                    "width":       round(own_width),
                    "height":      round(own_height),
                    "font_family": style.get("fontFamily"),
                    "font_weight": style.get("fontWeight"),
                    "font_size":   style.get("fontSize"),
                    "color":       color,
                }

                parent_bb = parent.get("absoluteBoundingBox") if parent else None
                if parent_bb:
                    pad = (parent_bb.get("height", 0) or 0) - own_height
                    same_row = all(
                        abs((sib.get("absoluteBoundingBox", {}).get("y", 0) or 0) - own_y)
                        <= _SAME_ROW_Y_TOLERANCE_PX
                        for sib in parent.get("children", []) or []
                        if sib is not node and sib.get("type") == "TEXT"
                        and (sib.get("characters") or "").strip()
                        and sib.get("visible", True) is not False
                    )
                    if _CONTAINER_PAD_MIN_PX < pad <= _CONTAINER_PAD_MAX_PX \
                            and (parent_bb.get("width", 0) or 0) >= own_width \
                            and same_row:
                        entry["spacing_box"] = {
                            "x":      max(0, round((parent_bb.get("x", 0) or 0) - frame_x)),
                            "y":      max(0, round((parent_bb.get("y", 0) or 0) - frame_y)),
                            "width":  round(parent_bb.get("width", 0) or 0),
                            "height": round(parent_bb.get("height", 0) or 0),
                        }

                nodes.append(entry)
        for child in node.get("children", []) or []:
            walk(child, node)

    walk(frame_doc, None)
    return nodes


_MAX_IMAGE_NODES_PER_FRAME = 200


def _extract_image_nodes(frame_doc: dict) -> list[dict]:
    """
    Collect every node with a visible IMAGE fill from an already-fetched
    full-depth frame document — Figma's equivalent of a live-page <img>.
    """
    bb = frame_doc.get("absoluteBoundingBox") or {}
    frame_x = bb.get("x", 0) or 0
    frame_y = bb.get("y", 0) or 0

    nodes: list[dict] = []

    def walk(node: dict) -> None:
        if len(nodes) >= _MAX_IMAGE_NODES_PER_FRAME:
            return
        # Skip the whole subtree under an invisible node — a hidden ancestor
        # makes every descendant invisible too, regardless of each child's own
        # "visible" flag (see the matching note in _extract_text_nodes).
        if node.get("visible", True) is False:
            return

        fills = node.get("fills") or []
        has_image_fill = any(
            f.get("type") == "IMAGE" and f.get("visible", True) is not False
            for f in fills
        )
        if has_image_fill:
            node_bb = node.get("absoluteBoundingBox") or {}
            w = round(node_bb.get("width", 0) or 0)
            h = round(node_bb.get("height", 0) or 0)
            if w > 4 and h > 4:
                nodes.append({
                    "name":   node.get("name", ""),
                    # Common matching key with live image_elements — a live
                    # <img>'s alt text is often empty, so its own label
                    # falls back to its filename; Figma's layer name is the
                    # closest equivalent on that side.
                    "label":  node.get("name", ""),
                    "x":      max(0, round((node_bb.get("x", 0) or 0) - frame_x)),
                    "y":      max(0, round((node_bb.get("y", 0) or 0) - frame_y)),
                    "width":  w,
                    "height": h,
                })
        for child in node.get("children", []) or []:
            walk(child)

    walk(frame_doc)
    return nodes


# Heuristics for recognizing a "button" in Figma — there's no semantic button
# node type, so a button is inferred as a small, contained, filled/stroked
# frame-like node wrapping a short text label. Necessarily approximate.
_BUTTON_TEXT_MAX_LEN = 40
_BUTTON_MAX_TEXT_DESCENDANTS = 2
_BUTTON_MIN_W, _BUTTON_MAX_W = 40, 420
_BUTTON_MIN_H, _BUTTON_MAX_H = 22, 90
_MAX_BUTTON_NODES_PER_FRAME = 100


def _collect_text_descendants(node: dict, acc: list[dict]) -> None:
    if node.get("visible", True) is False:
        return
    if node.get("type") == "TEXT":
        if (node.get("characters") or "").strip():
            acc.append(node)
        return
    for child in node.get("children", []) or []:
        _collect_text_descendants(child, acc)


def _extract_button_nodes(frame_doc: dict) -> list[dict]:
    """
    Collect nodes that look like buttons: a FRAME/COMPONENT/INSTANCE/GROUP,
    button-shaped in size, with a visible fill or stroke, wrapping one short
    text label (its visible caption). Necessarily heuristic — Figma has no
    dedicated "button" node type.
    """
    bb = frame_doc.get("absoluteBoundingBox") or {}
    frame_x = bb.get("x", 0) or 0
    frame_y = bb.get("y", 0) or 0

    nodes: list[dict] = []

    def walk(node: dict) -> None:
        if len(nodes) >= _MAX_BUTTON_NODES_PER_FRAME:
            return
        if node.get("visible", True) is False:
            return

        ntype = node.get("type", "")
        if ntype in ("FRAME", "COMPONENT", "INSTANCE", "GROUP"):
            node_bb = node.get("absoluteBoundingBox") or {}
            w = node_bb.get("width", 0) or 0
            h = node_bb.get("height", 0) or 0
            is_button_shaped = (
                _BUTTON_MIN_W <= w <= _BUTTON_MAX_W and _BUTTON_MIN_H <= h <= _BUTTON_MAX_H
            )
            has_fill = any(f.get("visible", True) is not False for f in (node.get("fills") or []))
            has_stroke = bool(node.get("strokes"))

            if is_button_shaped and (has_fill or has_stroke):
                text_descendants: list[dict] = []
                _collect_text_descendants(node, text_descendants)
                label = " ".join(t.get("characters", "").strip() for t in text_descendants).strip()

                if 0 < len(text_descendants) <= _BUTTON_MAX_TEXT_DESCENDANTS \
                        and 0 < len(label) <= _BUTTON_TEXT_MAX_LEN:
                    nodes.append({
                        "name":   node.get("name", ""),
                        "label":  label,
                        "x":      max(0, round((node_bb.get("x", 0) or 0) - frame_x)),
                        "y":      max(0, round((node_bb.get("y", 0) or 0) - frame_y)),
                        "width":  round(w),
                        "height": round(h),
                    })
                    return  # matched as a button — don't also descend into its children

        for child in node.get("children", []) or []:
            walk(child)

    walk(frame_doc)
    return nodes


# ─────────────────────────────────────────────────────────────────────────────
# Frame tree walker
# ─────────────────────────────────────────────────────────────────────────────

def _extract_frame_sections(frame_node: dict) -> list[dict]:
    """
    Return direct children of a Figma frame as section descriptors.
    Coordinates are relative to the frame's own top-left corner.
    Only children with meaningful dimensions are included.
    """
    bb = frame_node.get("absoluteBoundingBox", {})
    frame_x = bb.get("x", 0) or 0
    frame_y = bb.get("y", 0) or 0

    sections = []
    for child in frame_node.get("children", []):
        if child.get("visible", True) is False:
            continue
        child_bb = child.get("absoluteBoundingBox", {})
        cw = int(child_bb.get("width",  0) or 0)
        ch = int(child_bb.get("height", 0) or 0)
        cx = child_bb.get("x", 0) or 0
        cy = child_bb.get("y", 0) or 0

        if cw < 100 or ch < 80:
            continue

        sections.append({
            "name":   child.get("name", ""),
            "type":   child.get("type", ""),
            "rel_x":  max(0, int(cx - frame_x)),
            "rel_y":  max(0, int(cy - frame_y)),
            "width":  cw,
            "height": ch,
        })

    return sections


def _collect_frames(file_data: dict, target_node_id: str | None) -> list[dict]:
    """Walk the Figma document tree and collect frame nodes with their section children."""
    frames = []
    document = file_data.get("document", {})

    def walk(node: dict) -> None:
        node_type = node.get("type", "")
        node_id = node.get("id", "")
        name = node.get("name", "Untitled")
        bb = node.get("absoluteBoundingBox", {})
        w = bb.get("width")
        h = bb.get("height")

        if node_type in ("FRAME", "COMPONENT", "SECTION"):
            if target_node_id:
                if node_id == target_node_id or node_id.replace("-", ":") == target_node_id:
                    frames.append({
                        "name": name, "node_id": node_id, "width": w, "height": h,
                        "sections": _extract_frame_sections(node),
                    })
                    return
            else:
                frames.append({
                    "name": name, "node_id": node_id, "width": w, "height": h,
                    "sections": _extract_frame_sections(node),
                })
                return

        for child in node.get("children", []):
            walk(child)

    pages = document.get("children", [])
    if not target_node_id:
        pages = pages[:1]  # first page only when no node targeted — avoids 400 from oversized /images requests
    for page in pages:
        for child in page.get("children", []):
            walk(child)

    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Typography token extractor
# ─────────────────────────────────────────────────────────────────────────────

def _collect_typography(file_data: dict) -> dict:
    """
    Walk the full document tree and collect unique typography values from TEXT nodes.
    Returns { fonts, sizes, weights, colors } — all sorted lists of unique values.
    """
    fonts: set[str]   = set()
    sizes: set[int]   = set()
    weights: set[int] = set()
    colors: set[str]  = set()

    def walk(node: dict) -> None:
        if node.get("type") == "TEXT":
            style = node.get("style", {})
            if ff := style.get("fontFamily"):
                fonts.add(ff)
            if fs := style.get("fontSize"):
                sizes.add(int(fs))
            if fw := style.get("fontWeight"):
                weights.add(int(fw))
            for fill in node.get("fills", []):
                c = fill.get("color", {})
                if c:
                    r = int(c.get("r", 0) * 255)
                    g = int(c.get("g", 0) * 255)
                    b = int(c.get("b", 0) * 255)
                    colors.add(f"rgb({r},{g},{b})")
        for child in node.get("children", []):
            walk(child)

    document = file_data.get("document", {})
    for page in document.get("children", []):
        for child in page.get("children", []):
            walk(child)

    return {
        "fonts":   sorted(fonts),
        "sizes":   sorted(sizes),
        "weights": sorted(weights),
        "colors":  sorted(colors),
    }