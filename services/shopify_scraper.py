import ipaddress
import json
import logging
import socket
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]

_PAGE_PATHS = {
    "home": "/",
    "product": "/products",
    "collection": "/collections/all",
    "cart": "/cart",
}

_VIEWPORT_WIDTH  = 1440
_VIEWPORT_HEIGHT = 900
_DEVICE_SCALE    = 2   # @2x — matches Figma export scale

# ── Injected before screenshot ────────────────────────────────────────────────
# Freezes all animations/transitions so every frame is deterministic.
# Also hides Shopify's own theme-preview bar (injected when the URL carries
# ?preview_theme_id=...) — it's Shopify admin chrome, not page content, and
# left visible it pushes/overlaps real content and shows up as a false diff
# against the Figma design on every page.
_STABILIZE_CSS = """
*, *::before, *::after {
    animation-duration:       0.001ms !important;
    animation-delay:          0.001ms !important;
    animation-iteration-count: 1       !important;
    transition-duration:      0.001ms !important;
    transition-delay:         0.001ms !important;
    scroll-behavior:          auto     !important;
    caret-color: transparent !important;
}
shopify-preview-bar,
#shopify-preview-bar,
#preview-bar-iframe,
[id^="shopify-preview-bar"] {
    display: none !important;
}
"""

# ── Freeze every <video> on its first frame instead of hiding it ─────────────
# Videos are real page content (hero backgrounds, testimonials) — display:none
# would delete them from the capture entirely, producing a blank/black region
# that gets misread as a layout defect. Pausing at frame 0 keeps a stable,
# deterministic frame visible without autoplay-timing flicker between captures.
_FREEZE_VIDEOS_JS = """
async () => {
    const videos = Array.from(document.querySelectorAll('video'));
    await Promise.all(videos.map(v => new Promise(resolve => {
        const settle = () => {
            try { v.pause(); v.currentTime = 0; } catch (e) {}
            resolve();
        };
        if (v.readyState >= 2) { settle(); return; }
        v.addEventListener('loadeddata', settle, { once: true });
        setTimeout(settle, 4000);
    })));
}
"""

# ── Force-load every lazy image and scroll to trigger intersection observers ──
_FORCE_LOAD_JS = """
async () => {
    // Disable lazy loading and force src on every img
    document.querySelectorAll('img').forEach(img => {
        img.loading = 'eager';
        const lazySrc = img.dataset.src || img.dataset.lazySrc
                      || img.dataset.original || img.dataset.srcset;
        if (lazySrc && !img.src) img.src = lazySrc;
        if (img.dataset.srcset) img.srcset = img.dataset.srcset;
    });

    // Force background images declared in data- attributes (common in Shopify themes)
    document.querySelectorAll('[data-bg],[data-background]').forEach(el => {
        const bg = el.dataset.bg || el.dataset.background;
        if (bg) el.style.backgroundImage = "url('" + bg + "')";
    });

    // Scroll through the full page to trigger IntersectionObserver-based loaders
    const pageH = Math.max(
        document.body.scrollHeight,
        document.documentElement.scrollHeight
    );
    const step = 400;
    for (let y = 0; y <= pageH; y += step) {
        window.scrollTo({ top: y, behavior: 'instant' });
        await new Promise(r => setTimeout(r, 60));
    }
    window.scrollTo({ top: 0, behavior: 'instant' });
    await new Promise(r => setTimeout(r, 400));
}
"""

# ── Reset carousels/sliders to their first slide and stop autoplay ───────────
# Splide/Slick/Swiper-based product carousels auto-rotate on a timer that CSS
# animation-duration overrides don't touch (it's JS setInterval, not a CSS
# transition). The Figma export is frozen on slide 1 — if the live carousel has
# already auto-advanced by the time the screenshot fires, we end up diffing two
# genuinely different slides and get a nonsense "different content" mismatch.
_RESET_CAROUSELS_JS = """
() => {
    document.querySelectorAll('.splide').forEach(el => {
        try {
            if (el.splide) {
                el.splide.Components?.Autoplay?.pause();
                el.splide.go(0);
            }
        } catch (e) {}
    });
    document.querySelectorAll('[class*="swiper"]').forEach(el => {
        try {
            if (el.swiper) {
                el.swiper.autoplay?.stop();
                el.swiper.slideTo(0, 0);
            }
        } catch (e) {}
    });
    document.querySelectorAll('[class*="slick-"]').forEach(el => {
        try {
            if (window.jQuery && window.jQuery(el).slick) {
                window.jQuery(el).slick('slickPause');
                window.jQuery(el).slick('slickGoTo', 0);
            }
        } catch (e) {}
    });
}
"""

# ── Dismiss popups/modals before screenshotting ──────────────────────────────
# Sales popups, newsletter modals, cookie banners, and similar transient overlays
# are a normal part of a live storefront but have no Figma equivalent — comparing
# a page with one still open produces a false "everything is different" diff.
# This runs a best-effort dismissal pass: click known close buttons, force-hide
# known popup-provider containers, then hide any large fixed/absolute overlay
# as a generic fallback for unrecognized popup apps.
_DISMISS_POPUPS_JS = """
() => {
    const CLOSE_SELECTORS = [
        '[aria-label="Close" i]', 'button[class*="close" i]', 'a[class*="close" i]',
        '[class*="close-button" i]', '[class*="modal-close" i]', '[class*="popup-close" i]',
        '[data-testid*="close" i]', 'button[title*="close" i]', '[class*="dismiss" i]',
    ];
    for (const sel of CLOSE_SELECTORS) {
        document.querySelectorAll(sel).forEach(btn => {
            const rect = btn.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                try { btn.click(); } catch (e) {}
            }
        });
    }

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    // Known popup/consent-banner providers — force-hide regardless of close button.
    const KNOWN_POPUP_SELECTORS = [
        '[id*="klaviyo" i]', '[class*="klaviyo-form" i]',
        '[id*="privy" i]', '[class*="privy" i]',
        '[id*="justuno" i]', '[id*="wheelio" i]', '[id*="optin" i]',
        '[class*="om-overlay" i]', '[class*="newsletter-popup" i]',
        '[class*="popup-overlay" i]', '[id*="popup" i]', '[class*="modal-overlay" i]',
        '#shopify-pc__banner', '[id*="cookie-consent" i]', '[class*="cookie-banner" i]',
    ];
    for (const sel of KNOWN_POPUP_SELECTORS) {
        document.querySelectorAll(sel).forEach(el => {
            el.style.setProperty('display', 'none', 'important');
        });
    }

    // Generic fallback: any large FIXED overlay is almost certainly a
    // popup/modal, not real page content — sticky headers/announcement bars are
    // excluded by the height + coverage thresholds below.
    // Deliberately position:fixed only, NOT absolute — a fixed element is
    // pinned to the viewport regardless of scroll, which is what a real
    // popup/modal needs to stay visible and centered. position:absolute is
    // the normal way in-page content (e.g. a hero's background image layered
    // behind its text within a relatively-positioned section) is laid out, and
    // can easily cover 80%+ of the viewport without being an overlay at all —
    // including 'absolute' here false-positived on real hero content.
    const vw = window.innerWidth, vh = window.innerHeight;
    const viewportArea = vw * vh;
    document.querySelectorAll('body *').forEach(el => {
        const style = window.getComputedStyle(el);
        if (style.position !== 'fixed') return;
        if (style.display === 'none' || style.visibility === 'hidden') return;
        const rect = el.getBoundingClientRect();
        const area = rect.width * rect.height;
        if (area <= 0) return;
        const coverage = area / viewportArea;
        if (coverage > 0.22 && rect.height > 150) {
            const zIndex = parseInt(style.zIndex, 10) || 0;
            if (zIndex >= 100 || coverage > 0.5) {
                el.style.setProperty('display', 'none', 'important');
            }
        }
    });
}
"""

# ── True once every Splide carousel has finished mounting ───────────────────
# splide--fade carousels keep every slide but the active one at opacity:0 until
# Splide's JS mount() completes and tags it `is-initialized`. If our screenshot
# fires before that mount finishes (a real race under headless Chromium), the
# hero/slideshow region captures as a blank opacity:0 slide even though the
# same page renders fine in an interactive browser.
_SPLIDE_READY_JS = """
() => Array.from(document.querySelectorAll('.splide'))
    .every(el => el.classList.contains('is-initialized'))
"""

# ── True when every visible <img> has finished decoding ──────────────────────
_IMAGES_READY_JS = """
() => Array.from(document.querySelectorAll('img'))
    .filter(img => img.offsetParent !== null)   // only visible images
    .every(img => img.complete && img.naturalWidth > 0)
"""

# ── Extract leaf text elements with computed styles (for typography diffing) ─
# A "leaf" text element has no element children of its own — this targets the
# innermost node actually carrying the text run, matching Figma TEXT nodes.
_EXTRACT_TEXT_NODES_JS = """
() => {
    const SKIP_TAGS = new Set(['SCRIPT','STYLE','NOSCRIPT','SVG','PATH','TEMPLATE','BR']);
    const results = [];
    const scrollY = window.pageYOffset || 0;
    const MAX_NODES = 400;

    const all = document.querySelectorAll('body *');
    for (const el of all) {
        if (results.length >= MAX_NODES) break;
        if (SKIP_TAGS.has(el.tagName)) continue;

        const hasElementChild = Array.from(el.children).some(c => !SKIP_TAGS.has(c.tagName));
        if (hasElementChild) continue;

        const text = (el.textContent || '').trim();
        if (!text) continue;

        const rect = el.getBoundingClientRect();
        if (rect.width < 2 || rect.height < 2) continue;
        // Skip elements outside the viewport horizontally (hidden/off-canvas menus etc.)
        if (rect.width > 4000 || rect.height > 4000) continue;

        const style = window.getComputedStyle(el);
        if (style.visibility === 'hidden' || style.display === 'none' || parseFloat(style.opacity) === 0) continue;

        const entry = {
            text: text.slice(0, 120),
            x: Math.round(rect.left),
            y: Math.round(rect.top + scrollY),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
            font_family: style.fontFamily,
            font_weight: parseInt(style.fontWeight, 10) || style.fontWeight,
            font_size: parseFloat(style.fontSize),
            color: style.color,
        };

        // A short/icon-like text node (a lone "!", a single glyph in a badge)
        // has a tight rect that's much smaller than the padded visual
        // container it actually sits inside (a callout, a chip). Spacing
        // measured from the raw glyph rect reads the container's own padding
        // as part of the gap to whatever's next to it. Mirrors the equivalent
        // fix on the Figma extraction side — same padding-window bounds, so
        // both sides pick a comparable container depth.
        //
        // Only qualifies when the parent's OTHER text-bearing children sit on
        // roughly the same row (an icon next to a one-line label) — not
        // stacked below/above (a heading with its own paragraph inside one
        // bordered box). Using the whole box as just the heading's own
        // "spacing box" would silently absorb the paragraph's space into the
        // heading's effective bottom edge, corrupting gaps measured against
        // it — a confirmed regression from the first version of this fix.
        const parent = el.parentElement;
        if (parent) {
            const prect = parent.getBoundingClientRect();
            const pad = prect.height - rect.height;
            const sameRow = Array.from(parent.children).every(sib => {
                if (sib === el) return true;
                if (!(sib.textContent || '').trim()) return true;
                const srect = sib.getBoundingClientRect();
                if (srect.width < 2 || srect.height < 2) return true;
                return Math.abs(srect.top - rect.top) <= 20;
            });
            if (pad > 4 && pad <= 150 && prect.width >= rect.width && sameRow) {
                entry.spacing_box = {
                    x: Math.round(prect.left),
                    y: Math.round(prect.top + scrollY),
                    width: Math.round(prect.width),
                    height: Math.round(prect.height),
                };
            }
        }

        results.push(entry);
    }
    return results;
}
"""

# ── Extract rendered images and button-like elements with their real boxes ──
# Used for dimension/spacing comparison against Figma — measuring the actual
# rendered rect, same philosophy as text-node extraction above: a measured
# fact, not a vision-model guess from a screenshot.
_EXTRACT_VISUAL_ELEMENTS_JS = """
() => {
    const scrollY = window.pageYOffset || 0;
    const isVisible = (el, rect) => {
        if (rect.width < 4 || rect.height < 4) return false;
        const style = window.getComputedStyle(el);
        if (style.visibility === 'hidden' || style.display === 'none') return false;
        if (parseFloat(style.opacity) === 0) return false;
        return true;
    };

    const images = [];
    document.querySelectorAll('img').forEach(img => {
        const rect = img.getBoundingClientRect();
        if (!isVisible(img, rect)) return;
        const src = (img.currentSrc || img.src || '').split('?')[0].split('/').pop() || '';
        const alt = (img.alt || '').trim();
        images.push({
            alt,
            src,
            label: alt || src,   // common matching key with Figma image nodes
            x: Math.round(rect.left),
            y: Math.round(rect.top + scrollY),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
        });
    });

    const BUTTON_SELECTORS =
        'button, a.button, a[class*="btn" i], input[type="submit"], input[type="button"], [role="button"]';
    const buttons = [];
    const seen = new Set();
    document.querySelectorAll(BUTTON_SELECTORS).forEach(el => {
        const rect = el.getBoundingClientRect();
        if (!isVisible(el, rect)) return;
        const label = (el.innerText || el.value || '').trim().slice(0, 80);
        const key = Math.round(rect.left) + ',' + Math.round(rect.top + scrollY) + ',' + label;
        if (seen.has(key)) return;   // de-dupe nested matches (e.g. a <button> whose child matched too)
        seen.add(key);
        buttons.push({
            label,
            x: Math.round(rect.left),
            y: Math.round(rect.top + scrollY),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
        });
    });

    return { images, buttons };
}
"""

# ── Extract DOM sections with absolute coordinates ───────────────────────────
_EXTRACT_SECTIONS_JS = """
() => {
    const SELECTORS = [
        '.shopify-section',
        'header[class], header[id], header[role="banner"]',
        'footer[class], footer[id], footer[role="contentinfo"]',
        'main > section, main > div[class], main > article',
        '[data-section-type]',
    ];

    const seen  = new Set();
    const sections = [];

    for (const sel of SELECTORS) {
        document.querySelectorAll(sel).forEach(el => {
            const rect = el.getBoundingClientRect();
            const scrollY = window.pageYOffset || 0;
            const absY = Math.round(rect.top + scrollY);
            const w    = Math.round(el.offsetWidth);
            const h    = Math.round(el.offsetHeight);

            if (w < 100 || h < 80) return;  // skip tiny/hidden elements

            // Deduplicate: bucket to 10px grid so closely overlapping wrappers merge
            const key = `${Math.round(absY / 10)}_${Math.round(h / 10)}`;
            if (seen.has(key)) return;
            seen.add(key);

            const headingEl = el.querySelector('h1,h2,h3');
            sections.push({
                index:       sections.length,
                tag:         el.tagName.toLowerCase(),
                id:          el.id || '',
                classes:     Array.from(el.classList).join(' '),
                sectionType: (el.dataset && (el.dataset.section || el.dataset.sectionType)) || '',
                heading:     headingEl ? (headingEl.innerText || '').trim().slice(0, 80) : '',
                textSnippet: (el.innerText || '').trim().slice(0, 150),
                x:           Math.max(0, Math.round(rect.left)),
                y:           absY,
                width:       w,
                height:      h,
            });
        });
    }

    sections.sort((a, b) => a.y - b.y);
    return sections;
}
"""


def validate_url(url: str) -> None:
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


def _save_debug(job_id: str, filename: str, data: bytes | str) -> None:
    """Persist a debug artifact to artifacts/debug/{job_id}/."""
    try:
        debug_dir = Path("artifacts") / "debug" / (job_id or "tmp")
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / filename
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_bytes(data)
    except Exception as exc:
        logger.debug(f"Debug save failed ({filename}): {exc}")


def _screenshot_page(
    context,
    url: str,
    page_name: str,
    timeout_ms: int,
    job_id: str = "",
    viewport_width: int = _VIEWPORT_WIDTH,
) -> tuple[bytes, list[dict], list[dict], list[dict], list[dict]]:
    """
    Fully stabilized page capture.

    Returns:
        (full_page_screenshot_bytes, sections_list, text_nodes_list,
         image_elements_list, button_elements_list)

    Sections list is a list of dicts, each with DOM metadata + a 'screenshot' key
    holding element-level bytes (or None if capture failed for that element).

    Text nodes list is leaf text elements with computed styles (font family/
    weight/size/color), used for precise typography diffing against Figma specs.

    Image/button element lists carry each element's real rendered box (x, y,
    width, height) plus a label (alt text or visible caption), used for
    dimension and spacing diffing against Figma.
    """
    # page_name may contain "/" for custom paths (e.g. "pages/state") — sanitize
    # before using it in a filename so it can't be misread as a subdirectory.
    safe_name = page_name.replace("/", "-").replace("\\", "-")

    page = None
    try:
        page = context.new_page()
        if viewport_width != _VIEWPORT_WIDTH:
            page.set_viewport_size({"width": viewport_width, "height": _VIEWPORT_HEIGHT})

        # ── 1. Navigate — prefer networkidle for JS-heavy stores ─────────────
        try:
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            logger.info(f"networkidle reached — {page_name}")
        except Exception:
            # Fallback: domcontentloaded + extra wait (catches stores with long-poll WS)
            logger.warning(f"networkidle timeout for {page_name} — falling back to load+wait")
            try:
                page.goto(url, wait_until="load", timeout=timeout_ms)
            except Exception:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(3000)

        # ── 2. Inject stabilization CSS ──────────────────────────────────────
        page.add_style_tag(content=_STABILIZE_CSS)

        # ── 3. Force-load lazy images + scroll through page ───────────────────
        try:
            page.evaluate(_FORCE_LOAD_JS)
        except Exception as exc:
            logger.warning(f"Force-load JS failed for {page_name}: {exc}")
            page.wait_for_timeout(1500)

        # ── 4. Wait for fonts ─────────────────────────────────────────────────
        try:
            page.evaluate("async () => { await document.fonts.ready; }")
        except Exception:
            pass

        # ── 5. Wait for images to finish decoding ─────────────────────────────
        try:
            page.wait_for_function(_IMAGES_READY_JS, timeout=8000)
        except Exception:
            logger.debug(f"Some images still loading after timeout — {page_name}, continuing")

        # ── 5a2. Wait for Splide carousels to finish mounting ─────────────────
        # Must happen before the carousel-reset step below — go(0) on an
        # unmounted instance is a silent no-op, and splide--fade slides sit at
        # opacity:0 until mount finishes tagging the active one.
        try:
            page.wait_for_function(_SPLIDE_READY_JS, timeout=5000)
        except Exception:
            logger.debug(f"Splide mount wait timed out — {page_name}, continuing")

        # ── 5b. Freeze videos on their first frame (see _FREEZE_VIDEOS_JS) ────
        try:
            page.evaluate(_FREEZE_VIDEOS_JS)
        except Exception as exc:
            logger.debug(f"Video freeze failed — {page_name}: {exc}")

        # ── 5c. Reset carousels to slide 1, stop autoplay ─────────────────────
        try:
            page.evaluate(_RESET_CAROUSELS_JS)
            page.wait_for_timeout(200)
        except Exception as exc:
            logger.debug(f"Carousel reset failed — {page_name}: {exc}")

        # ── 6. Final settle: scroll back to top, brief render pause ──────────
        page.evaluate("window.scrollTo({ top: 0, behavior: 'instant' })")
        page.wait_for_timeout(600)

        # ── 6b. Dismiss popups/modals before screenshotting ───────────────────
        # Run twice with a short wait between — catches sequential popups
        # (e.g. an age-gate followed by a newsletter modal) and gives close
        # animations time to finish before the second pass re-checks.
        try:
            page.evaluate(_DISMISS_POPUPS_JS)
            page.wait_for_timeout(400)
            page.evaluate(_DISMISS_POPUPS_JS)
            page.wait_for_timeout(300)
        except Exception as exc:
            logger.debug(f"Popup dismissal failed — {page_name}: {exc}")

        # ── 7. Full-page screenshot ───────────────────────────────────────────
        # full_page=True captures the ENTIRE document height, not just the 900px viewport.
        # This is critical for matching against full Figma frames.
        full_screenshot = page.screenshot(
            full_page=True,
            animations="disabled",
            timeout=timeout_ms,
        )
        logger.info(
            f"Full-page screenshot OK — {page_name}, {len(full_screenshot)} bytes"
        )

        if job_id:
            _save_debug(job_id, f"full_{safe_name}.png", full_screenshot)

        # ── 8. Extract leaf text nodes (computed styles) for typography diffing ─
        # Must run before the section-capture loop below, which scrolls the page
        # around and would throw off getBoundingClientRect() coordinates.
        text_nodes: list[dict] = []
        try:
            text_nodes = page.evaluate(_EXTRACT_TEXT_NODES_JS)
            logger.info(f"Text nodes found — {page_name}: {len(text_nodes)}")
        except Exception as exc:
            logger.warning(f"Text node extraction failed — {page_name}: {exc}")

        # ── 8b. Extract rendered images/buttons for dimension+spacing diffing ──
        image_elements: list[dict] = []
        button_elements: list[dict] = []
        try:
            visual_elements = page.evaluate(_EXTRACT_VISUAL_ELEMENTS_JS)
            image_elements = visual_elements.get("images", [])
            button_elements = visual_elements.get("buttons", [])
            logger.info(
                f"Visual elements found — {page_name}: "
                f"images={len(image_elements)}, buttons={len(button_elements)}"
            )
        except Exception as exc:
            logger.warning(f"Visual element extraction failed — {page_name}: {exc}")

        # ── 9. Extract DOM sections ───────────────────────────────────────────
        sections: list[dict] = []
        try:
            raw_sections = page.evaluate(_EXTRACT_SECTIONS_JS)
            logger.info(f"DOM sections found — {page_name}: {len(raw_sections)}")

            if job_id:
                _save_debug(
                    job_id,
                    f"sections_{safe_name}.json",
                    json.dumps(raw_sections, ensure_ascii=False, indent=2),
                )

            # ── 9. Capture each section with its element handle ───────────────
            for sec in raw_sections:
                sec_screenshot = None
                try:
                    # Build a locator using compound attribute match
                    # Find the element by matching its DOM position via JS handle
                    handle = page.evaluate_handle(
                        """(sec) => {
                            const candidates = [];
                            [
                                '.shopify-section',
                                'header', 'footer',
                                'main > section', 'main > div', 'main > article',
                                '[data-section-type]',
                            ].forEach(sel => {
                                document.querySelectorAll(sel).forEach(el => candidates.push(el));
                            });
                            // Match by y-position and height (±8px tolerance)
                            return candidates.find(el => {
                                const rect = el.getBoundingClientRect();
                                const absY = rect.top + (window.pageYOffset || 0);
                                return Math.abs(Math.round(absY) - sec.y) < 8
                                    && Math.abs(Math.round(el.offsetHeight) - sec.height) < 8;
                            }) || null;
                        }""",
                        sec,
                    )

                    element = handle.as_element()
                    if element:
                        # Scroll element into view before capture
                        element.scroll_into_view_if_needed(timeout=3000)
                        page.wait_for_timeout(150)
                        sec_screenshot = element.screenshot(
                            animations="disabled",
                            timeout=10000,
                        )
                        if job_id and sec_screenshot:
                            _save_debug(
                                job_id,
                                f"section_{safe_name}_{sec['index']:02d}.png",
                                sec_screenshot,
                            )
                except Exception as exc:
                    logger.debug(
                        f"Section capture failed — {page_name}[{sec.get('index')}]: {exc}"
                    )

                sections.append({**sec, "screenshot": sec_screenshot})

        except Exception as exc:
            logger.warning(f"Section extraction failed — {page_name}: {exc}")

        return full_screenshot, sections, text_nodes, image_elements, button_elements

    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass


def capture_pages(
    base_url: str,
    pages: list[str],
    password: str | None = None,
    timeout_ms: int = 45000,
    job_id: str = "",
    page_widths: dict[str, int] | None = None,
) -> list[dict]:
    """
    Sync Playwright page capture — must run in a worker thread, not the asyncio loop.

    page_widths: optional {page_name: viewport_width} map, used so each page is
      captured at the same logical width as its matched Figma frame. Pages not
      present in the map fall back to _VIEWPORT_WIDTH.

    Returns list of dicts:
      page, url, screenshot (bytes), sections (list[dict]), error (str|None)

    Each section dict contains:
      index, tag, id, classes, sectionType, heading, textSnippet,
      x, y, width, height, screenshot (bytes|None)
    """
    validate_url(base_url)
    page_widths = page_widths or {}
    parsed_base = urlparse(base_url)
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
            ],
        )
        try:
            context = browser.new_context(
                viewport={"width": _VIEWPORT_WIDTH, "height": _VIEWPORT_HEIGHT},
                device_scale_factor=_DEVICE_SCALE,
                # Disable web security for Shopify stores with strict CSP
                java_script_enabled=True,
                bypass_csp=True,
            )

            # ── Unlock password-protected store ─────────────────────────────
            if password:
                unlock_url = urlunparse(parsed_base._replace(path="/password", query=""))
                pw_page = context.new_page()
                try:
                    pw_page.goto(unlock_url, wait_until="domcontentloaded", timeout=timeout_ms)
                    pwd_input = pw_page.locator("input[type='password']")
                    if pwd_input.count() > 0:
                        pwd_input.fill(password)
                        pw_page.locator("input[type='submit'], button[type='submit']").first.click()
                        pw_page.wait_for_load_state("networkidle", timeout=timeout_ms)
                        logger.info("Shopify store unlocked with password")
                finally:
                    pw_page.close()

            browser_alive = True
            for page_name in pages:
                path = _PAGE_PATHS.get(page_name, f"/{page_name}")
                url = urlunparse(parsed_base._replace(path=path))

                viewport_width = page_widths.get(page_name, _VIEWPORT_WIDTH)

                if not browser_alive:
                    results.append({
                        "page": page_name,
                        "url": url,
                        "screenshot": None,
                        "sections": [],
                        "text_nodes": [],
                        "image_elements": [],
                        "button_elements": [],
                        "viewport_width": viewport_width,
                        "error": "Browser closed unexpectedly",
                    })
                    continue

                logger.info(f"Capturing {page_name} → {url} (viewport={viewport_width}px)")

                try:
                    screenshot, sections, text_nodes, image_elements, button_elements = _screenshot_page(
                        context, url, page_name, timeout_ms, job_id, viewport_width
                    )
                    results.append({
                        "page": page_name,
                        "url": url,
                        "screenshot": screenshot,
                        "sections": sections,
                        "text_nodes": text_nodes,
                        "image_elements": image_elements,
                        "button_elements": button_elements,
                        # DOM coordinates (getBoundingClientRect) are in logical CSS px —
                        # this is the reference width for them, NOT the raw screenshot's
                        # pixel width (which is viewport_width * device_scale_factor).
                        "viewport_width": viewport_width,
                        "error": None,
                    })
                    logger.info(
                        f"Capture OK — {page_name}, full={len(screenshot)} bytes, "
                        f"sections={len(sections)}, text_nodes={len(text_nodes)}, "
                        f"images={len(image_elements)}, buttons={len(button_elements)}"
                    )
                except Exception as e:
                    err_str = str(e)
                    logger.warning(f"Capture failed — {page_name}: {e}")
                    results.append({
                        "page": page_name,
                        "url": url,
                        "screenshot": None,
                        "sections": [],
                        "text_nodes": [],
                        "image_elements": [],
                        "button_elements": [],
                        "viewport_width": viewport_width,
                        "error": err_str,
                    })
                    if "closed" in err_str.lower() or "Target closed" in err_str:
                        logger.error("Browser closed unexpectedly — stopping")
                        browser_alive = False

        finally:
            browser.close()

    return results