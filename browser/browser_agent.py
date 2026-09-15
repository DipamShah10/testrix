import asyncio
from urllib.parse import urlsplit, urlunsplit

from playwright.async_api import BrowserContext, Page, async_playwright

from qa.models import BrowserObservation
from utils.screenshot_manager import save_screenshot


DEVICE_VIEWPORTS = {
    "desktop": {"width": 1440, "height": 900},
    "tablet": {"width": 834, "height": 1112},
    "mobile": {"width": 390, "height": 844},
}


class BrowserAgent:
    def __init__(self, headless: bool, timeout_ms: int, screenshot_dir) -> None:
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.screenshot_dir = screenshot_dir
        self.max_parallel_pages = 4

    async def inspect(self, base_url: str, pages: list[str]) -> list[BrowserObservation]:
        observations: list[BrowserObservation] = []
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=self.headless)
            try:
                for viewport_name, viewport in DEVICE_VIEWPORTS.items():
                    context = await browser.new_context(viewport=viewport)
                    context.set_default_timeout(self.timeout_ms)
                    observations.extend(await self._inspect_with_context(context, base_url, pages, viewport_name))
                    await context.close()
            finally:
                await browser.close()
        return observations

    async def _inspect_with_context(
        self,
        context: BrowserContext,
        base_url: str,
        pages: list[str],
        viewport_name: str,
    ) -> list[BrowserObservation]:
        semaphore = asyncio.Semaphore(self.max_parallel_pages)
        tasks = [
            self._inspect_route(context, base_url, route, viewport_name, semaphore)
            for route in pages
        ]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _build_page_url(base_url: str, route: str) -> str:
        parts = urlsplit(base_url)
        path = "/" + route.strip("/") if route not in ("", "/") else "/"
        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))

    async def _inspect_route(
        self,
        context: BrowserContext,
        base_url: str,
        route: str,
        viewport_name: str,
        semaphore: asyncio.Semaphore,
    ) -> BrowserObservation:
        async with semaphore:
            page_url = self._build_page_url(base_url, route)
            page = None
            console_errors: list[str] = []
            network_failures: list[str] = []

            try:
                page = await context.new_page()
                page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
                page.on("requestfailed", lambda req: network_failures.append(f"{req.method} {req.url}"))
                await page.goto(page_url, wait_until="networkidle")
                notes = await self._simulate_actions(page)
                accessibility_notes = await self._quick_accessibility_scan(page)
                screenshot = await page.screenshot(full_page=True)
                screenshot_path = save_screenshot(
                    screenshot,
                    self.screenshot_dir,
                    f"{viewport_name}-{route.strip('/').replace('/', '-') or 'home'}",
                )
                return BrowserObservation(
                    page=route,
                    url=page_url,
                    viewport=viewport_name,
                    screenshot_path=str(screenshot_path),
                    console_errors=console_errors,
                    network_failures=network_failures,
                    interaction_notes=notes,
                    accessibility_notes=accessibility_notes,
                )
            except Exception as exc:  # noqa: BLE001
                return BrowserObservation(
                    page=route,
                    url=page_url,
                    viewport=viewport_name,
                    screenshot_path=None,
                    console_errors=console_errors,
                    network_failures=network_failures,
                    error=str(exc),
                )
            finally:
                if page is not None:
                    await page.close()

    async def _simulate_actions(self, page: Page) -> list[str]:
        notes: list[str] = []
        clickable = page.locator("button, a, [role='button']")
        count = min(await clickable.count(), 3)
        for idx in range(count):
            try:
                element = clickable.nth(idx)
                text = (await element.inner_text()).strip()[:80]
                await element.click(timeout=2500)
                notes.append(f"Clicked element: {text or 'unnamed'}")
            except Exception:  # noqa: BLE001
                continue

        forms = page.locator("input, textarea")
        forms_count = min(await forms.count(), 3)
        for idx in range(forms_count):
            try:
                field = forms.nth(idx)
                placeholder = await field.get_attribute("placeholder")
                await field.fill("qa-test")
                notes.append(f"Filled field: {placeholder or 'no-placeholder'}")
            except Exception:  # noqa: BLE001
                continue
        return notes

    async def _quick_accessibility_scan(self, page: Page) -> list[str]:
        notes: list[str] = []
        missing_alt_count = await page.locator("img:not([alt])").count()
        if missing_alt_count:
            notes.append(f"Found {missing_alt_count} image(s) without alt text.")

        empty_button_count = await page.locator("button:has-text('')").count()
        if empty_button_count:
            notes.append(f"Found {empty_button_count} potentially unlabeled button(s).")

        unlabeled_input_count = await page.locator("input:not([aria-label]):not([id])").count()
        if unlabeled_input_count:
            notes.append(f"Found {unlabeled_input_count} input(s) without id/aria-label.")

        skipped_heading = await page.evaluate(
            """
() => {
  const headings = Array.from(document.querySelectorAll("h1,h2,h3,h4,h5,h6"));
  let previous = 0;
  for (const h of headings) {
    const current = Number(h.tagName.substring(1));
    if (previous && current > previous + 1) return true;
    previous = current;
  }
  return false;
}
"""
        )
        if skipped_heading:
            notes.append("Detected heading level jumps (potential accessibility issue).")

        axe_notes = await self._run_axe_scan(page)
        notes.extend(axe_notes)
        return notes

    async def _run_axe_scan(self, page: Page) -> list[str]:
        notes: list[str] = []
        try:
            await page.add_script_tag(url="https://cdn.jsdelivr.net/npm/axe-core@4.9.1/axe.min.js")
            result = await page.evaluate(
                """
async () => {
  if (!window.axe) return { violations: [] };
  return await window.axe.run(document, {
    runOnly: {
      type: "tag",
      values: ["wcag2a", "wcag2aa"]
    }
  });
}
"""
            )
            violations = result.get("violations", [])
            if violations:
                notes.append(f"axe-core reported {len(violations)} WCAG violation group(s).")
                for item in violations[:5]:
                    impact = item.get("impact", "unknown")
                    desc = item.get("description", "No description")
                    notes.append(f"axe [{impact}] {item.get('id')}: {desc}")
        except Exception:
            notes.append("axe-core scan unavailable for this page/context.")
        return notes
