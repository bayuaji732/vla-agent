from __future__ import annotations

import asyncio
import base64
import logging
import random
from typing import Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from agent.config import get_settings
from agent.models import Action, ActionType, MarkedElement, Observation

logger = logging.getLogger(__name__)
cfg = get_settings()

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


class Actor:
    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=cfg.headless,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--disable-infobars",
                "--disable-extensions",
            ],
        )

        user_agent = random.choice(_USER_AGENTS)
        self._context = await self._browser.new_context(
            viewport={"width": cfg.viewport_width, "height": cfg.viewport_height},
            user_agent=user_agent,
            locale="en-US",
            timezone_id="Asia/Jakarta",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            },
        )

        await self._context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            window.chrome = { runtime: {} };
        """)

        self._page = await self._context.new_page()
        logger.info("Browser started (headless=%s)", cfg.headless)

    async def stop(self) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser stopped")

    # ── Observation ───────────────────────────────────────────────────────────

    async def observe(self) -> Observation:
        assert self._page
        try:
            await self._page.wait_for_load_state("domcontentloaded", timeout=cfg.browser_timeout_ms)
        except Exception:
            pass

        screenshot_bytes = await self._page.screenshot(full_page=False)
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
        elements = await self._extract_elements()

        return Observation(
            screenshot_b64=screenshot_b64,
            url=self._page.url,
            page_title=await self._page.title(),
            marked_elements=elements,
        )

    async def _extract_elements(self) -> list[MarkedElement]:
        assert self._page
        try:
            raw = await self._page.evaluate("""
            () => {
                const sel = 'a[href], button, input:not([type=hidden]), select, textarea, [role="button"], [role="link"], [role="menuitem"]';
                return Array.from(document.querySelectorAll(sel))
                    .slice(0, 60)
                    .map((el, idx) => {
                        const rect = el.getBoundingClientRect();
                        if (rect.width < 2 || rect.height < 2) return null;
                        return {
                            id: idx + 1,
                            tag: el.tagName.toLowerCase(),
                            role: el.getAttribute('role') || '',
                            text: (el.innerText || el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.value || '').trim().slice(0, 80),
                            href: el.href || null,
                            x: Math.round(rect.left),
                            y: Math.round(rect.top),
                            width: Math.round(rect.width),
                            height: Math.round(rect.height),
                        };
                    }).filter(Boolean);
            }
            """)
            return [MarkedElement(**e) for e in raw]
        except Exception as exc:
            logger.warning("Element extraction failed: %s", exc)
            return []

    # ── Action execution ──────────────────────────────────────────────────────

    async def execute(self, action: Action, elements: list[MarkedElement]) -> None:
        assert self._page
        page = self._page

        match action.type:
            case ActionType.NAVIGATE:
                url = action.url or ""
                if not url.startswith("http"):
                    url = "https://" + url
                logger.info("Navigating to %s", url)
                await page.goto(url, timeout=cfg.browser_timeout_ms, wait_until="domcontentloaded")
                await asyncio.sleep(cfg.navigate_delay_ms / 1000)

            case ActionType.CLICK:
                elem = self._find_element(action, elements)
                if elem:
                    cx = elem.x + elem.width // 2
                    cy = elem.y + elem.height // 2
                    await self._move_and_click(cx, cy)
                    logger.info("Clicked element_id=%s", action.element_id)
                elif action.selector:
                    await page.locator(action.selector).first.click(timeout=10_000)
                elif action.x is not None and action.y is not None:
                    await self._move_and_click(action.x, action.y)
                else:
                    raise ValueError(f"Cannot resolve click target: {action}")

            case ActionType.TYPE:
                elem = self._find_element(action, elements)
                text = action.text or ""
                if elem:
                    cx = elem.x + elem.width // 2
                    cy = elem.y + elem.height // 2
                    await self._move_and_click(cx, cy)
                    await page.keyboard.type(text, delay=80)
                    logger.info("Typed '%s'", text[:40])
                elif action.selector:
                    await page.locator(action.selector).first.fill(text)
                else:
                    raise ValueError(f"Cannot resolve type target: {action}")

            case ActionType.SCROLL:
                delta = -400 if action.direction == "up" else 400
                await page.mouse.wheel(0, delta)
                logger.info("Scrolled %s", action.direction)

            case ActionType.HOVER:
                elem = self._find_element(action, elements)
                if elem:
                    await page.mouse.move(elem.x + elem.width // 2, elem.y + elem.height // 2)

            case ActionType.KEY_PRESS:
                await page.keyboard.press(action.key or "Enter")
                logger.info("Key pressed: %s", action.key)

            case ActionType.WAIT:
                await asyncio.sleep(2)

            case ActionType.SCREENSHOT | ActionType.DONE | ActionType.FAIL:
                pass

            case _:
                logger.warning("Unknown action type: %s", action.type)

        # Global post-action pause (configurable, default 100ms)
        await asyncio.sleep(cfg.action_delay_ms / 1000)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_element(self, action: Action, elements: list[MarkedElement]) -> Optional[MarkedElement]:
        if action.element_id is not None:
            return next((e for e in elements if e.id == action.element_id), None)
        return None

    async def _move_and_click(self, x: int, y: int) -> None:
        """Minimal human-like mouse movement — configurable steps via mouse_steps."""
        assert self._page
        steps = cfg.mouse_steps
        if steps <= 1:
            await self._page.mouse.click(x, y)
            return
        sx = random.randint(200, 600)
        sy = random.randint(200, 500)
        for i in range(steps):
            t = (i + 1) / steps
            await self._page.mouse.move(
                int(sx + (x - sx) * t),
                int(sy + (y - sy) * t),
            )
            await asyncio.sleep(0.01)
        await self._page.mouse.click(x, y)