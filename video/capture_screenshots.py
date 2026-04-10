"""Capture demo screenshots of the Opsly UI (light theme) at localhost:8501."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

OUT = Path(__file__).parent / "public/assets/screens"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://localhost:8501"
VIEWPORT = {"width": 1440, "height": 900}


async def shot(page, name: str):
    """Scroll to latest content then screenshot."""
    # Scroll the main content area to the bottom so latest chat is visible
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await page.wait_for_timeout(600)
    await page.screenshot(path=str(OUT / name), full_page=False)
    print(f"  ✓ {name}")


async def wait_for_response(page, timeout_ms=90_000):
    """Wait until the spinner is gone (response arrived)."""
    try:
        # Streamlit shows a running indicator while processing
        await page.wait_for_selector('[data-testid="stStatusWidget"]', timeout=8000)
        await page.wait_for_selector(
            '[data-testid="stStatusWidget"]',
            state="hidden",
            timeout=timeout_ms,
        )
    except Exception:
        # Fallback: just wait
        await page.wait_for_timeout(min(timeout_ms, 70_000))


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport=VIEWPORT)
        page = await ctx.new_page()

        # ── 1. Landing page ──────────────────────────────────────────────────
        print("Loading landing page …")
        await page.goto(BASE, wait_until="networkidle", timeout=30_000)
        await page.wait_for_timeout(3000)
        await shot(page, "01_landing.png")

        # ── 2. Submit inheritance query ───────────────────────────────────────
        print("Submitting query: inherit from APIRouter …")
        chat_input = page.locator("textarea").first
        await chat_input.fill("What classes inherit from APIRouter?")
        await chat_input.press("Enter")
        await page.wait_for_timeout(2000)
        await shot(page, "02_query_submitted.png")

        print("  Waiting for response …")
        await wait_for_response(page, 90_000)
        await page.wait_for_timeout(1500)
        await shot(page, "03_query_result.png")

        # ── 3. Graph View tab ────────────────────────────────────────────────
        print("Opening Graph View …")
        await page.get_by_text("🕸️ Graph View").click()
        await page.wait_for_timeout(3000)
        await shot(page, "04_graph_view.png")

        # ── 4. Agent Activity tab ─────────────────────────────────────────────
        print("Opening Agent Activity …")
        await page.get_by_text("🎯 Agent Activity").click()
        await page.wait_for_timeout(1500)
        await shot(page, "05_agent_activity.png")

        # ── 5. Second query: symbol context ──────────────────────────────────
        print("Submitting query: FastAPI symbol context …")
        chat_input2 = page.locator("textarea").first
        await chat_input2.fill("Show me the symbol context of the FastAPI class")
        await chat_input2.press("Enter")
        await page.wait_for_timeout(2000)
        await shot(page, "06_lifecycle_query.png")

        print("  Waiting for response …")
        await wait_for_response(page, 90_000)
        await page.wait_for_timeout(1500)
        await shot(page, "07_lifecycle_result.png")

        # ── 6. Rich graph ─────────────────────────────────────────────────────
        print("Opening Graph View for rich graph …")
        await page.get_by_text("🕸️ Graph View").click()
        await page.wait_for_timeout(4000)
        await shot(page, "08_rich_graph.png")

        await browser.close()
    print("\nAll screenshots saved to", OUT)


asyncio.run(main())
