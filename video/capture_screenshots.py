"""Capture demo screenshots of the Opsly UI (light theme) at localhost:8501."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

OUT = Path(__file__).parent / "public/assets/screens"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://localhost:8501"
VIEWPORT = {"width": 1440, "height": 900}


async def shot(page, name: str):
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await page.wait_for_timeout(600)
    await page.screenshot(path=str(OUT / name), full_page=False)
    print(f"  ✓ {name}")


async def wait_for_response(page, timeout_ms=90_000):
    try:
        await page.wait_for_selector('[data-testid="stStatusWidget"]', timeout=8000)
        await page.wait_for_selector(
            '[data-testid="stStatusWidget"]', state="hidden", timeout=timeout_ms,
        )
    except Exception:
        await page.wait_for_timeout(min(timeout_ms, 80_000))


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport=VIEWPORT)
        page = await ctx.new_page()

        # 1. Landing
        print("Loading landing page …")
        await page.goto(BASE, wait_until="networkidle", timeout=30_000)
        await page.wait_for_timeout(3000)
        await shot(page, "01_landing.png")

        # 2. Inheritance query
        print("Query: inherit from APIRouter …")
        await page.locator("textarea").first.fill("What classes inherit from APIRouter?")
        await page.locator("textarea").first.press("Enter")
        await page.wait_for_timeout(2000)
        await shot(page, "02_query_submitted.png")
        print("  waiting …")
        await wait_for_response(page, 90_000)
        await page.wait_for_timeout(1500)
        await shot(page, "03_query_result.png")

        # 3. Graph view
        print("Graph view …")
        await page.get_by_text("🕸️ Graph View").click()
        await page.wait_for_timeout(3000)
        await shot(page, "04_graph_view.png")

        # 4. Agent activity
        print("Agent activity …")
        await page.get_by_text("🎯 Agent Activity").click()
        await page.wait_for_timeout(1500)
        await shot(page, "05_agent_activity.png")

        # 5. Lifecycle query (now fast-pathed)
        print("Query: FastAPI request lifecycle …")
        await page.locator("textarea").first.fill(
            "Explain the complete lifecycle of a request in FastAPI server"
        )
        await page.locator("textarea").first.press("Enter")
        await page.wait_for_timeout(2000)
        await shot(page, "06_lifecycle_query.png")
        print("  waiting …")
        await wait_for_response(page, 90_000)
        await page.wait_for_timeout(1500)
        await shot(page, "07_lifecycle_result.png")

        # 6. Rich graph
        print("Rich graph …")
        await page.get_by_text("🕸️ Graph View").click()
        await page.wait_for_timeout(4000)
        await shot(page, "08_rich_graph.png")

        await browser.close()
    print("\nDone →", OUT)


asyncio.run(main())
