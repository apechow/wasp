
import asyncio
from playwright.async_api import async_playwright

async def main():
    # Since the profile is in /tmp/playwright_chromiumdev_profile-KCESc8, we can launch a persistent context using that directory!
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir="/tmp/playwright_chromiumdev_profile-KCESc8",
            headless=True
        )
        page = await context.new_page()
        await page.goto("http://localhost:9999/f/allentown/28/a-note-from-a-gentleman-thief-1005")
        cookies = await context.cookies()
        print("COOKIES:", cookies)
        # Let's also check if we can click the upvote button
        # The form has two buttons: choice=1 and choice=-1
        # Let's see the vote status
        score = await page.locator(".score").text_content() if await page.locator(".score").count() > 0 else "None"
        print("SCORE:", score)
        await context.close()

asyncio.run(main())
