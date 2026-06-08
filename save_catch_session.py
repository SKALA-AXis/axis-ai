"""Catch 로그인 세션 저장.

사용법:
  uv run python save_catch_session.py

열린 브라우저에서 직접 로그인한 뒤 터미널에서 Enter를 누르면
axis-ai/.secrets/catch_storage_state.json 에 세션 쿠키가 저장된다.
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

SESSION_PATH = Path(__file__).resolve().parent / ".secrets" / "catch_storage_state.json"
CATCH_HOME_URL = "https://www.catch.co.kr/"


def main() -> None:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context(
            locale="ko-KR",
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()
        page.goto(CATCH_HOME_URL, wait_until="domcontentloaded", timeout=30_000)

        input("브라우저에서 Catch 로그인을 완료한 뒤 Enter를 누르세요: ")
        context.storage_state(path=str(SESSION_PATH))
        browser.close()

    print(f"Catch 세션 저장 완료: {SESSION_PATH}")


if __name__ == "__main__":
    main()
