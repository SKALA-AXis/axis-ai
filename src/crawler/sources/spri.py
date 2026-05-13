"""SPRi 소프트웨어정책연구소 산업동향 PDF 크롤러.

사용 예시:
.venv/bin/python src/crawler/spri_crawler.py --month 2026-04

기능:
1. SPRi SW중심사회 목록 페이지 접속
2. 터미널에서 입력한 월호 찾기
3. 해당 월호 상세 페이지 접속
4. PDF 다운로드 링크 추출
5. PDF 파일 저장
6. PDF 전체 텍스트 추출
7. 그림은 실제 그래프/도형 영역 중심으로 crop
8. 표는 캡션 아래 영역을 예전 방식으로 crop
9. 출처 위에 있는 시각자료도 보조적으로 crop
10. RawArticle 기반 공통 JSON 스키마로 저장
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import fitz  # PyMuPDF
import httpx
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crawler.base import RawArticle  # noqa: E402
from src.crawler.base_crawler import BaseCrawler  # noqa: E402
from src.crawler.parsers.pdf_payload import extract_pdf_payload  # noqa: E402
from src.crawler.playwright_client import PlaywrightClient  # noqa: E402

log = logging.getLogger(__name__)

SPRI_LIST_URL = "https://spri.kr/posts?code=magazine"
SPRI_BACKFILL_MAX_LIST_PAGES = int(os.getenv("SPRI_BACKFILL_MAX_LIST_PAGES", "20"))

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

CAPTION_PATTERN = re.compile(r"[\[【]?\s*(그림|표)\s*\d+[-–]?\d*\s*[\]】]?")
CAPTION_PATTERN_STRICT = re.compile(r"^\s*[\[【]?\s*(그림|표)\s*\d+[-–]?\d*\s*[\]】]?")
# 출처/자료 라인은 문서에 따라 "* 출처:" / "• 출처:" 같은 형태로 등장한다.
SOURCE_PATTERN = re.compile(r"^\s*[*•\u25cf\u2022]?\s*(출처|자료)\s*[:：]")


def make_id(source_type: str, source_name: str, title: str, url: str) -> str:
    raw = f"{source_type}|{source_name}|{title}|{url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    # PDF 텍스트에는 제어문자(BEL 등)가 섞여 나오는 경우가 있어 휴리스틱을 깨뜨린다.
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def safe_filename(text: str) -> str:
    text = re.sub(r"[^\w가-힣.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:120] or "spri_report"


def month_to_label(month: str) -> str:
    year, month_num = month.split("-")
    return f"{year}년{month_num}월호"


def month_to_title(month: str) -> str:
    year, month_num = month.split("-")
    return f"{year}년 {int(month_num)}월호"


async def fetch_text(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url)
    response.raise_for_status()
    return response.text


async def fetch_bytes(client: httpx.AsyncClient, url: str) -> bytes:
    response = await client.get(url)
    response.raise_for_status()
    return response.content


def spri_page_url(url: str, page_no: int) -> str:
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params["page"] = str(page_no)
    return urlunparse(parsed._replace(query=urlencode(params)))


def normalize_listing_url(base_url: str, href: str) -> str | None:
    if not href:
        return None

    url = urljoin(base_url, href.strip())
    parsed = urlparse(url)

    if parsed.netloc != urlparse(SPRI_LIST_URL).netloc:
        return None

    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if params.get("code") != "magazine":
        return None

    return urlunparse(parsed._replace(fragment=""))


def extract_pagination_links(list_html: str, list_url: str) -> list[str]:
    soup = BeautifulSoup(list_html, "html.parser")
    urls: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "page=" not in href and "posts" not in href:
            continue

        normalized = normalize_listing_url(list_url, href)
        if not normalized:
            continue

        page = dict(parse_qsl(urlparse(normalized).query, keep_blank_values=True)).get("page")
        if page is not None and not page.isdigit():
            continue

        urls.append(normalized)

    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        unique_urls.append(url)
    return unique_urls


async def find_issue_url_with_pagination(
    client: httpx.AsyncClient,
    month: str,
    *,
    max_pages: int = SPRI_BACKFILL_MAX_LIST_PAGES,
) -> str:
    playwright = PlaywrightClient()
    pending_urls = [SPRI_LIST_URL]
    pending_urls.extend(spri_page_url(SPRI_LIST_URL, page_no) for page_no in range(2, max_pages + 1))
    seen_urls: set[str] = set()

    while pending_urls and len(seen_urls) < max_pages:
        list_url = pending_urls.pop(0)
        normalized = normalize_listing_url(SPRI_LIST_URL, list_url) or list_url
        if normalized in seen_urls:
            continue
        seen_urls.add(normalized)

        list_html = await fetch_text(client, list_url)
        try:
            issue_url = find_issue_url(list_html, month)
            log.info("SPRi 월호 발견 | month=%s list_url=%s", month, list_url)
            return issue_url
        except ValueError:
            rendered_html = await playwright.fetch_html(list_url)
            if rendered_html:
                try:
                    issue_url = find_issue_url(rendered_html, month)
                    log.info(
                        "SPRi 월호 발견 (rendered) | month=%s list_url=%s",
                        month,
                        list_url,
                    )
                    return issue_url
                except ValueError:
                    list_html = rendered_html

        discovered_urls = [
            discovered_url
            for discovered_url in extract_pagination_links(list_html, list_url)
            if discovered_url not in seen_urls and discovered_url not in pending_urls
        ]
        pending_urls = discovered_urls + pending_urls

    raise ValueError(f"{month} 월호 상세 페이지를 찾지 못했습니다.")


def find_issue_url(list_html: str, month: str) -> str:
    soup = BeautifulSoup(list_html, "html.parser")

    target_label = month_to_label(month)
    target_title = month_to_title(month)
    target_title_no_zero = target_label.replace("년0", "년 ")

    candidates = [target_label, target_title, target_title_no_zero]

    block_tags = soup.find_all(["li", "tr", "article", "div"])
    matched_blocks = []

    for block in block_tags:
        block_text = normalize_text(block.get_text(" ", strip=True))
        if any(candidate in block_text for candidate in candidates):
            matched_blocks.append(block)

    matched_blocks = sorted(
        matched_blocks,
        key=lambda node: len(normalize_text(node.get_text(" ", strip=True))),
    )

    for block in matched_blocks:
        link = block.find("a", href=lambda href: href and "posts/view" in href)
        if link:
            return urljoin(SPRI_LIST_URL, link["href"])

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")

        if "posts/view" not in href:
            continue

        current = a

        for _ in range(8):
            if current is None:
                break

            nearby_text = normalize_text(current.get_text(" ", strip=True))

            if any(candidate in nearby_text for candidate in candidates):
                return urljoin(SPRI_LIST_URL, href)

            current = current.parent

    page_text = normalize_text(soup.get_text(" ", strip=True))

    if any(candidate in page_text for candidate in candidates):
        first_view_link = soup.find("a", href=lambda href: href and "posts/view" in href)
        if first_view_link:
            return urljoin(SPRI_LIST_URL, first_view_link["href"])

    raise ValueError(f"{month} 월호 상세 페이지를 찾지 못했습니다.")


def parse_published_at(detail_html: str) -> datetime | None:
    soup = BeautifulSoup(detail_html, "html.parser")
    page_text = soup.get_text("\n", strip=True)

    date_match = re.search(r"(\d{4})[.-](\d{2})[.-](\d{2})", page_text)

    if not date_match:
        return None

    year, month_num, day = date_match.groups()

    return datetime(
        int(year),
        int(month_num),
        int(day),
        0,
        0,
        0,
        tzinfo=timezone.utc,
    )


def extract_issue_title(detail_html: str, month: str) -> str:
    soup = BeautifulSoup(detail_html, "html.parser")

    title_node = soup.select_one("h1, h2, h3, .title, .view_tit")
    if title_node:
        title = normalize_text(title_node.get_text(" ", strip=True))
        if title:
            return title

    return month_to_title(month)


def extract_pdf_links(detail_html: str, issue_url: str) -> list[dict]:
    soup = BeautifulSoup(detail_html, "html.parser")

    pdf_items = []
    seen_urls = set()

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        link_text = normalize_text(a.get_text(" ", strip=True))

        is_pdf_link = "/download/" in href or href.lower().endswith(".pdf")

        if not is_pdf_link:
            continue

        pdf_url = urljoin(issue_url, href)

        if pdf_url in seen_urls:
            continue

        seen_urls.add(pdf_url)

        pdf_items.append(
            {
                "pdf_title": link_text or f"SPRi PDF {len(pdf_items) + 1}",
                "pdf_url": pdf_url,
            }
        )

    if not pdf_items:
        raise ValueError("PDF 다운로드 링크를 찾지 못했습니다.")

    return pdf_items


def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    doc = fitz.open(pdf_path)

    page_texts = []

    for page in doc:
        page_text = page.get_text("text")
        page_texts.append(page_text)

    page_count = doc.page_count
    doc.close()

    content = "\n\n".join(page_texts)
    content = normalize_text(content)

    return content, page_count


def rect_area(rect: fitz.Rect) -> float:
    return max(0.0, rect.width) * max(0.0, rect.height)


def overlap_ratio(rect_a: fitz.Rect, rect_b: fitz.Rect) -> float:
    inter = rect_a & rect_b
    inter_area = rect_area(inter)

    if inter_area <= 0:
        return 0.0

    smaller_area = min(rect_area(rect_a), rect_area(rect_b))
    if smaller_area <= 0:
        return 0.0

    return inter_area / smaller_area


def horizontal_overlap_ratio(rect_a: fitz.Rect, rect_b: fitz.Rect) -> float:
    inter = rect_a & rect_b
    if inter.width <= 0:
        return 0.0

    denom = min(max(0.0, rect_a.width), max(0.0, rect_b.width))
    if denom <= 0:
        return 0.0

    return max(0.0, inter.width) / denom


def is_duplicate_rect(
    rect: fitz.Rect,
    saved_rects: list[fitz.Rect],
    threshold: float = 0.45,
) -> bool:
    return any(overlap_ratio(rect, saved_rect) >= threshold for saved_rect in saved_rects)


def expand_rect(rect: fitz.Rect, page_rect: fitz.Rect, margin: float = 10) -> fitz.Rect:
    return fitz.Rect(
        max(page_rect.x0 + 20, rect.x0 - margin),
        max(page_rect.y0 + 20, rect.y0 - margin),
        min(page_rect.x1 - 20, rect.x1 + margin),
        min(page_rect.y1 - 20, rect.y1 + margin),
    )


def get_text_blocks(page) -> list[dict]:
    page_dict = page.get_text("dict")
    blocks = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        lines = block.get("lines", [])
        texts = []
        max_font_size = 0.0

        for line in lines:
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                if span_text:
                    texts.append(span_text)
                    try:
                        max_font_size = max(max_font_size, float(span.get("size", 0.0) or 0.0))
                    except Exception:
                        pass

        block_text = normalize_text(" ".join(texts))
        if not block_text:
            continue

        blocks.append(
            {
                "text": block_text,
                "rect": fitz.Rect(block["bbox"]),
                "max_font_size": max_font_size,
            }
        )

    return blocks


def find_caption_blocks(page) -> list[dict]:
    captions = []

    for block in get_text_blocks(page):
        text = block["text"]
        # 본문 중간의 "[그림 n 참조]" 같은 언급을 캡션으로 오인하지 않도록
        # 캡션은 블록 시작부에 등장하는 경우만 인정한다.
        match = CAPTION_PATTERN_STRICT.match(text)

        if not match:
            continue

        captions.append(
            {
                "caption": text,
                "caption_label": match.group(0).strip(),
                "visual_type": match.group(1),
                "rect": block["rect"],
            }
        )

    return captions


def find_source_blocks(page) -> list[dict]:
    sources = []

    for block in get_text_blocks(page):
        text = block["text"]

        if not SOURCE_PATTERN.search(text):
            continue

        sources.append(
            {
                "source_text": text,
                "rect": block["rect"],
            }
        )

    return sources


def is_heading_like(text: str) -> bool:
    text = normalize_text(text)

    if not text:
        return False

    heading_patterns = [
        r"^[ⅠⅡⅢⅣⅤⅥ]+\s",
        r"^[IVX]+\s",
        r"^\d+\.\s",
        r"^참고문헌",
        r"^FOCUS",
        r"^ISSUE",
        r".*\[?\s*참고\s*\d+\s*\]?.*",
        r"^<\s*방안",
        r"^\(?단기\)?",
        r"^\(?중기\)?",
        r"^\(?장기\)?",
    ]

    return any(re.search(pattern, text) for pattern in heading_patterns)


def is_paragraph_like_text_block(text: str, rect: fitz.Rect, page_rect: fitz.Rect) -> bool:
    """본문 설명 문단/참고 박스 같은 큰 텍스트 블록을 대략 감지한다."""
    text = normalize_text(text)
    if not text:
        return False

    if CAPTION_PATTERN.search(text) or SOURCE_PATTERN.search(text) or is_heading_like(text):
        return False

    # 긴 문단 + 넓은 폭 + 일정 높이 이상이면 본문일 확률이 높다.
    if len(text) < 60:
        return False

    if rect.width < page_rect.width * 0.28:
        return False

    # 설명/각주 문장은 한 줄로도 등장할 수 있다(특히 그림 아래).
    # 높이가 낮아도 "긴 문장 + 넓은 폭"이면 본문/설명으로 간주한다.
    if rect.height < page_rect.height * 0.03:
        if len(text) < 90 or rect.width < page_rect.width * 0.6:
            return False

    # bullet/설명 박스 패턴
    if re.search(r"^[•\-\u00b7]\s", text):
        return True
    if re.search(r"^\*+\s", text):
        return True
    if re.search(r"^\(?수요|\(?공급|\(?전망", text):
        return True

    # 공백이 많고 문장형이면 본문으로 본다.
    space_ratio = text.count(" ") / max(1, len(text))
    if space_ratio > 0.08:
        return True

    return True


def is_label_like_text_block(
    text: str,
    rect: fitz.Rect,
    page_rect: fitz.Rect,
    *,
    max_font_size: float | None = None,
) -> bool:
    """축 라벨/범례/숫자처럼 짧은 텍스트 블록만 포함시키기 위한 휴리스틱."""
    text = normalize_text(text)
    if not text:
        return False

    if CAPTION_PATTERN.search(text) or SOURCE_PATTERN.search(text) or is_heading_like(text):
        return False

    # bullet/각주/출처 라인은 라벨이 아니다.
    if re.search(r"^\s*[*•\u25cf\u2022\-]\s*", text):
        return False
    if "참조" in text:
        return False

    # 문장형 텍스트는 제외(라벨은 보통 짧고 공백이 적다).
    if text.count(" ") >= 3:
        return False
    if re.search(r"[.,;:!?]", text) and len(text) >= 16:
        return False

    # 흔한 조사/어미 포함 문장 패턴은 라벨이 아니라 본문일 가능성이 높다.
    # (한국어에서 \\b가 잘 안 먹는 케이스가 있어서 단순 포함 검사도 같이 한다.)
    if re.search(r"(에서|으로|로|을|를|은|는|이|가|과|와|및)\b", text):
        return False
    if re.search(r"(은|는|이|가|을|를|과|와|에서|으로|로)([\\s.,)]|$)", text):
        return False
    if any(
        p in text
        for p in ["에서", "으로", "및", "하는", "되는", "있다", "한다", "하며", "그리고", "반면"]
    ):
        return False

    if len(text) > 45:
        return False

    if rect.height > page_rect.height * 0.06:
        return False

    if rect.width > page_rect.width * 0.45:
        return False

    # 큰 글자 블록(슬라이드형 페이지의 제목/설명)은 라벨이 아니다.
    # 축/범례는 보통 6~11pt 수준이며, 12pt 이상이면 문장/설명일 확률이 높다.
    if max_font_size is not None and max_font_size >= 12.5 and len(text) >= 10:
        return False

    return True


def tighten_crop_to_visual_and_labels(
    page,
    crop_rect: fitz.Rect,
    search_rect: fitz.Rect,
    page_rect: fitz.Rect,
) -> fitz.Rect | None:
    """그림 crop에서 본문 텍스트가 섞이는 문제를 줄이기 위해 crop을 다시 조인다."""
    # 1) crop 안의 실제 시각요소(이미지/벡터) union을 먼저 구한다.
    visual_parts: list[fitz.Rect] = []
    for rect in get_visual_candidate_rects(page):
        # 페이지 전체를 덮는 배경/템플릿 rect는 그림 crop에 방해가 되므로 제외한다.
        if rect_area(rect) / max(1.0, rect_area(page_rect)) > 0.55:
            continue

        inter = rect & crop_rect
        inter = inter & search_rect
        if rect_area(inter) <= 0:
            continue
        # 조각 노이즈 제거
        if inter.width < page_rect.width * 0.04 and inter.height < page_rect.height * 0.02:
            continue
        # 매우 얇은 가이드 라인/구분선은 label 영역을 오염시키므로 제외한다.
        if inter.height < page_rect.height * 0.003 and inter.width > page_rect.width * 0.35:
            continue
        if inter.width < page_rect.width * 0.01 and inter.height > page_rect.height * 0.2:
            continue
        visual_parts.append(inter)

    if not visual_parts:
        return None

    visual_parts = merge_rects(visual_parts, gap=45)
    visual_union = fitz.Rect(visual_parts[0])
    for rect in visual_parts[1:]:
        visual_union.include_rect(rect)

    # 2) visual_union 주변의 "작은 라벨 텍스트"만 제한적으로 포함한다.
    label_search = fitz.Rect(
        visual_union.x0 - 60,
        visual_union.y0 - 40,
        visual_union.x1 + 60,
        visual_union.y1 + 60,
    )
    label_search = label_search & search_rect

    result_rect = fitz.Rect(visual_union)
    for block in get_text_blocks(page):
        rect = block["rect"]
        text = block["text"]
        max_font_size = block.get("max_font_size")
        if not rect.intersects(label_search):
            continue
        if not is_label_like_text_block(text, rect, page_rect, max_font_size=max_font_size):
            continue
        result_rect.include_rect(rect)

    # 3) 그림 아래/주변 설명 문단, 출처, 각주가 같이 저장되지 않도록 하단을 잘라낸다.
    # visual_union 아래에서 시작하는 본문 블록이 보이면 그 위에서 컷.
    for block in get_text_blocks(page):
        rect = block["rect"]
        text = block["text"]
        if not rect.intersects(search_rect):
            continue
        if rect.y0 < visual_union.y1 - 4:
            continue
        if not (is_paragraph_like_text_block(text, rect, page_rect) or SOURCE_PATTERN.search(text)):
            continue
        # 본문/출처가 좌측에만 있고, 그림은 우측에 있는 슬라이드형 구성에서는
        # 출처 라인 때문에 x축/하단이 잘리는 문제가 생길 수 있다.
        # 따라서 그림(visual_union)과 가로로 유의미하게 겹칠 때만 하단 컷을 적용한다.
        if horizontal_overlap_ratio(rect, visual_union) < 0.28:
            continue
        result_rect.y1 = min(result_rect.y1, rect.y0 - 8)

    # 4) 최종 마진/클램프
    result_rect = expand_rect(result_rect, page_rect, margin=16)
    result_rect = result_rect & search_rect

    if not is_valid_crop_rect(result_rect, page_rect):
        return None

    if not has_enough_visual_content_for_figure(page, result_rect):
        return None

    return result_rect


def get_visual_candidate_rects(page) -> list[fitz.Rect]:
    """
    페이지 안의 실제 시각 요소 후보를 가져온다.

    그림용:
    - PDF image block
    - vector drawing rect
    """
    rects: list[fitz.Rect] = []
    page_rect = page.rect

    page_dict = page.get_text("dict")

    for block in page_dict.get("blocks", []):
        if block.get("type") == 1 and "bbox" in block:
            rects.append(fitz.Rect(block["bbox"]))

    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if not rect:
            continue

        r = fitz.Rect(rect)
        fill = drawing.get("fill")
        items = drawing.get("items", []) or []

        # 페이지 배경/레이아웃 템플릿으로 찍힌 큰 흰색 박스(또는 단일 도형)는
        # 시각자료 후보를 오염시키므로 제외한다.
        if fill is not None:
            try:
                fr, fg, fb = fill
                is_near_white = fr >= 0.98 and fg >= 0.98 and fb >= 0.98
            except Exception:
                is_near_white = False

            area_ratio = rect_area(r) / max(1.0, rect_area(page_rect))
            if is_near_white and area_ratio >= 0.12 and len(items) <= 2:
                continue

        rects.append(r)

    return rects


def merge_rects(rects: list[fitz.Rect], gap: float = 25) -> list[fitz.Rect]:
    merged: list[fitz.Rect] = []

    for rect in rects:
        added = False

        for idx, existing in enumerate(merged):
            expanded_existing = fitz.Rect(
                existing.x0 - gap,
                existing.y0 - gap,
                existing.x1 + gap,
                existing.y1 + gap,
            )

            if expanded_existing.intersects(rect):
                existing.include_rect(rect)
                merged[idx] = existing
                added = True
                break

        if not added:
            merged.append(fitz.Rect(rect))

    changed = True
    while changed:
        changed = False
        new_merged: list[fitz.Rect] = []

        for rect in merged:
            added = False

            for idx, existing in enumerate(new_merged):
                expanded_existing = fitz.Rect(
                    existing.x0 - gap,
                    existing.y0 - gap,
                    existing.x1 + gap,
                    existing.y1 + gap,
                )

                if expanded_existing.intersects(rect):
                    existing.include_rect(rect)
                    new_merged[idx] = existing
                    added = True
                    changed = True
                    break

            if not added:
                new_merged.append(rect)

        merged = new_merged

    return merged


def is_valid_crop_rect(rect: fitz.Rect, page_rect: fitz.Rect) -> bool:
    page_area = rect_area(page_rect)
    crop_area = rect_area(rect)

    if crop_area <= 0:
        return False

    if rect.height < page_rect.height * 0.04:
        return False

    if rect.width < page_rect.width * 0.18:
        return False

    if crop_area / page_area > 0.48:
        return False

    return True


def get_text_area_ratio_in_rect(page, crop_rect: fitz.Rect) -> float:
    crop_area = rect_area(crop_rect)

    if crop_area <= 0:
        return 1.0

    text_area = 0.0

    for block in get_text_blocks(page):
        rect = block["rect"]
        inter = rect & crop_rect
        text_area += rect_area(inter)

    return text_area / crop_area


def get_visual_area_ratio_in_rect(page, crop_rect: fitz.Rect) -> float:
    crop_area = rect_area(crop_rect)

    if crop_area <= 0:
        return 0.0

    visual_rects = get_visual_candidate_rects(page)
    visual_area = 0.0

    for rect in visual_rects:
        inter = rect & crop_rect
        visual_area += rect_area(inter)

    return visual_area / crop_area


def has_enough_visual_content_for_figure(page, crop_rect: fitz.Rect) -> bool:
    visual_ratio = get_visual_area_ratio_in_rect(page, crop_rect)
    text_ratio = get_text_area_ratio_in_rect(page, crop_rect)

    if visual_ratio < 0.006:
        return False

    if text_ratio > 0.42 and visual_ratio < 0.07:
        return False

    return True


def find_boundary_y_below(
    page,
    start_y: float,
    include_source: bool = True,
    include_paragraph_blocks: bool = False,
) -> float:
    page_rect = page.rect
    text_blocks = get_text_blocks(page)
    boundary_y = page_rect.y1 - 35

    for block in text_blocks:
        rect = block["rect"]
        text = block["text"]

        if rect.y0 <= start_y + 10:
            continue

        is_boundary = (
            CAPTION_PATTERN_STRICT.match(text)
            or is_heading_like(text)
            or (include_source and SOURCE_PATTERN.search(text))
            or (include_paragraph_blocks and is_paragraph_like_text_block(text, rect, page_rect))
        )

        if is_boundary:
            boundary_y = min(boundary_y, rect.y0 - 8)

    return boundary_y


def find_crop_rect_for_figure_below_caption(page, caption_rect: fitz.Rect) -> fitz.Rect | None:
    """
    그림용 crop.

    캡션 아래에서 실제 그래프/도형 후보를 찾아서 crop한다.
    가장 큰 조각 하나만 선택하지 않고, 같은 구역의 도형들을 union해서 잘림을 줄인다.
    """
    page_rect = page.rect
    y0 = caption_rect.y1 + 3
    boundary_y = find_boundary_y_below(
        page,
        y0,
        # 출처 라인이 페이지 좌측에만 존재하는 경우(그림은 우측),
        # 출처를 경계로 삼으면 x축/하단 이미지가 잘릴 수 있다.
        # 하단 컷은 tighten 단계에서 "겹칠 때만" 적용한다.
        include_source=False,
        include_paragraph_blocks=True,
    )

    search_rect = fitz.Rect(
        page_rect.x0 + 25,
        y0,
        page_rect.x1 - 25,
        boundary_y,
    )

    candidate_rects = []

    for rect in get_visual_candidate_rects(page):
        if not rect.intersects(search_rect):
            continue

        if rect.width < page_rect.width * 0.04:
            continue

        if rect.height < page_rect.height * 0.012:
            continue

        if rect_area(rect) / rect_area(page_rect) > 0.38:
            continue

        clipped = rect & search_rect
        if rect_area(clipped) <= 0:
            continue

        candidate_rects.append(clipped)

    if not candidate_rects:
        return None

    merged_candidates = merge_rects(candidate_rects, gap=55)

    # 가장 큰 영역만 쓰면 그림4/복합 도식이 잘릴 수 있어서,
    # 큰 후보들과 가까운 후보들을 함께 union한다.
    merged_candidates = sorted(
        merged_candidates,
        key=lambda r: rect_area(r),
        reverse=True,
    )

    largest_area = rect_area(merged_candidates[0])
    selected_rects = []

    for rect in merged_candidates:
        area = rect_area(rect)

        if area >= largest_area * 0.12:
            selected_rects.append(rect)
            continue

        # 작은 조각이어도 가장 큰 도형 근처에 있으면 포함
        near_largest = fitz.Rect(
            merged_candidates[0].x0 - 60,
            merged_candidates[0].y0 - 60,
            merged_candidates[0].x1 + 60,
            merged_candidates[0].y1 + 60,
        )

        if near_largest.intersects(rect):
            selected_rects.append(rect)

    if not selected_rects:
        return None

    crop_rect = fitz.Rect(selected_rects[0])
    for rect in selected_rects[1:]:
        crop_rect.include_rect(rect)

    # 주변 텍스트(축 라벨 등)는 tighten 단계에서 "짧은 라벨"만 선별적으로 포함한다.
    crop_rect = expand_rect(crop_rect, page_rect, margin=18)

    crop_rect.y0 = max(crop_rect.y0, y0)
    crop_rect.y1 = min(crop_rect.y1, boundary_y)

    tightened = tighten_crop_to_visual_and_labels(page, crop_rect, search_rect, page_rect)
    if tightened is None:
        return None
    crop_rect = tightened

    if not is_valid_crop_rect(crop_rect, page_rect):
        return None

    if not has_enough_visual_content_for_figure(page, crop_rect):
        return None

    return crop_rect


def find_crop_rect_for_table_below_caption(page, caption_rect: fitz.Rect) -> fitz.Rect | None:
    """
    표용 crop.

    표는 텍스트로 구성된 경우가 많으므로, 예전처럼 캡션 아래 영역을 그대로 잡는다.
    단, 다음 캡션/출처/참고/제목 전까지만 자른다.
    """
    page_rect = page.rect

    y0 = caption_rect.y1 + 5
    boundary_y = find_boundary_y_below(
        page,
        y0,
        include_source=True,
        include_paragraph_blocks=False,
    )

    # 표는 텍스트 기반일 수 있으므로 넓게 잡는다.
    crop_rect = fitz.Rect(
        page_rect.x0 + 35,
        y0,
        page_rect.x1 - 35,
        boundary_y,
    )

    if crop_rect.height < page_rect.height * 0.08:
        crop_rect.y1 = min(page_rect.y1 - 35, y0 + page_rect.height * 0.25)

    crop_rect = expand_rect(crop_rect, page_rect, margin=6)

    if not is_valid_crop_rect(crop_rect, page_rect):
        return None

    return crop_rect


def find_crop_rect_above_source(page, source_rect: fitz.Rect) -> fitz.Rect | None:
    """
    출처 위쪽 시각자료 crop.

    출처 위에 실제 이미지/vector 후보가 있을 때만 저장한다.
    텍스트 fallback은 하지 않는다.
    """
    page_rect = page.rect

    candidate_rects = get_visual_candidate_rects(page)
    candidate_rects = merge_rects(candidate_rects, gap=35)

    valid_candidates = []

    for rect in candidate_rects:
        if rect.y1 > source_rect.y0 + 5:
            continue

        if source_rect.y0 - rect.y1 > page_rect.height * 0.42:
            continue

        expanded = expand_rect(rect, page_rect, margin=14)

        if not is_valid_crop_rect(expanded, page_rect):
            continue

        if not has_enough_visual_content_for_figure(page, expanded):
            continue

        valid_candidates.append(expanded)

    if not valid_candidates:
        return None

    union_rect = fitz.Rect(valid_candidates[0])
    for rect in valid_candidates[1:]:
        union_rect.include_rect(rect)

    union_rect.y1 = min(union_rect.y1, source_rect.y0 - 5)
    union_rect = expand_rect(union_rect, page_rect, margin=8)

    if not is_valid_crop_rect(union_rect, page_rect):
        return None

    if not has_enough_visual_content_for_figure(page, union_rect):
        return None

    return union_rect


def save_crop_image(
    page,
    crop_rect: fitz.Rect,
    image_path: Path,
    zoom: float = 2.7,
) -> None:
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(
        matrix=matrix,
        clip=crop_rect,
        alpha=False,
    )
    pixmap.save(str(image_path))


def extract_visual_images_by_caption_and_source(
    pdf_path: Path,
    image_dir: Path,
    file_prefix: str,
    zoom: float = 2.7,
) -> list[dict]:
    """
    그림/표/출처 기준으로 시각자료를 저장한다.

    그림:
    - 실제 그래프/도형 후보 중심 crop

    표:
    - 텍스트 표가 많으므로 캡션 아래 영역 crop

    출처:
    - 출처 위 실제 시각 후보가 있을 때만 crop
    """
    image_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    results: list[dict] = []
    saved_rects_by_page: dict[int, list[fitz.Rect]] = {}

    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_rect = page.rect
        page_num = page_index + 1
        saved_rects_by_page[page_num] = []

        caption_blocks = find_caption_blocks(page)

        for caption_index, caption in enumerate(caption_blocks, start=1):
            visual_type = caption["visual_type"]

            if visual_type == "그림":
                crop_rect = find_crop_rect_for_figure_below_caption(page, caption["rect"])
                method = "figure_visual"
            else:
                crop_rect = find_crop_rect_for_table_below_caption(page, caption["rect"])
                method = "table_below"

            if crop_rect is None:
                continue

            if not is_valid_crop_rect(crop_rect, page_rect):
                continue

            if is_duplicate_rect(crop_rect, saved_rects_by_page[page_num]):
                continue

            try:
                caption_label_safe = safe_filename(caption["caption_label"])
                image_filename = (
                    f"{file_prefix}_page{page_num:03d}_{caption_label_safe}_{caption_index:02d}.png"
                )
                image_path = image_dir / image_filename

                save_crop_image(page, crop_rect, image_path, zoom=zoom)

                saved_rects_by_page[page_num].append(crop_rect)

                results.append(
                    {
                        "method": "figure" if method == "figure_visual" else "table",
                        "caption": caption["caption"],
                        "caption_label": caption["caption_label"],
                        "visual_type": visual_type,
                        "source_text": None,
                        "page": page_num,
                        "image_path": str(image_path),
                        "bbox": [crop_rect.x0, crop_rect.y0, crop_rect.x1, crop_rect.y1],
                    }
                )

            except Exception as e:
                log.warning(
                    "캡션 기반 시각자료 저장 실패 | pdf=%s page=%d caption=%s error=%s",
                    pdf_path,
                    page_num,
                    caption["caption"],
                    e,
                )

        source_blocks = find_source_blocks(page)

        for source_index, source in enumerate(source_blocks, start=1):
            crop_rect = find_crop_rect_above_source(page, source["rect"])

            if crop_rect is None:
                continue

            if not is_valid_crop_rect(crop_rect, page_rect):
                continue

            if is_duplicate_rect(crop_rect, saved_rects_by_page[page_num]):
                continue

            try:
                image_filename = (
                    f"{file_prefix}_page{page_num:03d}_source_above_{source_index:02d}.png"
                )
                image_path = image_dir / image_filename

                save_crop_image(page, crop_rect, image_path, zoom=zoom)

                saved_rects_by_page[page_num].append(crop_rect)

                results.append(
                    {
                        "method": "source_above",
                        "caption": None,
                        "caption_label": None,
                        "visual_type": "출처",
                        "source_text": source["source_text"],
                        "page": page_num,
                        "image_path": str(image_path),
                        "bbox": [crop_rect.x0, crop_rect.y0, crop_rect.x1, crop_rect.y1],
                    }
                )

            except Exception as e:
                log.warning(
                    "출처 기반 시각자료 저장 실패 | pdf=%s page=%d source=%s error=%s",
                    pdf_path,
                    page_num,
                    source["source_text"],
                    e,
                )

    doc.close()
    return results


class SpriCrawler(BaseCrawler):
    """SPRi 월간 산업동향 PDF 크롤러."""

    def __init__(self, month: str, output_path: Path):
        super().__init__(company=[])
        self.month = month
        self.output_path = output_path

    async def crawl(self) -> list[RawArticle]:
        pdf_dir = self.output_path.parent / "spri_pdfs" / self.month
        image_dir = self.output_path.parent / "spri_crawler_images" / self.month

        pdf_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        # 이전 실행 이미지 정리
        for old_image in image_dir.glob("*.png"):
            old_image.unlink()

        articles: list[RawArticle] = []

        async with httpx.AsyncClient(
            headers=REQUEST_HEADERS,
            timeout=30,
            follow_redirects=True,
        ) as client:
            issue_url = await find_issue_url_with_pagination(client, self.month)

            detail_html = await fetch_text(client, issue_url)

            issue_title = extract_issue_title(detail_html, self.month)
            published_at = parse_published_at(detail_html)
            pdf_items = extract_pdf_links(detail_html, issue_url)

            for index, pdf_item in enumerate(pdf_items, start=1):
                pdf_title = pdf_item["pdf_title"]
                pdf_url = pdf_item["pdf_url"]

                title = f"{issue_title} - {pdf_title}"

                article_id = make_id(
                    source_type="trend_report",
                    source_name="SPRi",
                    title=title,
                    url=pdf_url,
                )

                try:
                    pdf_bytes = await fetch_bytes(client, pdf_url)

                    safe_pdf_title = safe_filename(pdf_title)
                    pdf_filename = f"{index:02d}_{safe_pdf_title}.pdf"
                    pdf_path = pdf_dir / pdf_filename
                    pdf_path.write_bytes(pdf_bytes)

                    pdf_payload = extract_pdf_payload(pdf_bytes)
                    content = pdf_payload["text"]

                    visual_images = extract_visual_images_by_caption_and_source(
                        pdf_path=pdf_path,
                        image_dir=image_dir,
                        file_prefix=f"{index:02d}_{safe_pdf_title}",
                        zoom=2.7,
                    )

                    article = RawArticle(
                        id=article_id,
                        source_type="trend_report",
                        source_name="SPRi",
                        title=title,
                        content=content,
                        url=issue_url,
                        published_at=published_at,
                        publisher="소프트웨어정책연구소",
                        company=[],
                        language="ko",
                        content_type="pdf",
                        crawl_status="success",
                        error_message=None,
                        extra={
                            "month": self.month,
                            "issue_title": issue_title,
                            "pdf_title": pdf_title,
                            "pdf_url": pdf_url,
                            "pdf_path": str(pdf_path),
                            "page_count": pdf_payload["page_count"],
                            "pdf_pages": pdf_payload["page_count"],
                            "pdf_parsed_pages": pdf_payload["parsed_page_count"],
                            "pdf_page_blocks": pdf_payload["pages"],
                            "pdf_parse_strategy": pdf_payload["pdf_parse_strategy"],
                            "contains_images": pdf_payload["contains_images"],
                            "image_count": pdf_payload["image_count"],
                            "contains_tables": pdf_payload["contains_tables"],
                            "table_count": pdf_payload["table_count"],
                            "tables": pdf_payload["tables"],
                            "table_parse_strategy": pdf_payload["table_parse_strategy"],
                            "chart_parse_strategy": pdf_payload["chart_parse_strategy"],
                            "visual_images": visual_images,
                            "visual_image_count": len(visual_images),
                            "article_index": index,
                        },
                    )

                except Exception as e:
                    log.exception("SPRi PDF 처리 실패 | title=%s url=%s", pdf_title, pdf_url)

                    article = RawArticle(
                        id=article_id,
                        source_type="trend_report",
                        source_name="SPRi",
                        title=title,
                        content=None,
                        url=issue_url,
                        published_at=published_at,
                        publisher="소프트웨어정책연구소",
                        company=[],
                        language="ko",
                        content_type="pdf",
                        crawl_status="failed",
                        error_message=str(e),
                        extra={
                            "month": self.month,
                            "issue_title": issue_title,
                            "pdf_title": pdf_title,
                            "pdf_url": pdf_url,
                            "pdf_path": None,
                            "page_count": None,
                            "pdf_pages": None,
                            "pdf_parsed_pages": 0,
                            "pdf_page_blocks": [],
                            "pdf_parse_strategy": "failed",
                            "contains_images": False,
                            "image_count": 0,
                            "contains_tables": False,
                            "table_count": 0,
                            "tables": [],
                            "table_parse_strategy": "not_parsed",
                            "chart_parse_strategy": "not_parsed",
                            "visual_images": [],
                            "visual_image_count": 0,
                            "article_index": index,
                        },
                    )

                articles.append(article)

        return articles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPRi 산업동향 PDF 크롤러")

    parser.add_argument(
        "--month",
        required=True,
        help="수집할 월호. 예: 2026-04",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="결과 JSON 저장 경로. 미지정 시 src/crawler/crawler_results/spri_crawler.json",
    )

    return parser.parse_args()


def default_output_path() -> Path:
    return Path(__file__).resolve().parent / "crawler_results" / f"{Path(__file__).stem}.json"


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args = parse_args()

    output_path = Path(args.output) if args.output else default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crawler = SpriCrawler(
        month=args.month,
        output_path=output_path,
    )

    articles = await crawler.crawl()

    payload = [article.to_common_dict() for article in articles]

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    success_count = sum(1 for article in articles if article.crawl_status == "success")
    failed_count = sum(1 for article in articles if article.crawl_status == "failed")
    total_visuals = sum(int(article.extra.get("visual_image_count", 0)) for article in articles)

    print(f"SPRi 수집 월호: {args.month}")
    print(f"전체 결과: {len(articles)}건")
    print(f"성공: {success_count}건")
    print(f"실패: {failed_count}건")
    print(f"시각자료 이미지 수: {total_visuals}개")
    print(f"JSON 저장 위치: {output_path}")
    print(f"PDF 저장 폴더: {output_path.parent / 'spri_pdfs' / args.month}")
    print(f"이미지 저장 폴더: {output_path.parent / 'spri_crawler_images' / args.month}")

    for article in articles[:3]:
        print("-" * 80)
        print(f"title: {article.title}")
        print(f"source_type: {article.source_type}")
        print(f"source_name: {article.source_name}")
        print(f"content_type: {article.content_type}")
        print(f"published_at: {article.published_at.isoformat() if article.published_at else None}")
        print(f"content length: {len(article.content or '')}")
        print(f"visual_image_count: {article.extra.get('visual_image_count')}")
        print(f"url: {article.url}")
        print(f"pdf_url: {article.extra.get('pdf_url')}")


if __name__ == "__main__":
    asyncio.run(main())
