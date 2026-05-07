"""뉴스/웹 기사 상세 본문, 부제, 이미지 URL 추출 유틸."""

from __future__ import annotations

import re
from html import unescape
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup

BODY_MIN_LEN = 50

BODY_SELECTORS = [
    "#dic_area",
    "article#dic_area",
    "div#articeBody",
    "div#articleBodyContents",
    "div.newsct_article",
    "div.article_body",
    "div#article_body",
    "div#articleBody",
    "div.article-body",
    "div.vcon_con_intxt",
    "div#ctl00_ContentPlaceHolder1_WebNewsView_ltContentDiv",
    "div.rns_text",
    "div.article_txt",
    "div.article-text",
    "div.article_content",
    "div.article-content",
    "div.view_content",
    "div.view-cont",
    "div#newsEndContents",
    "div.news_end",
    "section.article_view",
    "div.newsView",
    "div.view_con",
    "div.view_txt",
]

SUBTITLE_SELECTORS = [
    ".subheading",
    ".sub-heading",
    ".sub_title",
    ".sub-title",
    ".subtitle",
    ".article-subtitle",
    ".article_subtitle",
    ".view-sub-title",
    ".view_sub_title",
    ".summary",
    ".article-summary",
    ".article_summary",
    ".lead",
    "strong.media_end_summary",
]

BODY_TEXT_NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "header",
    "footer",
    "nav",
    "aside",
    ".related",
    ".relation",
    ".recommend",
    ".ranking",
    ".comment",
    ".reply",
    ".sns",
    ".share",
    ".copyright",
    ".copy",
    ".byline",
    ".reporter",
    ".article_info",
    ".news_info",
    ".keyword",
    ".keywords",
    ".tag",
    ".tags",
    ".hashtag",
    ".hash_tag",
    ".tag-list",
    ".tag_list",
    ".view_tag",
    ".article_keyword",
]

BODY_STOP_LINE_MARKERS = (
    "관련기사",
    "키워드",
    "기자의 전체기사",
    "저작권자",
)


def extract_body_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    body_node = find_body_node(soup)

    if body_node:
        text = extract_clean_body_text(body_node)
        if len(text) >= BODY_MIN_LEN:
            return text[:5000]

    probable_root = find_probable_article_node(soup)

    if not probable_root:
        return ""

    clean_text = extract_clean_body_text(probable_root)

    paragraphs = [
        normalize_title_text(p)
        for p in clean_text.splitlines()
        if len(normalize_title_text(p)) >= 30
    ]

    if paragraphs:
        return "\n".join(paragraphs)[:5000]

    return ""


def extract_clean_body_text(node) -> str:
    soup = BeautifulSoup(str(node), "html.parser")

    for noise_node in soup.select(",".join(BODY_TEXT_NOISE_SELECTORS)):
        noise_node.decompose()

    for link in soup.find_all("a", href=True):
        href = str(link.get("href", ""))
        text = link.get_text(" ", strip=True)

        if is_body_keyword_link(href, text):
            removable = link.find_parent("li") or link
            removable.decompose()

    lines: list[str] = []

    for raw_line in soup.get_text(separator="\n", strip=True).splitlines():
        line = normalize_title_text(raw_line)

        if is_body_stop_line(line):
            break

        if is_valid_body_line(line):
            lines.append(line)

    return "\n".join(lines)


def extract_subtitle(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    title_text = ""

    title_node = soup.select_one("h1, .heading, .article-title, .article_title")

    if title_node:
        title_text = normalize_title_text(title_node.get_text(" ", strip=True))

    subtitles: list[str] = []
    seen_norms: set[str] = set()

    for selector in SUBTITLE_SELECTORS:
        for node in soup.select(selector):
            text = normalize_title_text(node.get_text(" ", strip=True))

            if is_valid_subtitle(text, title_text):
                text_norm = normalize_no_space(text)

                if text_norm not in seen_norms:
                    seen_norms.add(text_norm)
                    subtitles.append(text)

            if len(subtitles) >= 5:
                break

        if len(subtitles) >= 5:
            break

    return " ".join(subtitles)[:500]


def extract_image_urls(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    body_node = find_body_node(soup)
    search_root = body_node or find_probable_article_node(soup) or soup

    urls: list[str] = []
    seen_keys: set[str] = set()

    for tag in search_root.find_all(["img", "source"]):
        if not is_article_image_candidate(tag, search_root):
            continue

        for raw_url in iter_image_url_candidates(tag):
            add_image_url(urls, raw_url, base_url, seen_keys)

    return urls[:10]


def find_body_node(soup: BeautifulSoup):
    for selector in BODY_SELECTORS:
        node = soup.select_one(selector)

        if node:
            return node

    return None


def find_probable_article_node(soup: BeautifulSoup):
    candidates = soup.find_all(["article", "section", "div", "main"])
    best_node = None
    best_score = 0.0

    for node in candidates:
        text_len = len(node.get_text(" ", strip=True))
        image_count = len(node.find_all("img"))

        if image_count == 0 or text_len < BODY_MIN_LEN:
            continue

        attrs = normalize_image_text(
            " ".join(
                str(value)
                for value in (
                    node.get("id", ""),
                    " ".join(node.get("class", [])),
                    node.get("role", ""),
                )
            )
        )

        text_per_image = text_len / max(image_count, 1)
        score = min(text_len, 3000) / 20 + min(text_per_image, 1000) / 2

        if re.search(r"\b(article|body|content|news|view|story|post|read)\b", attrs):
            score += 50

        if has_noise_marker(attrs):
            score -= 100

        if image_count > 8 and text_per_image < 250:
            score -= 150

        if score > best_score:
            best_node = node
            best_score = score

    return best_node


def is_body_keyword_link(href: str, text: str) -> bool:
    if "articleList.html" in href and ("sc_word=" in href or "sc_area=K" in href):
        return True

    return text.startswith("#") and len(text) <= 40


def is_valid_body_line(text: str) -> bool:
    if not text:
        return False

    if text.startswith("#") and len(text) <= 40:
        return False

    noise = (
        "관련기사",
        "다른기사 보기",
        "저작권자",
        "무단전재",
        "재배포",
        "AI학습 및 활용 금지",
        "기자",
        "입력",
        "수정",
        "승인",
        "댓글",
    )

    return not any(word in text for word in noise)


def is_body_stop_line(text: str) -> bool:
    return any(
        text == marker or text.startswith(f"{marker} ")
        for marker in BODY_STOP_LINE_MARKERS
    )


def normalize_title_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_no_space(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def is_valid_subtitle(text: str, title_text: str) -> bool:
    if not text or len(text) < 8 or len(text) > 300:
        return False

    if title_text and normalize_no_space(text) == normalize_no_space(title_text):
        return False

    noise = ("기자", "입력", "수정", "승인", "댓글", "SNS 기사보내기", "저작권자")

    return not any(word in text for word in noise)


def iter_image_url_candidates(tag) -> list[str]:
    urls: list[str] = []

    attrs = (
        "src",
        "data-src",
        "data-original",
        "data-lazy-src",
        "data-url",
        "data-image",
        "data-full-url",
        "content",
    )

    for attr in attrs:
        raw_url = tag.get(attr)

        if raw_url:
            urls.append(str(raw_url))

    for attr in ("srcset", "data-srcset"):
        srcset_url = pick_srcset_url(tag.get(attr))

        if srcset_url:
            urls.append(srcset_url)

    style = tag.get("style")

    if style:
        urls.extend(re.findall(r"url\(['\"]?([^'\")]+)", str(style)))

    return urls


def pick_srcset_url(srcset: str | None) -> str | None:
    if not srcset:
        return None

    candidates: list[tuple[int, str]] = []

    for part in str(srcset).split(","):
        chunks = part.strip().split()

        if not chunks:
            continue

        url = chunks[0]

        if url:
            width = 0

            if len(chunks) > 1:
                width_match = re.search(r"\d+", chunks[1])
                width = int(width_match.group()) if width_match else 0

            candidates.append((width, url))

    if not candidates:
        return None

    return max(candidates, key=lambda candidate: candidate[0])[1]


def is_article_image_candidate(tag, body_node) -> bool:
    if tag.name == "source" and tag.find_parent("picture") is None:
        return False

    if has_profile_image_url(tag):
        return False

    context = image_context_text(tag, body_node)

    if has_noise_marker(context):
        return False

    if is_inside_noise_container(tag, body_node):
        return False

    width = parse_dimension(tag.get("width") or tag.get("data-width"))
    height = parse_dimension(tag.get("height") or tag.get("data-height"))

    if (width is not None and width < 120) or (height is not None and height < 80):
        return False

    if width is not None and height is not None and (width * height < 20000):
        return False

    score = 0

    if tag.name == "source":
        score += 1

    if tag.find_parent(["figure", "picture"]):
        score += 3

    if re.search(
        r"\b(article|body|content|news|photo|image|img|fig|view|end photo|articlephoto)\b",
        context,
    ):
        score += 2

    if tag.get("alt"):
        score += 1

    if (width and width >= 300) or (height and height >= 180):
        score += 2

    if has_image_like_url(tag):
        score += 1

    return score >= 1


def is_inside_noise_container(tag, body_node) -> bool:
    current = tag.parent

    while current and current is not body_node:
        attrs = normalize_image_text(
            " ".join(
                str(value)
                for value in (
                    current.get("id", ""),
                    " ".join(current.get("class", [])),
                    current.get("role", ""),
                )
            )
        )

        if has_noise_marker(attrs):
            return True

        current = current.parent

    return False


def image_context_text(tag, body_node) -> str:
    values: list[str] = []
    current = tag

    while current and current is not body_node:
        for attr in ("alt", "class", "id", "role", "src", "data-src"):
            value = current.get(attr)

            if value:
                values.append(" ".join(value) if isinstance(value, list) else str(value))

        current = current.parent

    return normalize_image_text(" ".join(values))


def normalize_image_text(text: str) -> str:
    return re.sub(r"[\s_\-/]+", " ", text.lower())


def has_noise_marker(text: str) -> bool:
    noise_words = (
        "logo",
        "icon",
        "sns",
        "share",
        "button",
        "btn",
        "profile",
        "avatar",
        "reporter",
        "banner",
        "ad banner",
        "aside",
        "ad",
        "ads",
        "advert",
        "promotion",
        "promote",
        "subscription",
        "subscribe",
        "comment",
        "reply",
        "ranking",
        "related",
        "recommend",
        "thumbnail",
        "thumb",
        "thumb upload",
        "outlink",
        "footer",
        "footer banner",
        "header",
        "nav",
        "youtube",
        "journalimg",
        "magnifier",
        "썸네일",
        "배너",
        "구독",
    )

    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in noise_words)


def parse_dimension(value) -> int | None:
    if value is None:
        return None

    match = re.search(r"\d+", str(value))

    return int(match.group()) if match else None


def has_image_like_url(tag) -> bool:
    return any(is_image_url(raw_url) for raw_url in iter_image_url_candidates(tag))


def has_profile_image_url(tag) -> bool:
    return any(is_profile_image_url(raw_url) for raw_url in iter_image_url_candidates(tag))


def is_profile_image_url(url: str) -> bool:
    if not url:
        return False

    parsed = urlparse(unescape(url.lower()))
    path = normalize_image_text(parsed.path)

    profile_markers = (
        "member",
        "profile",
        "reporter",
        "journalist",
        "avatar",
    )

    return any(re.search(rf"\b{re.escape(marker)}\b", path) for marker in profile_markers)


def add_image_url(
    urls: list[str],
    raw_url: str | None,
    base_url: str,
    seen_keys: set[str],
) -> None:
    if not raw_url:
        return

    url = normalize_image_url(raw_url, base_url)

    if not url.startswith(("http://", "https://")):
        return

    if not is_image_url(url):
        return

    key = image_dedupe_key(url)

    if key not in seen_keys:
        seen_keys.add(key)
        urls.append(url)


def normalize_image_url(raw_url: str, base_url: str) -> str:
    raw_url = unescape(raw_url.strip())

    if raw_url.startswith("//"):
        raw_url = f"{urlparse(base_url).scheme}:{raw_url}"

    return urljoin(base_url, raw_url)


def is_image_url(url: str) -> bool:
    lowered = url.lower()

    if lowered.startswith("data:"):
        return False

    if lowered.endswith((".svg", ".gif", ".ico")):
        return False

    parsed = urlparse(lowered)

    if parsed.path.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        return True

    if "image" in parsed.path and not parsed.path.endswith((".svg", ".gif", ".ico")):
        return True

    query = parse_qs(parsed.query)
    image_params = ("src", "simg", "file", "filename", "image", "img", "url")

    return any(
        any(
            value.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
            for value in values
        )
        for key, values in query.items()
        if key in image_params
    )


def image_dedupe_key(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    for key in ("simg", "src", "file", "filename", "image", "img"):
        values = query.get(key)

        if values:
            return values[0]

    return f"{parsed.netloc}{parsed.path}"
