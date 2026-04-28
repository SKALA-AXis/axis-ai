"""기사 관련성, 기간, 유사도 필터."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from html import unescape

RECENT_WINDOW = timedelta(days=1)
SIMILARITY_THRESHOLD = 0.55
TOKEN_OVERLAP_THRESHOLD = 0.45
TITLE_DUPLICATE_THRESHOLD = 0.95
ISSUE_TOKEN_OVERLAP_THRESHOLD = 0.22

ISSUE_PHRASES = [
    "인공지능",
    "생성형 ai",
    "llm",
    "ai 에이전트",
    "에이전틱 ai",
    "agentic ai",
    "ai agent",
    "ai 전환",
    "디지털 전환",
    "클라우드",
    "cloud",
    "데이터센터",
    "data center",
    "gpu",
    "it서비스",
    "it 서비스",
    "it 운영",
    "시스템 통합",
    "자동화",
    "보안",
    "금융 인프라",
    "제조 ax",
    "인프라",
]

SHORT_ENGLISH_ISSUE_PHRASES = {
    "ai",
    "ax",
    "dx",
    "si",
}

STRONG_ISSUE_PHRASES = {
    "생성형 ai",
    "llm",
    "에이전틱 ai",
    "agentic ai",
    "ai 에이전트",
    "ai agent",
    "ai 전환",
    "데이터센터",
    "data center",
    "gpu",
    "인프라",
    "제조 ax",
}


def strip_html(text: str) -> str:
    return unescape(text or "").replace("<b>", "").replace("</b>", "")


def normalize(text: str) -> str:
    text = strip_html(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = normalize(text)
    normalized_phrase = normalize(phrase)

    if not normalized_text or not normalized_phrase:
        return False

    if normalized_phrase in normalized_text:
        return True

    compact_text = normalized_text.replace(" ", "")
    compact_phrase = normalized_phrase.replace(" ", "")

    if len(compact_phrase) >= 4 and compact_phrase in compact_text:
        return True

    return False


def contains_short_english_phrase(text: str, phrase: str) -> bool:
    normalized_text = normalize(text)
    normalized_phrase = normalize(phrase)

    if not normalized_text or not normalized_phrase:
        return False

    pattern = rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def count_exact_phrase(text: str, phrase: str) -> int:
    normalized_text = normalize(text)
    normalized_phrase = normalize(phrase)

    if not normalized_text or not normalized_phrase:
        return 0

    compact_text = normalized_text.replace(" ", "")
    compact_phrase = normalized_phrase.replace(" ", "")

    normal_count = normalized_text.count(normalized_phrase)
    compact_count = 0
    if len(compact_phrase) >= 4:
        compact_count = compact_text.count(compact_phrase)

    return max(normal_count, compact_count)


def matches_article(
    title: str,
    content: str,
    aliases: list[str],
    topics: list[str],
    excluded_aliases: list[str],
) -> bool:
    text = f"{title} {content}"

    about_peer = any(
        contains_phrase(title, alias) or count_exact_phrase(text, alias) >= 2
        for alias in aliases
    )
    has_topic = any(contains_phrase(text, topic) for topic in topics)
    mentions_other_peer = any(
        contains_phrase(title, alias) or count_exact_phrase(text, alias) >= 2
        for alias in excluded_aliases
    )

    return about_peer and has_topic and not mentions_other_peer


def is_recent(published_at: datetime | None, now: datetime | None = None) -> bool:
    if published_at is None:
        return False

    now = now or datetime.now().astimezone()
    if published_at.tzinfo is None:
        published_at = published_at.astimezone()

    return now - RECENT_WINDOW <= published_at <= now


def is_within_days(
    published_at: datetime | None,
    days: int,
    now: datetime | None = None,
) -> bool:
    if published_at is None:
        return False

    now = now or datetime.now().astimezone()
    if published_at.tzinfo is None:
        published_at = published_at.astimezone()

    return now - timedelta(days=days) <= published_at <= now


def similarity_key(title: str, content: str) -> str:
    title = re.split(r"\s+-\s+", title, maxsplit=1)[0]
    text = normalize(f"{title} {content}")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^0-9a-z가-힣 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_duplicate_title(title: str, previous_titles: list[str]) -> bool:
    title_key = similarity_key(title, "")
    return any(
        SequenceMatcher(None, title_key, previous_title).ratio() >= TITLE_DUPLICATE_THRESHOLD
        for previous_title in previous_titles
    )


def is_too_similar(text: str, previous_texts: list[str]) -> bool:
    return any(
        SequenceMatcher(None, text, previous_text).ratio() >= SIMILARITY_THRESHOLD
        or _token_overlap(text, previous_text) >= TOKEN_OVERLAP_THRESHOLD
        or _same_issue(text, previous_text)
        for previous_text in previous_texts
    )


def issue_signature(text: str) -> set[str]:
    return _issue_signature(text)


def _same_issue(left: str, right: str) -> bool:
    shared_phrases = _issue_signature(left) & _issue_signature(right)
    overlap = _token_overlap(left, right)

    if shared_phrases & STRONG_ISSUE_PHRASES:
        return overlap >= ISSUE_TOKEN_OVERLAP_THRESHOLD

    return len(shared_phrases) >= 2 and overlap >= 0.30


def _issue_signature(text: str) -> set[str]:
    signatures = {phrase for phrase in ISSUE_PHRASES if contains_phrase(text, phrase)}

    signatures.update(
        phrase
        for phrase in SHORT_ENGLISH_ISSUE_PHRASES
        if contains_short_english_phrase(text, phrase)
    )

    return signatures


def _token_overlap(left: str, right: str) -> float:
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)

    if not left_tokens or not right_tokens:
        return 0.0

    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _meaningful_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[0-9a-z가-힣]+", text.lower()))
    return {token for token in tokens if len(token) >= 2}
