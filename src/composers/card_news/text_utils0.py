"""card_news text_utils0 — extracted from facade (move-only)."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.composers.card_news.schema import (  # noqa: F401
    _CARD_DETAIL_MAX,
    _CARD_PROMPT_VERSION,
    _DEFAULT_COVER_IMAGE_ALT,
    _DEFAULT_COVER_IMAGE_URL,
    _DISPLAY_ITEM_MAX,
    _DISPLAY_ITEM_MIN,
    _DISPLAY_ITEM_PREFERRED,
    _DISPLAY_TRUNCATED_MARKER_PATTERN,
    _DISPLAY_ZONE,
    _EVENT_TO_FACT_BASIS_TYPE,
    _FACT_BASIS_EVIDENCE_TYPES,
    _FRONTEND_EVENT_TYPES,
    _FRONTEND_SECTOR_IDS,
    _ISSUE_CARD_PROMPT,
    _PROMPT_VERSION,
    _SUMMARY_ACTION_TOKENS,
    _SUMMARY_LINE_MAX,
    _SUMMARY_LINE_MIN,
    _attach_card_news_schema_fields,
)
from src.config.companies import COMPANY_ALIASES, company_name_ko
from src.config.global_companies import GLOBAL_COMPANY_ALIASES, global_company_name_ko
from src.llm import LLMSpec, build_chat_llm

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        model_name = os.getenv("CARD_NEWS_COMPOSER_MODEL", "gpt-4o")
        # env 로 모델 지정 가능 → gpt-5 라도 reasoning_effort 미전달(기존 동작) 위해 None.
        _llm = build_chat_llm(
            LLMSpec(model=model_name, temperature=0.3, max_tokens=1024, reasoning_effort=None)
        )
    return _llm


def _build_cluster_fetch_ids(
    representative_id: int, cluster_article_ids: list[int] | None
) -> list[int]:
    """대표 기사를 앞에 두고 클러스터 전체 ID 목록을 만든다."""
    if not cluster_article_ids:
        return [representative_id]
    others = [aid for aid in cluster_article_ids if aid != representative_id]
    return [representative_id, *others]


def _order_articles(
    articles: list[dict[str, Any]],
    ordered_ids: list[int],
) -> list[dict[str, Any]]:
    """DB 조회 결과를 대표기사 우선 순서로 되돌린다."""
    order = {article_id: index for index, article_id in enumerate(ordered_ids)}
    return sorted(articles, key=lambda article: order.get(int(article.get("id") or 0), len(order)))


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _list_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in _list_value(value) if isinstance(item, dict)]


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return "피어사 주요 뉴스"


def _looks_like_sentence_title(title: str) -> bool:
    value = re.sub(r"\s+", " ", str(title or "")).strip()
    if len(value) > 60 and value.endswith(("다", "다.", "했다", "했다.", "됐다", "됐다.")):
        return True
    return bool(re.search(r"(했다|공개했다|체결했다|진출했다|선보였다|밝혔다)[.]?$", value))


def _compact_card_title(
    title: str,
    *,
    summary: dict[str, Any],
    classification: dict[str, Any],
    articles: list[dict[str, Any]],
) -> str:
    value = _clean_source_headline(title)
    if not _looks_like_sentence_title(value) and len(value) <= 52:
        return value
    candidates = [
        classification.get("title"),
        summary.get("display_headline"),
        summary.get("headline"),
    ]
    candidates.extend(article.get("title") for article in articles if isinstance(article, dict))
    for candidate in candidates:
        compact = _clean_source_headline(candidate)
        if compact and compact != value and len(compact) <= 52:
            return compact
    return value[:52].rstrip(" ,·…")


def _clean_source_headline(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^\[[^\]]{1,12}\]\s*", "", text)
    return text.strip(" -")


def _is_financial_only_title(title: str) -> bool:
    text = str(title or "")
    if not re.search(r"매출|영업이익|순이익|실적|목표가|목표주가|주가", text):
        return False
    return not re.search(
        r"클라우드|AI|에이전트|데이터센터|IT서비스|IT 서비스|SI|ITO|"
        r"물류|플랫폼|솔루션|ERP|MSP|CSP|AX|SDV|모빌리티|보안|센터|"
        r"사업|서비스|수주|계약|전환|구축",
        text,
        re.I,
    )


def _format_articles(articles: list[dict[str, Any]]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        credibility_score = a.get("credibility_score")
        credibility_text = f"{credibility_score:.2f}" if credibility_score is not None else "미계산"
        lines.append(
            f"[{i}] 제목: {a['title']}\n"
            f"    출처: {a['source_name']} (신뢰도: {credibility_text})"
            f" | URL: {a['url']}\n"
            f"    내용: {' '.join((a.get('content') or '').split())}"
        )
    return "\n\n".join(lines)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _card_news_id(cluster_id: int | None, published_date: str) -> str:
    date_key = published_date[:10].replace("-", "")
    suffix = f"{cluster_id:04d}" if cluster_id is not None else "0000"
    return f"CN-{date_key}-{suffix}"


def _normalize_summary_similarity_token(token: str) -> str:
    value = str(token or "").strip()
    if not value:
        return ""
    if value.isascii():
        return value.casefold()
    value = re.sub(
        r"(으로서|으로써|로서|로써|으로|에게서|에게|에서|부터|까지|보다|처럼|"
        r"은|는|이|가|을|를|의|에|와|과|도|만|로)$",
        "",
        value,
    )
    suffix_replacements = (
        (r"선정(?:됐다|되었다|했다|하였다)$", "선정"),
        (r"체결(?:됐다|되었다|했다|하였다)$", "체결"),
        (r"확정(?:됐다|되었다|했다|하였다)$", "확정"),
        (r"구축(?:됐다|되었다|했다|하였다)$", "구축"),
        (r"운영(?:됐다|되었다|했다|하였다)$", "운영"),
        (r"상승(?:했다|하였다)$", "상승"),
        (r"하락(?:했다|하였다)$", "하락"),
        (r"(?:했다|하였다|됐다|되었다|한다|됩니다|합니다|입니다|이다)$", ""),
    )
    for pattern, replacement in suffix_replacements:
        updated = re.sub(pattern, replacement, value)
        if updated != value:
            value = updated
            break
    return value if len(value) >= 2 else ""


def _has_numeric_or_period_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\d|년|월|일|까지|부터|규모|금액|기간|비율|대비|투자|자본금|"
            r"매출|수량|장|대|명|억원|조원|만원|%",
            str(text or ""),
        )
    )


def _has_target_capacity_or_schedule(text: str) -> bool:
    value = str(text or "")
    has_quantity = bool(re.search(r"\d[\d,.\s]*(장|대|개|건|명|곳|식|세트|억원|조원)", value))
    has_schedule = _has_schedule_signal(value)
    has_execution = bool(re.search(r"구축|운영|도입|전환|착공|확보|조성|목표|예정|계획", value))
    return has_execution and (has_quantity or has_schedule)


def _has_schedule_signal(text: str) -> bool:
    return bool(re.search(r"\d{4}\s*년|까지|부터|기간|단계|분기|월|일", str(text or "")))


def _has_non_money_quantity(text: str) -> bool:
    return bool(re.search(r"\d[\d,.\s]*(장|대|개|건|명|곳|식|세트)", str(text or "")))


def _is_market_reaction_summary_line(text: str) -> bool:
    return bool(
        re.search(
            r"주가|한국거래소|전\s*거래일|거래\s*(중|마쳤)|"
            r"장\s*(초반|마감)|상승|하락|급등|급락|투자자|시장\s*반응",
            str(text or ""),
        )
    )


def _format_numeric_text(value: str) -> str:
    try:
        number = float(str(value).replace(",", ""))
    except ValueError:
        return str(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.1f}".rstrip("0").rstrip(".")


def _strip_number_prefix(value: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", str(value or "")).strip()


def _db_record(
    *,
    card_id: str,
    company: str | None,
    cluster_id: int | None,
    title: str,
    summary_lines: list[str],
    event_type: str,
    importance: str,
    importance_score: float | None,
    sector: str,
    sectors: list[str],
    signals: dict[str, Any],
    evidence_chain: dict[str, Any],
    sources: list[dict[str, Any]],
    validation_pass: bool,
    validation_sc_score: float,
) -> dict[str, Any]:
    return {
        "id": card_id,
        "company": company,
        "cluster_id": cluster_id,
        "title": title[:500],
        "summary_lines": summary_lines,
        "event_type": event_type,
        "importance": importance,
        "importance_score": importance_score or 0.0,
        "implication": {
            "sector": sector,
            "sectors": sectors,
            "exposure_score": importance_score or 0.0,
            "exposure_band": importance,
            "signals": signals,
            "evidence_chain": evidence_chain,
        },
        "sources": sources,
        "validation_pass": validation_pass,
        "validation_sc_score": validation_sc_score,
    }


def _labeled_frontend_ready_items(blocks: list[dict[str, str]], *, heading: str) -> list[str]:
    items: list[str] = []
    for block in blocks:
        main = str(block.get("main") or "").strip()
        detail = str(block.get("detail") or "").strip()
        if not main or not detail:
            continue
        items.append(f"{heading}: {main}\n근거/설명: {detail}")
    return items


def _is_labeled_public_frontend_item(value: Any) -> bool:
    return bool(re.match(r"^핵심\s*(시사점|대응)\s*[:：]", str(value or "").strip()))


def _split_public_conclusion_evidence(value: Any) -> tuple[str, str]:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    sentences = _public_sentences(text)
    if len(sentences) >= 2:
        conclusion_count = (
            2 if _first_sentence_is_too_thin(sentences[0]) and len(sentences) >= 3 else 1
        )
        return (
            " ".join(sentences[:conclusion_count]).strip(),
            " ".join(sentences[conclusion_count:]).strip(),
        )

    # Some generated copy is one long sentence. Split only on reasoning
    # connectors so the original logic is preserved.
    for pattern in (
        r"\s+(기사에서는|근거는|이\s*근거|따라서|다만|그래야|이\s*기준|이\s*정보|후속으로는)\s+",
        r"\s+(때문에|확인되므로|확인되어야)\s+",
    ):
        match = re.search(pattern, text)
        if match and match.start() >= 35:
            return text[: match.start()].strip(), text[match.start() :].strip()
    return "", ""


def _public_sentences(value: Any) -> list[str]:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return []
    normalized = re.sub(r"(다\.|[.!?。])\s+", r"\1\n", text)
    return [part.strip() for part in normalized.splitlines() if part.strip()]


def _first_sentence_is_too_thin(sentence: str) -> bool:
    text = str(sentence or "").strip()
    if len(text) < 45:
        return True
    return bool(re.search(r"^(이번|해당|현재)\s", text)) and not re.search(
        r"때문|근거|확인|보여|의미|따라서|다만|왜|기준",
        text,
    )


def _public_copy_cleanup(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = _strip_public_section_prefixes(text)
    return text.strip()


def _insight_candidates_from_strategy(
    *,
    analysis: dict[str, Any],
    implication: dict[str, Any],
    strategic_root: dict[str, Any],
) -> list[dict[str, Any]]:
    peer = implication.get("peer_implication") or {}
    profile_linkage = strategic_root.get("profile_linkage") or {}
    candidates: list[dict[str, Any]] = [
        {
            "text": analysis.get("analysis_summary"),
            "path": "analysis.analysis_summary",
            "priority": 60,
            "semantic_role": "peer_business_meaning",
        },
        {
            "text": peer.get("peer_meaning") if isinstance(peer, dict) else None,
            "path": "peer_implication.peer_meaning",
            "priority": 58,
            "semantic_role": "peer_business_meaning",
        },
        {
            "text": peer.get("capability_change") if isinstance(peer, dict) else None,
            "path": "peer_implication.capability_change",
            "priority": 56,
            "semantic_role": "competition_standard_change",
        },
        {
            "text": analysis.get("market_signal"),
            "path": "analysis.market_signal",
            "priority": 54,
            "semantic_role": "competition_standard_change",
        },
    ]
    for index, item in enumerate(_list_string(analysis.get("strategic_meaning"))):
        candidates.append(
            {
                "text": item,
                "path": f"analysis.strategic_meaning[{index}]",
                "priority": 57 - index,
                "semantic_role": (
                    "peer_business_meaning" if index == 0 else "competition_standard_change"
                ),
            }
        )
    if isinstance(profile_linkage, dict):
        candidates.append(
            {
                "text": profile_linkage.get("reason"),
                "path": "profile_linkage.reason",
                "priority": 45,
                "allow_watch_point": True,
                "semantic_role": "uncertainty_or_follow_up",
            }
        )
    return candidates


def _action_semantic_role_for_index(index: int) -> str:
    if index == 0:
        return "skax_capability_gap_check"
    if index == 1:
        return "skax_operating_response"
    return "monitoring_response"


def _action_semantic_role_for_linkage_field(field: str) -> str:
    if field == "internal_checkpoints":
        return "skax_capability_gap_check"
    if field == "recommended_focus":
        return "skax_operating_response"
    if field == "monitoring_points":
        return "monitoring_response"
    return ""


def _looks_like_non_summary_line(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return True
    if text.endswith(("?", "？")):
        return True
    if re.search(r"(이유|왜|전망은|가능성은|과제는|향방은|뭘까|무엇인가)\??$", text):
        return True
    if re.search(
        r"^(단독|종합|속보|인터뷰|기획|칼럼|사설|재계|업계|현장)\s*[,·:]",
        text,
    ):
        return True
    if re.search(
        r"뉴스\s*듣기|글자\s*크기|기사\s*공유|주소복사|댓글|다크모드|"
        r"폰트|구독|프린트|카카오톡|페이스북",
        text,
    ):
        return True
    return False


def _article_content_sentences(content: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(content or "")).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?。！？다])\s+", normalized)
    sentences: list[str] = []
    for part in parts:
        text = part.strip(" -·,")
        if not (24 <= len(text) <= 180):
            continue
        if _looks_like_article_boilerplate(text):
            continue
        sentences.append(text)
        if len(sentences) >= _SUMMARY_LINE_MAX:
            break
    return sentences


def _looks_like_article_boilerplate(text: str) -> bool:
    return bool(
        re.search(
            r"뉴스\s*듣기|글자\s*크기|기사\s*공유|주소복사|다크모드|"
            r"무단전재|재배포\s*금지|저작권|기자\s*=|기자\s*$|"
            r"페이스북|카카오톡|이메일|구독|프린트",
            text,
            re.I,
        )
    )


def _first_list_item(value: Any) -> str:
    items = _list_string(value)
    return items[0] if items else ""


def _subject_company_names(summary: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for company_id in _list_string(summary.get("mentioned_peer_companies")):
        names.extend(
            [
                company_id,
                company_name_ko(company_id),
                global_company_name_ko(company_id),
            ]
        )
    main_company = str(summary.get("main_company") or "").strip()
    if main_company:
        names.extend(
            [
                main_company,
                company_name_ko(main_company),
                global_company_name_ko(main_company),
            ]
        )
    return [
        name
        for name in _unique_nonempty(names)
        if name and name not in {"피어사", "unknown", "None"}
    ]


def _first_amount_like_term(values: list[str]) -> str:
    for value in values:
        text = str(value or "").strip()
        if re.search(r"\d", text) and re.search(r"조|억|만|천|원", text):
            return text
    return ""


def _issue_execution_detail_phrase(summary: dict[str, Any]) -> str:
    intelligence = summary.get("cluster_fact_intelligence") or {}
    facts = " ".join(_list_string(summary.get("fact_summary")))
    numbers: list[str] = []
    if isinstance(intelligence, dict):
        numbers.extend(_list_string(intelligence.get("numbers_and_dates")))
    numbers.extend(
        re.findall(
            r"(?<![A-Za-z0-9])\d[\d,]*(?:조|억|만|천)?\s*(?:원|장|대|개|년|월|일)(?![A-Za-z0-9])",
            facts,
        )
    )
    cleaned_numbers: list[str] = []
    for value in numbers:
        text = re.sub(r"\s+", "", str(value or "")).strip(" ,.")
        if text and text not in cleaned_numbers:
            cleaned_numbers.append(text)
        if len(cleaned_numbers) >= 3:
            break
    details: list[str] = []
    if cleaned_numbers:
        details.append(", ".join(cleaned_numbers))
    if re.search(r"서비스\s*개시|오픈|운영\s*시작|구축\s*완료", facts):
        details.append("서비스 개시 또는 구축 완료 일정")
    if re.search(r"협약|계약|선정|컨소시엄|SPC|주주간", facts, flags=re.IGNORECASE):
        details.append("추진 구조")
    return ", ".join(details[:3])


def _skax_business_phrase_from_implication(skax: dict[str, Any]) -> str:
    mapped = _list_string(skax.get("business_line_mapping"))
    if mapped:
        return "·".join(mapped[:2])
    text = " ".join(
        [
            str(skax.get("why_important") or ""),
            str(skax.get("potential_impact") or ""),
            " ".join(_list_string(skax.get("recommended_actions"))),
        ]
    )
    patterns = (
        r"SK\s*AX의\s*([가-힣A-Za-z0-9&·/+_\-\s]{2,50}?)(?:\s*접점|과|와)",
        r"SK\s*AX도\s*([가-힣A-Za-z0-9&·/+_\-\s]{2,50}?)(?:\s*사업|역량)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        phrase = re.sub(r"\s+", " ", match.group(1)).strip(" ,.")
        if phrase:
            return phrase
    return ""


def _profile_linkage_level(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("profile_linkage") or {}
    if isinstance(linkage, dict):
        level = str(linkage.get("linkage_level") or "").strip().casefold()
        if level in {"high", "medium", "low", "none"}:
            return level
    return ""


def _linkage_area_records(linkage: Any, *, field: str) -> list[dict[str, Any]]:
    data = linkage if isinstance(linkage, dict) else {}
    records: list[dict[str, Any]] = []
    for item in _list_value(data.get(field)):
        if isinstance(item, dict):
            records.append(item)
    return records


def _linkage_record_terms(record: dict[str, Any], *fields: str) -> list[str]:
    terms: list[str] = []
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and "," in value:
            terms.extend(part.strip() for part in value.split(",") if part.strip())
        else:
            terms.extend(_list_string(value))
    return _unique_nonempty(terms)


def _display_token_pieces(text: str) -> list[str]:
    pieces: list[str] = []
    for piece in re.split(r"[·/&+_\-\s]+", str(text or "")):
        normalized = _normalize_display_token(piece)
        if normalized and len(normalized) >= 2:
            pieces.append(normalized)
    return pieces


def _low_specificity_display_match_term(term: str) -> bool:
    return term in {
        "서비스",
        "사업",
        "지원",
        "기업",
        "관련",
        "기반",
        "summary",
        "역량",
        "도입",
        "제공",
        "확보",
    }


def _normalize_display_token(text: str) -> str:
    token = re.sub(r"\s+", "", str(text or "")).strip(" ,.")
    if len(token) > 3:
        token = re.sub(r"(으로|에서|에게|과|와|은|는|이|가|을|를|의)$", "", token)
    return token.casefold() if token.isascii() else token


def _unique_nonempty(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip(" ,.")
        key = text.casefold()
        if text and key not in seen:
            out.append(text)
            seen.add(key)
    return out


def _join_context_terms(values: list[str]) -> str:
    terms = _unique_nonempty(values)
    return "·".join(terms[:4])


def _compact_repeated_subject_in_phrase(phrase: str, subject: str) -> str:
    value = re.sub(r"\s+", " ", str(phrase or "")).strip()
    topic = re.sub(r"\s+", " ", str(subject or "")).strip()
    if not value or not topic:
        return value
    if not value.startswith(topic):
        return value
    rest = value[len(topic) :].lstrip()
    if not rest:
        return "해당 사업"
    if rest.startswith("의"):
        return f"해당 사업{rest}"
    if rest[:1] in {"은", "는", "이", "가", "을", "를", "과", "와"}:
        return f"해당 사업{rest}"
    return f"해당 사업 {rest}"


def _has_final_consonant(char: str) -> bool:
    value = str(char or "")[:1]
    if not value:
        return False
    code = ord(value)
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return False


def _clean_display_truncated_fragment(text: Any) -> str:
    """Clean display-only truncation without changing analysis source facts."""

    out = re.sub(r"\s+", " ", str(text or "")).strip()
    if not out:
        return ""
    out = re.sub(r"\[[^\]]*(?:\.\.\.|…)[^\]]*$", "", out)
    out = re.sub(r"\([^)]*(?:\.\.\.|…)[^)]*$", "", out)
    out = _DISPLAY_TRUNCATED_MARKER_PATTERN.sub(" ", out)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"[\s\[\(「『\"'·,;:/\\|-]+$", "", out).strip()
    return out


def _normalize_card_news_statement_style(text: str) -> str:
    """Keep card-news frontend copy in plain declarative Korean style."""

    out = str(text or "").strip()
    replacements = (
        (r"보여줍니다", "보여준다"),
        (r"드러납니다", "드러난다"),
        (r"이어집니다", "이어진다"),
        (r"연결됩니다", "연결된다"),
        (r"중요해질\s*수\s*있습니다", "중요해질 수 있다"),
        (r"필요해질\s*수\s*있습니다", "필요해질 수 있다"),
        (r"볼\s*수\s*있습니다", "볼 수 있다"),
        (r"할\s*수\s*있습니다", "할 수 있다"),
        (r"될\s*수\s*있습니다", "될 수 있다"),
        (r"구분해야\s*합니다", "구분해야 한다"),
        (r"검토해야\s*합니다", "검토해야 한다"),
        (r"확인해야\s*합니다", "확인해야 한다"),
        (r"점검해야\s*합니다", "점검해야 한다"),
        (r"마련해야\s*합니다", "마련해야 한다"),
        (r"좁혀야\s*합니다", "좁혀야 한다"),
        (r"낮춰야\s*합니다", "낮춰야 한다"),
        (r"나눠야\s*합니다", "나눠야 한다"),
        (r"해야\s*합니다", "해야 한다"),
        (r"필요합니다", "필요하다"),
        (r"어렵습니다", "어렵다"),
        (r"가능합니다", "가능하다"),
        (r"제시됩니다", "제시된다"),
        (r"확인됩니다", "확인된다"),
        (r"포함됩니다", "포함된다"),
        (r"나타납니다", "나타난다"),
        (r"설명됩니다", "설명된다"),
        (r"평가됩니다", "평가된다"),
        (r"됩니다", "된다"),
        (r"있습니다", "있다"),
        (r"없습니다", "없다"),
        (r"합니다", "한다"),
    )
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out)
    return out


def _polish_frontend_ready_internal_terms(text: str) -> str:
    out = str(text or "").strip()
    replacements = (
        (
            r"내부거래와\s*외부거래,\s*자체\s*수행과\s*외부\s*협력의\s*구분\s*기준",
            "내부거래와 외부거래의 구분 기준",
        ),
        (
            r"내부거래와\s*외부거래,\s*자체\s*수행과\s*외부\s*협력",
            "내부거래와 외부거래",
        ),
        (
            r"나눠\s*보며\s*검토\s*범위와\s*더\s*확인할\s*조건을\s*분리해야\s*합니다",
            "기준으로 검토 범위를 구분해야 합니다",
        ),
        (r"자사\s*관여\s*가능\s*영역\s*과", "검토 범위와"),
        (r"자사\s*관여\s*가능\s*영역", "검토 범위"),
        (r"추가\s*검증이\s*필요한\s*조건", "더 확인할 조건"),
        (r"추가\s*검증\s*조건", "더 확인할 조건"),
        (r"입력\s*근거", "기사 근거"),
        (r"\banchor\b|앵커", "근거 표현"),
        (r"이\s*기준이\s*있어야", "이 기준을 정리해야"),
        (r"기사\s*안에서\s*확인됩니다", "기사에서 확인됩니다"),
        (r"점검\s*항목", "점검 기준"),
        (r"비교\s*항목", "비교 기준"),
        (r"검증\s*항목", "검증 기준"),
        (r"관리\s*항목", "관리 기준"),
        (
            r"나눠\s*보며\s*검토\s*범위와\s*더\s*확인할\s*조건을\s*분리해야\s*합니다",
            "기준으로 검토 범위를 구분해야 합니다",
        ),
    )
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def _strip_public_section_prefixes(text: str) -> str:
    out = str(text or "").strip()
    out = re.sub(
        r"^(?:핵심\s*(?:시사점|대응)|근거\s*[/／]?\s*설명)\s*[:：]\s*",
        "",
        out,
    ).strip()
    return out


def _company_surface_replacements() -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    for company_id, aliases in COMPANY_ALIASES.items():
        display = company_name_ko(company_id)
        for alias in aliases:
            pattern = _surface_alias_pattern(alias)
            if pattern:
                replacements.append((pattern, display))
    for company_id, aliases in GLOBAL_COMPANY_ALIASES.items():
        display = global_company_name_ko(company_id)
        for alias in aliases:
            pattern = _surface_alias_pattern(alias)
            if pattern:
                replacements.append((pattern, display))
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    return replacements


def _surface_alias_pattern(alias: Any) -> str:
    value = re.sub(r"\s+", " ", str(alias or "").strip())
    if not value:
        return ""
    return re.escape(value).replace(r"\ ", r"\s*")


def _profile_phrase_from_peer_copy(text: str) -> str:
    source = re.sub(r"\s+", " ", str(text or "")).strip()
    patterns = (
        r"프로필(?:상|에서는)?\s*([가-힣A-Za-z0-9&·/+_\-\s]{2,70}?)(?:가|이)\s",
        r"피어의\s*([가-힣A-Za-z0-9&·/+_\-\s]{2,70}?)(?:\s*접점|과|와)",
    )
    for pattern in patterns:
        match = re.search(pattern, source)
        if not match:
            continue
        phrase = re.sub(r"\s+", " ", match.group(1)).strip(" ,.")
        phrase = re.sub(r"^(에서는|상)\s*", "", phrase)
        if phrase and not re.search(r"^(이번|현재|따라서|확인된)$", phrase):
            return phrase
    return ""


def _is_internal_analysis_copy(text: str) -> bool:
    return bool(
        re.search(
            r"피어\s*프로필|프로필의|프로필상|맥락이|연결되는\s*신호|"
            r"관찰할\s*수\s*있|해석할\s*수\s*있|변화\s*신호|"
            r"현재\s*사건의\s*구축\s*범위와\s*추진\s*구조",
            str(text or ""),
        )
    )


def _linkage_text(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            if key in {"matched_profile_areas", "matched_skax_areas"}:
                for record in _list_value(item):
                    if isinstance(record, dict):
                        parts.extend(str(v) for v in record.values() if v)
            elif isinstance(item, (str, int, float)):
                parts.append(str(item))
            elif isinstance(item, list):
                parts.extend(str(v) for v in item if isinstance(v, (str, int, float)))
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(_linkage_text(item) for item in value)
    return str(value or "")


def _grounding_path_matches(entry_path: str, candidate_path: str) -> bool:
    if not entry_path or not candidate_path:
        return False
    return (
        entry_path == candidate_path
        or entry_path.startswith(f"{candidate_path}.")
        or candidate_path.startswith(f"{entry_path}.")
    )


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if _is_untranslated_english_text(text):
            continue
        if text:
            return text
    return ""


def _bounded_detail_items(*values: Any) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for value in values:
        for text in _list_string(value):
            line = re.sub(r"\s+", " ", text).strip()
            if _is_untranslated_english_text(line):
                continue
            key = _detail_line_key(line)
            if line and key and key not in seen and not _is_near_duplicate_detail(line, lines):
                lines.append(line)
                seen.add(key)
            if len(lines) >= _CARD_DETAIL_MAX:
                return lines
    return lines[:_CARD_DETAIL_MAX]


def _detail_line_key(text: str) -> str:
    key = re.sub(r"[\s.。!?！？,，]+", "", str(text or "")).strip()
    key = re.sub(r"(합니다|해야합니다|있습니다|됩니다|입니다)$", "", key)
    return key


def _is_untranslated_english_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    hangul_count = len(re.findall(r"[가-힣]", value))
    english_words = re.findall(r"[A-Za-z]{3,}", value)
    if hangul_count == 0 and len(english_words) >= 5:
        return True
    alpha_count = len(re.findall(r"[A-Za-z]", value))
    return len(value) >= 60 and alpha_count > max(12, hangul_count * 3)


def _is_near_duplicate_detail(text: str, existing_lines: list[str]) -> bool:
    tokens = set(re.findall(r"[가-힣A-Za-z0-9&·+_-]{2,}", text or ""))
    if not tokens:
        return False
    for line in existing_lines:
        other = set(re.findall(r"[가-힣A-Za-z0-9&·+_-]{2,}", line or ""))
        if not other:
            continue
        overlap = len(tokens & other) / max(1, min(len(tokens), len(other)))
        if overlap >= 0.6:
            return True
    return False


def _detail_line_candidates(value: Any) -> list[str]:
    candidates: list[str] = []
    for item in _list_string(value):
        split_items = [
            re.sub(r"\s+", " ", part).strip()
            for part in re.split(r"(?<=[.!?。！？])\s+|(?<=다)\.\s*", item)
            if part.strip()
        ]
        candidates.extend(split_items or [re.sub(r"\s+", " ", item).strip()])
    return candidates


def _follow_up_action_from_statement(text: str) -> str:
    subject = _statement_to_check_subject(text)
    if subject.endswith(("는지", "인지", "한지", "할지")):
        return f"후속 검토에서 {subject} 확인합니다"
    return f"후속 검토에서 {subject} 여부를 확인합니다"


def _statement_to_check_subject(text: str) -> str:
    subject = text.rstrip(".。!?！？ ").strip()
    replacements = (
        (r"할\s*수\s*있습니다$", "할 수 있는지"),
        (r"될\s*수\s*있습니다$", "될 수 있는지"),
        (r"가능성이\s*있습니다$", "가능성이 있는지"),
        (r"필요가\s*있습니다$", "필요한지"),
        (r"해야\s*합니다$", "해야 하는지"),
        (r"합니다$", "하는지"),
        (r"있습니다$", "있는지"),
        (r"입니다$", "인지"),
    )
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, subject)
        if updated != subject:
            return updated
    return subject


def _looks_like_action(text: str) -> bool:
    return bool(
        re.search(
            r"점검|검토|확인|비교|분석|모니터링|관리|추적|정리|구조화|구분|보완|반영",
            text,
        )
    )


def _published_datetime(articles: list[dict[str, Any]], fallback: str) -> str:
    values = [
        parsed
        for parsed in (
            _parse_datetime_for_display(article.get("published_at")) for article in articles
        )
        if parsed is not None
    ]
    if not values:
        fallback_dt = _parse_datetime_for_display(fallback)
        return fallback_dt.astimezone(_DISPLAY_ZONE).isoformat() if fallback_dt else fallback
    earliest = min(values, key=lambda value: value.astimezone(UTC))
    return earliest.astimezone(_DISPLAY_ZONE).isoformat()


def _parse_datetime_for_display(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return datetime.fromisoformat(text).replace(tzinfo=_DISPLAY_ZONE)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _normalize_peer_id(value: Any) -> str | None:
    peer_id = str(value or "").strip()
    return peer_id or None


def _normalize_event_type(value: Any) -> str:
    event_type = str(value or "").strip()
    if event_type == "tech_release":
        return "technology_update"
    if event_type == "financial":
        return "earnings"
    if event_type in {"expansion", "company", "new_biz"}:
        return "general_update"
    if event_type == "ma":
        return "investment"
    if event_type == "personnel":
        return "organization"
    if event_type == "tech":
        return "technology_update"
    return event_type if event_type in _FRONTEND_EVENT_TYPES else "general_update"


def _issue_terms(summary: dict[str, Any], card_text: str) -> list[str]:
    del card_text
    terms: list[str] = []
    intelligence = summary.get("cluster_fact_intelligence")
    if isinstance(intelligence, dict):
        terms.extend(_list_string(intelligence.get("products_or_services")))
    cleaned = [_clean_issue_term(term) for term in terms]
    return _dedupe_keep_order([term for term in cleaned if _is_signature_noun_phrase(term)])[:8]


def _clean_issue_term(term: str) -> str:
    text = re.sub(r"\s+", " ", str(term or "").strip())
    text = re.sub(r"^[,.'\"‘’“”\s]+|[,.'\"‘’“”\s]+$", "", text)
    return text.strip(" ,.'\"‘’“”")


def _is_signature_noun_phrase(term: str) -> bool:
    text = str(term or "").strip()
    if len(text) < 2:
        return False
    if len(text) > 40:
        return False
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9+\-/]{1,20}", text):
        return True
    if re.fullmatch(r"[가-힣A-Za-z0-9·+\-/]+(?:\s+[가-힣A-Za-z0-9·+\-/]+){1,3}", text):
        return True
    if re.fullmatch(r"[가-힣A-Za-z0-9·+\-/]{2,20}", text):
        return True
    return False


def _issue_signature(terms: list[str]) -> str:
    normalized = sorted({_compact_ascii(term) for term in terms if _compact_ascii(term)})
    return "|".join(normalized[:5])


def _compact_ascii(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(text or "").lower())


def _card_text(
    title: str,
    summary_lines: list[str],
    summary: dict[str, Any],
    articles: list[dict[str, Any]],
) -> str:
    return " ".join(
        [
            title,
            " ".join(summary_lines),
            str(summary.get("one_line_summary") or ""),
            str(summary.get("main_event") or ""),
            " ".join(
                f"{article.get('title') or ''} {article.get('content') or ''}"
                for article in articles
            ),
        ]
    )


def _card_text_from_card(card: dict[str, Any]) -> str:
    return " ".join(
        [
            str(card.get("title") or ""),
            " ".join(_list_string(card.get("summary_lines"))),
            _json_like_text(card.get("news_summary")),
        ]
    )


def _json_like_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)


def _card_signals(card: dict[str, Any]) -> dict[str, Any]:
    implication = card.get("implication")
    if isinstance(implication, dict) and isinstance(implication.get("signals"), dict):
        return dict(implication["signals"])
    if isinstance(card.get("signals"), dict):
        return dict(card["signals"])
    return {}


def _set_card_signals(card: dict[str, Any], signals: dict[str, Any]) -> None:
    if "signals" in card:
        card["signals"] = signals
    implication = card.get("implication")
    if isinstance(implication, dict):
        implication["signals"] = signals
    db_record = card.get("db_record")
    if isinstance(db_record, dict):
        db_implication = db_record.get("implication")
        if isinstance(db_implication, dict):
            db_implication["signals"] = signals


def _accent_color(sector: str) -> str:
    return {
        "ax": "coral",
        "security": "blue",
        "infra": "green",
        "biz_area": "orange",
        "other": "gray",
    }.get(sector, "gray")


def _list_string(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _metadata(article: dict[str, Any]) -> dict[str, Any]:
    metadata = article.get("metadata") or article.get("extra") or {}
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    if text:
        try:
            return datetime.fromisoformat(text).isoformat()
        except ValueError:
            pass
    return text or None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
