"""summarize strategic_evidence — extracted from facade (move-only)."""

from __future__ import annotations

import re
from typing import Any

from src.analysis.summarize.article_selection import (  # noqa: F401
    _analysis_article_limit,
    _analysis_article_score,
    _article_dedupe_text,
    _article_evidence_score,
    _article_prompt_snippets,
    _build_fetch_ids,
    _company_alias_title_tokens,
    _diverse_articles_from_same_event_group,
    _format_articles,
    _full_text_article_ids,
    _is_near_duplicate_article,
    _is_near_duplicate_snippet,
    _normalize_title_event_token,
    _same_event_title_groups,
    _same_title_event,
    _select_analysis_articles,
    _title_event_tokens,
    _title_group_features,
    _useful_title_event_token,
)
from src.analysis.summarize.config import (  # noqa: F401
    _ALLOWED_EVIDENCE_TYPES,
    _ARTICLE_CONTENT_CHARS,
    _ARTICLE_FACT_EXTRACTION_PROMPT,
    _ARTICLE_UI_BOILERPLATE_MARKERS,
    _COMPACT_ARTICLE_CONTENT_CHARS,
    _EVENT_TYPE_VALUES,
    _EVENT_TYPES,
    _FACT_EXTRACTION_BATCH_SIZE,
    _FACT_EXTRACTION_MAX_TOKENS,
    _FACT_EXTRACTION_MODE,
    _FACT_ID_SUMMARY_PROMPT,
    _FACT_TYPES,
    _FULL_TEXT_ARTICLE_LIMIT,
    _INDUSTRY_TREND_ALIASES,
    _INDUSTRY_TREND_COMPANY_ID,
    _KNOWN_COMPANY_ALIASES,
    _LLM_MODEL,
    _MAJORITY_THRESHOLD,
    _MAX_ANALYZED_ARTICLES,
    _MIN_ANALYZED_ARTICLES,
    _MIXED_THRESHOLD,
    _NEAR_DUPLICATE_SIMILARITY,
    _NUMBER_TOKEN_PATTERN,
    _PEER_ALIASES,
    _PROMPT_VERSION,
    _SNIPPET_CANDIDATE_SENTENCES,
    _SNIPPET_DEDUP_SIMILARITY,
    _SNIPPETS_PER_ARTICLE,
    _SUMMARY_LINE_MAX,
    _SUMMARY_LINE_MIN,
    _SUMMARY_MAX_TOKENS,
    _SUMMARY_ROLES,
    _SUPPORTING_ARTICLE_CONTENT_CHARS,
    _USE_FACT_EXTRACTION_LLM,
    _VALIDATION_MAX_TOKENS,
    _env_bool,
    _env_float,
    _env_int,
    _get_llm,
    _llm,
)
from src.analysis.summarize.fact_assembly import (  # noqa: F401
    _add_article_fallback_facts,
    _build_cluster_fact_intelligence,
    _build_extracted_facts,
    _classify_cluster_event_type,
    _classify_event_type_from_text,
    _is_duplicate_extracted_fact,
    _soften_uncertain_sentence,
    _title_to_fact_sentence,
)
from src.analysis.summarize.fact_extraction import (  # noqa: F401
    _coerce_summary_role,
    _combined_evidence_type,
    _default_summary_role,
    _evidence_type_from_fact_type,
    _extract_article_fact_notes,
    _extract_article_fact_notes_batch,
    _important_single_core_facts,
    _invoke_fact_extraction_llm,
    _is_financial_only_fact,
    _is_length_limit_error,
    _is_market_data_fact,
    _LengthLimitError,
    _merge_article_facts,
    _normalize_article_fact_note,
    _normalize_fact_object,
    _normalize_fact_type,
    _normalize_fact_type_value,
    _normalize_summary_role,
    _normalize_uncertain_fact,
    _normalize_unique_fact,
    _response_finish_reason,
    _response_hit_length_limit,
    _response_token_usage,
    _summary_role_priority,
)
from src.analysis.summarize.rule_based_facts import (  # noqa: F401
    _contract_candidate_sentences,
    _contract_detail_facts_from_article,
    _contract_entities,
    _contract_fact_sentence,
    _dedupe_contract_facts,
    _first_sentence_matching,
    _is_article_context_detail_snippet,
    _is_article_relevant_snippet,
    _rule_based_article_fact_notes,
    _rule_based_entities,
    _rule_based_event_type,
    _rule_based_fact_notes_need_llm,
    _rule_based_fact_type_and_role,
    _scope_fact_sentence,
    _select_rule_based_sentences,
    _snippet_score,
)
from src.analysis.summarize.summary_lines import (  # noqa: F401
    _additional_facts_for_prompt,
    _clean_summary_line,
    _compact_fact_for_prompt,
    _compact_facts_for_prompt,
    _company_aliases_for_detection,
    _company_like_mentions,
    _company_name_start_count,
    _compose_fallback_line,
    _desired_summary_line_count,
    _ensure_fact_summary_lines,
    _entity_context_sentence,
    _extract_reported_clause,
    _fact_basis_from_summary_line_fact_ids,
    _fact_id_empty_result,
    _fact_selection_score,
    _fact_similarity_text,
    _fact_text_for_summary_line,
    _facts_for_line,
    _fallback_fact_id_summary,
    _is_summary_extension_worthy,
    _line_summary_role_preferences,
    _nominalize_korean_predicate,
    _normalize_fact_id_summary_result,
    _normalize_summary_line_items,
    _primary_entity_for_summary_line,
    _product_context_subject,
    _reported_context_sentence,
    _rewrite_repeated_company_sentence,
    _select_fact_ids_for_summary_lines,
    _selected_facts_by_line,
    _similar_selected_fact_penalty,
    _starting_company_alias,
    _strip_leading_entity_subject,
    _summarize_from_fact_ids,
    _summary_line_company_attribution_warning,
    _summary_line_role_counts,
    _summary_subject_type,
    _sync_summary_line_item_texts,
    _text_mentions_any_alias,
    _topic_subject,
    _validate_fact_id_summary,
    normalize_subject_predicate_consistency,
)
from src.analysis.summarize.text_utils import (  # noqa: F401
    _append_reason,
    _article_company_alias_mentioned,
    _article_ids,
    _article_numeric_id,
    _article_similarity_tokens,
    _article_target_company_alias_mentioned,
    _article_topic_tokens,
    _articles_text,
    _as_int_list,
    _as_list,
    _body_peer_companies,
    _candidate_peer_companies,
    _chunked,
    _clamp_float,
    _clean_domain_term,
    _clean_json_response,
    _compact,
    _company_display_name,
    _company_list,
    _coverage_info,
    _date_tokens,
    _dedupe_ints,
    _dedupe_keep_order,
    _dedupe_similar_texts,
    _detect_conflict_notes,
    _empty_summary,
    _ensure_sentence,
    _escape_json_string_newlines,
    _event_verbs_in_text,
    _extract_json_object_text,
    _fact_is_off_topic_for_article,
    _fact_is_peer_related,
    _fact_key,
    _has_bad_korean_join,
    _has_business_scope_terms,
    _has_detail_preservation_terms,
    _has_effect_or_outcome_terms,
    _has_peer_owned_asset_terms,
    _has_risk_or_signal_terms,
    _has_technology_mechanism_terms,
    _has_uncertain_fact_marker,
    _has_unique_fact_importance,
    _is_article_ui_boilerplate,
    _is_company_neutral_context_detail,
    _is_customer_site_example_without_mechanism,
    _is_industry_trend_cluster,
    _is_peer_comparison_issue,
    _join_warnings,
    _looks_like_customer_site_case,
    _looks_like_industry_background_or_third_party,
    _matched_companies,
    _metadata,
    _normalize_content,
    _normalize_event_type,
    _normalize_number_token,
    _normalize_string_list,
    _number_token_covered,
    _number_tokens,
    _parse_json,
    _render_prompt,
    _repair_json_text,
    _safe_int,
    _safe_json_loads,
    _split_evidence_sentences,
    _strip_article_ui_boilerplate,
    _summary_mentions_company,
    _summary_metadata,
    _target_company_aliases,
    _text_similarity,
    normalize_korean_spacing,
)


def _enrich_peer_comparison_issue(
    summary: dict[str, Any],
    *,
    articles: list[dict[str, Any]],
    target_companies: list[str],
) -> dict[str, Any]:
    if not _is_peer_comparison_issue(articles):
        return summary
    out = dict(summary)
    mentioned = _dedupe_keep_order([*target_companies, *_body_peer_companies(articles)])
    comparison_facts = _peer_metric_comparison_facts(articles, mentioned)
    risk_facts = _risk_facts_from_articles(articles)
    market_structure_facts = _market_structure_facts_from_articles(articles)
    evidence_inventory = _strategic_evidence_inventory_from_articles(
        articles,
        comparison_facts=comparison_facts,
    )
    if (
        not comparison_facts
        and not risk_facts
        and not market_structure_facts
        and not any(evidence_inventory.values())
    ):
        return out

    out["mentioned_peer_companies"] = mentioned or out.get("mentioned_peer_companies")
    out["target_peer_companies"] = mentioned or out.get("target_peer_companies")
    out["issue_frame"] = {
        "frame_type": "peer_comparison",
        "comparison_axis": _comparison_axis_from_facts(comparison_facts),
        "main_company": out.get("main_company"),
        "mentioned_peer_companies": mentioned,
    }
    out["comparison_facts"] = comparison_facts
    out["risk_facts"] = risk_facts
    out["market_structure_facts"] = market_structure_facts
    out["strategic_evidence_inventory"] = evidence_inventory
    out["supporting_facts"] = evidence_inventory.get("supporting_facts", [])
    out["background_facts"] = evidence_inventory.get("background_facts", [])
    out["cause_or_driver_facts"] = evidence_inventory.get("cause_or_driver_facts", [])
    out["uncertainty_or_limitation_facts"] = evidence_inventory.get(
        "uncertainty_or_limitation_facts",
        [],
    )
    out["strategic_tensions"] = evidence_inventory.get("strategic_tensions", [])
    out["actionable_questions"] = evidence_inventory.get("actionable_questions", [])

    existing_lines = _normalize_string_list(out.get("fact_summary"))
    added_lines = _comparison_summary_lines(
        comparison_facts=comparison_facts,
        risk_facts=risk_facts,
        market_structure_facts=market_structure_facts,
    )
    out["fact_summary"] = _dedupe_keep_order([*existing_lines, *added_lines])[:_SUMMARY_LINE_MAX]
    if comparison_facts:
        out["headline"] = _peer_comparison_headline(comparison_facts) or out.get("headline")
        out["one_line_summary"] = _peer_comparison_one_liner(comparison_facts) or out.get(
            "one_line_summary"
        )
    out["integrated_text"] = " ".join(
        [
            str(out.get("one_line_summary") or ""),
            " ".join(out["fact_summary"]),
            " ".join(risk_facts),
            " ".join(market_structure_facts),
            " ".join(evidence_inventory.get("background_facts", [])),
            " ".join(evidence_inventory.get("cause_or_driver_facts", [])),
            " ".join(evidence_inventory.get("uncertainty_or_limitation_facts", [])),
            " ".join(evidence_inventory.get("strategic_tensions", [])),
        ]
    ).strip()
    out["reason"] = _append_reason(
        out.get("reason"),
        "peer_comparison_enriched: 본문에 등장한 비교 피어와 구조적 리스크를 보존함",
    )
    return out


def _peer_metric_comparison_facts(
    articles: list[dict[str, Any]],
    company_ids: list[str],
) -> list[dict[str, Any]]:
    sentences = _split_evidence_sentences(_articles_text(articles), limit=80)
    facts: list[dict[str, Any]] = []
    for company_id in company_ids:
        aliases = _PEER_ALIASES.get(company_id, [company_id])
        candidates: list[dict[str, Any]] = []
        for sentence_index, sentence in enumerate(sentences):
            matched_alias = next(
                (
                    str(alias)
                    for alias in aliases
                    if alias and re.search(re.escape(str(alias)), sentence, re.IGNORECASE)
                ),
                "",
            )
            if not matched_alias:
                continue
            if not _nearby_percentage(sentence, matched_alias) and sentence_index + 1 < len(
                sentences
            ):
                sentence = f"{sentence} {sentences[sentence_index + 1]}"
            if not re.search(r"내부거래|특수관계자|전체\s*매출|매출", sentence):
                continue
            values = _number_tokens(sentence)
            percentages = [value for value in values if "%" in value or "％" in value]
            nearby_percentage = _nearby_percentage(sentence, matched_alias)
            if not values or (percentages and not nearby_percentage):
                continue
            candidates.append(
                {
                    "company_id": company_id,
                    "metric": (
                        "internal_transaction_ratio"
                        if re.search(r"내부거래|특수관계자", sentence)
                        else "financial_metric"
                    ),
                    "value": nearby_percentage or (percentages[0] if percentages else values[0]),
                    "numbers": values[:6],
                    "evidence_text": sentence,
                    "_score": _peer_metric_sentence_score(sentence, matched_alias),
                }
            )
        if candidates:
            selected = sorted(candidates, key=lambda item: item.get("_score", 0), reverse=True)[0]
            selected.pop("_score", None)
            facts.append(selected)
    return facts


def _nearby_percentage(sentence: str, alias: str) -> str:
    for match in re.finditer(re.escape(alias), sentence, re.IGNORECASE):
        windows = [
            sentence[match.start() : min(len(sentence), match.end() + 55)],
            sentence[max(0, match.start() - 35) : min(len(sentence), match.end() + 45)],
        ]
        for window in windows:
            percentage_match = re.search(r"\d+(?:\.\d+)?\s*[%％]", window)
            if percentage_match:
                return percentage_match.group(0).replace("％", "%").strip()
    return ""


def _peer_metric_sentence_score(sentence: str, alias: str) -> int:
    score = 0
    matches = list(re.finditer(re.escape(alias), sentence, re.IGNORECASE))
    if not matches:
        return score
    after_alias = max((sentence[match.end() :] for match in matches), key=len)
    if re.search(r"내부거래|특수관계자|전체\s*매출|매출", after_alias[:80]):
        score += 4
    if re.search(r"\d+(?:\.\d+)?\s*[%％]", after_alias[:80]):
        score += 4
    if re.search(r"\d+\.\d+\s*[%％]", after_alias[:80]):
        score += 2
    if re.search(r"↑|↓|증가|감소|상승|하락", sentence):
        score += 1
    if re.search(rf"^\s*{re.escape(alias)}\s*(?:은|는|이|가|의|역시)", sentence, re.IGNORECASE):
        score += 6
    if re.search(r"^\s*[^.]{0,8}" + re.escape(alias), sentence, re.IGNORECASE):
        score += 1
    if (
        not re.search(
            rf"^\s*{re.escape(alias)}\s*(?:은|는|이|가|의|역시)",
            sentence,
            re.IGNORECASE,
        )
        and len(_mentioned_peer_ids_in_text(sentence)) >= 2
    ):
        score -= 6
    return score


def _mentioned_peer_ids_in_text(text: str) -> list[str]:
    mentioned: list[str] = []
    for company_id, aliases in _PEER_ALIASES.items():
        if any(
            alias and re.search(re.escape(str(alias)), text, re.IGNORECASE) for alias in aliases
        ):
            mentioned.append(company_id)
    return mentioned


def _risk_facts_from_articles(articles: list[dict[str, Any]]) -> list[str]:
    patterns = (r"공정위|공정거래위원회", r"사법\s*리스크|과징금|규제|감시")
    return _sentences_matching_any(articles, patterns, limit=3)


def _market_structure_facts_from_articles(articles: list[dict[str, Any]]) -> list[str]:
    patterns = (
        r"외부\s*경쟁력|외부\s*고객|대외\s*사업|홀로서기",
        r"클라우드&AI|AI\s*성과|재무제표|분리|검증",
        r"그룹\s*일감|계열사\s*일감|내부거래",
    )
    return _sentences_matching_any(articles, patterns, limit=4)


def _strategic_evidence_inventory_from_articles(
    articles: list[dict[str, Any]],
    *,
    comparison_facts: list[dict[str, Any]],
) -> dict[str, list[str]]:
    background_facts = _sentences_matching_any(
        articles,
        (
            r"업종|시장|업계|구조|태생적|특성상|최근\s*\d+\s*년|평균|상위권",
            r"공정위|공정거래위원회|규제|감시|공시대상기업집단",
        ),
        limit=5,
    )
    cause_or_driver_facts = _sentences_matching_any(
        articles,
        (
            r"주효|영향|결과|풀이|때문|연관|연결|배경|전략|확장|강화|전념|주도|전담",
            r"선정|공공사업|정부|그룹\s*차원|신사업|플랫폼|자동화|인프라",
        ),
        limit=6,
    )
    risk_facts = _sentences_matching_any(
        articles,
        (
            r"리스크|위험|한계|문제|족쇄|악화|제한|비판|규제|과징금|소송|감시",
            r"자유로울\s*수\s*없|검증하기\s*어렵|분리하지\s*않",
        ),
        limit=6,
    )
    uncertainty_or_limitation_facts = _sentences_matching_any(
        articles,
        (
            r"검증하기\s*어렵|확인하기\s*어렵|분리하지\s*않|투명하게\s*분리|한계",
            r"목소리가\s*높|따로\s*떼어내|공시|재무제표",
        ),
        limit=4,
    )
    market_structure_facts = _sentences_matching_any(
        articles,
        (
            r"외부\s*고객|외부\s*시장|대외\s*사업|대외\s*성과|독자적인\s*가치",
            r"계열사\s*일감|그룹\s*일감|모그룹|내부거래|거래\s*구조|고착",
        ),
        limit=6,
    )
    strategic_tensions = _strategic_tension_facts(
        background_facts=background_facts,
        market_structure_facts=market_structure_facts,
        risk_facts=risk_facts,
        uncertainty_facts=uncertainty_or_limitation_facts,
    )
    supporting_facts = _dedupe_keep_order(
        [
            *market_structure_facts,
            *cause_or_driver_facts,
            *risk_facts,
            *uncertainty_or_limitation_facts,
        ]
    )[:8]
    return {
        "core_facts": _comparison_core_fact_lines(comparison_facts),
        "supporting_facts": supporting_facts,
        "background_facts": background_facts,
        "cause_or_driver_facts": cause_or_driver_facts,
        "risk_facts": risk_facts,
        "uncertainty_or_limitation_facts": uncertainty_or_limitation_facts,
        "market_structure_facts": market_structure_facts,
        "strategic_tensions": strategic_tensions,
        "actionable_questions": _actionable_questions_from_inventory(
            comparison_facts=comparison_facts,
            background_facts=background_facts,
            cause_or_driver_facts=cause_or_driver_facts,
            risk_facts=risk_facts,
            uncertainty_facts=uncertainty_or_limitation_facts,
        ),
    }


def _comparison_core_fact_lines(comparison_facts: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for fact in comparison_facts[:8]:
        company = _company_display_name(fact.get("company_id"))
        value = str(fact.get("value") or "").strip()
        evidence = str(fact.get("evidence_text") or "").strip()
        if company and value and evidence:
            lines.append(f"{company} {value}: {evidence}")
        elif evidence:
            lines.append(evidence)
    return _dedupe_keep_order(lines)


def _strategic_tension_facts(
    *,
    background_facts: list[str],
    market_structure_facts: list[str],
    risk_facts: list[str],
    uncertainty_facts: list[str],
) -> list[str]:
    tensions: list[str] = []
    if background_facts and market_structure_facts:
        background = background_facts[0]
        market = next(
            (
                fact
                for fact in market_structure_facts
                if _inventory_fact_key(fact) != _inventory_fact_key(background)
            ),
            "",
        )
        tensions.append(f"{background} / {market}" if market else background)
    if risk_facts:
        tensions.append(risk_facts[0])
    if uncertainty_facts:
        tensions.append(uncertainty_facts[0])
    return _dedupe_keep_order(tensions)[:4]


def _inventory_fact_key(value: Any) -> str:
    return re.sub(r"\W+", "", normalize_korean_spacing(value)).lower()


def _actionable_questions_from_inventory(
    *,
    comparison_facts: list[dict[str, Any]],
    background_facts: list[str],
    cause_or_driver_facts: list[str],
    risk_facts: list[str],
    uncertainty_facts: list[str],
) -> list[str]:
    questions: list[str] = []
    if comparison_facts:
        axis = _comparison_axis_from_facts(comparison_facts)
        questions.append(
            f"SK AX는 {axis} 비교축을 내부 관리 지표로 어떻게 분리해 설명할 수 있는가?"
        )
    if cause_or_driver_facts:
        questions.append(
            "피어 간 차이가 사업 수행 방식, 고객 기반, 그룹 과제 중 어디에서 발생했는가?"
        )
    if risk_facts:
        questions.append(
            "해당 차이가 규제, 고객 확보, 재무 안정성, 운영 책임 중 어떤 리스크로 이어지는가?"
        )
    if background_facts or uncertainty_facts:
        questions.append(
            "외부에 설명 가능한 성과와 아직 검증하기 어려운 영역을 어떻게 구분할 것인가?"
        )
    return _dedupe_keep_order(questions)[:4]


def _sentences_matching_any(
    articles: list[dict[str, Any]],
    patterns: tuple[str, ...],
    *,
    limit: int,
) -> list[str]:
    out: list[str] = []
    for sentence in _split_evidence_sentences(_articles_text(articles), limit=100):
        sentence = _clean_inventory_sentence(sentence)
        if not sentence:
            continue
        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in patterns):
            out.append(sentence)
        if len(out) >= limit:
            break
    return _dedupe_keep_order(out)


def _clean_inventory_sentence(sentence: str) -> str:
    value = normalize_korean_spacing(sentence)
    if not value:
        return ""
    if re.search(r"…|\.{3}|VS", value) and re.search(r"\d{1,2}일\s+업계에\s+따르면", value):
        value = re.sub(r"^.*?(?=\d{1,2}일\s+업계에\s+따르면)", "", value).strip()
    if re.search(r"…|\.{3}|↑|↓|VS", value) and len(value) < 120:
        return ""
    return value


def _comparison_axis_from_facts(facts: list[dict[str, Any]]) -> str:
    if any(fact.get("metric") == "internal_transaction_ratio" for fact in facts):
        return "internal_transaction_ratio"
    return "peer_metric_comparison"


def _comparison_summary_lines(
    *,
    comparison_facts: list[dict[str, Any]],
    risk_facts: list[str],
    market_structure_facts: list[str],
) -> list[str]:
    lines: list[str] = []
    if len(comparison_facts) >= 2:
        values = [
            f"{_company_display_name(fact.get('company_id'))} {fact.get('value')}"
            for fact in comparison_facts
            if fact.get("company_id") and fact.get("value")
        ]
        if values:
            lines.append(
                "SI 업계 내부거래 의존도 비교에서 " + ", ".join(values[:4]) + "가 제시됐다."
            )
    if risk_facts:
        lines.append(risk_facts[0])
    if market_structure_facts:
        lines.append(market_structure_facts[-1])
    return lines


def _peer_comparison_headline(comparison_facts: list[dict[str, Any]]) -> str:
    if any(fact.get("metric") == "internal_transaction_ratio" for fact in comparison_facts):
        return "SI업계, 내부거래 의존도 격차 확대"
    return ""


def _peer_comparison_one_liner(comparison_facts: list[dict[str, Any]]) -> str:
    if len(comparison_facts) < 2:
        return ""
    values = [
        f"{_company_display_name(fact.get('company_id'))} {fact.get('value')}"
        for fact in comparison_facts
        if fact.get("company_id") and fact.get("value")
    ]
    if not values:
        return ""
    return "SI 업계에서 " + ", ".join(values[:4]) + " 등 내부거래 의존도 격차가 제시됐다."
