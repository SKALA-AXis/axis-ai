# 작성일: 2026-05-19
# 작성자: 박지원
# 변경이력:
#   2026-05-19 박지원 — 파서·에이전트 수정으로 시작, 카드 뉴스 품질·근거(provenance)
#   2026-06-05 심유정 — strategic insight grounding 및 카드 dry run 개선
#   2026-06-11 최종민 — ChatOpenAI lazy-import 적용, LLM gen-search 이행 및 summarizer 예외 명문화
#   2026-06-18 최종민 — 코드 변경
"""소스 사실 요약 컴포넌트.

클러스터에 묶인 기사들을 바탕으로 분석 가능한 사실 요약을 생성한다.
"""

# ruff: noqa: E501

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

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
from src.analysis.summarize.strategic_evidence import (  # noqa: F401
    _actionable_questions_from_inventory,
    _clean_inventory_sentence,
    _comparison_axis_from_facts,
    _comparison_core_fact_lines,
    _comparison_summary_lines,
    _enrich_peer_comparison_issue,
    _inventory_fact_key,
    _market_structure_facts_from_articles,
    _mentioned_peer_ids_in_text,
    _nearby_percentage,
    _peer_comparison_headline,
    _peer_comparison_one_liner,
    _peer_metric_comparison_facts,
    _peer_metric_sentence_score,
    _risk_facts_from_articles,
    _sentences_matching_any,
    _strategic_evidence_inventory_from_articles,
    _strategic_tension_facts,
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
    _fact_key,
    _has_bad_korean_join,
    _has_business_scope_terms,
    _has_detail_preservation_terms,
    _has_uncertain_fact_marker,
    _has_unique_fact_importance,
    _is_article_ui_boilerplate,
    _is_company_neutral_context_detail,
    _is_industry_trend_cluster,
    _is_peer_comparison_issue,
    _join_warnings,
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
from src.db.article_store import get_articles_by_ids

log = logging.getLogger(__name__)


class SourceSummarizer:
    """클러스터 단위로 source 사실 요약을 생성한다."""

    def summarize(
        self,
        cluster_id: int,
        representative_id: int,
        cluster_article_ids: list[int] | None = None,
        max_cluster_articles: int | None = None,
    ) -> dict[str, Any]:
        """DB의 raw_articles를 읽어 클러스터 사실 요약을 생성한다."""
        ids_to_fetch = _build_fetch_ids(
            representative_id=representative_id,
            cluster_article_ids=cluster_article_ids,
            max_cluster_articles=max_cluster_articles,
        )
        articles = get_articles_by_ids(ids_to_fetch)
        return self.summarize_articles(
            cluster_id=cluster_id,
            representative_id=representative_id,
            articles=articles,
            cluster_article_ids=cluster_article_ids,
        )

    def summarize_articles(
        self,
        cluster_id: int,
        representative_id: int,
        articles: list[dict[str, Any]],
        cluster_article_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """이미 로드된 기사 목록으로 클러스터 사실 요약을 생성한다."""
        if not articles:
            return _empty_summary(
                cluster_id=cluster_id,
                representative_id=representative_id,
                reason="요약할 기사가 없습니다.",
            )

        all_source_article_ids = _article_ids(articles)
        requested_cluster_ids = list(cluster_article_ids or [])
        coverage_warning = ""
        if not requested_cluster_ids:
            coverage_warning = "cluster_article_ids 없음"
        cluster_ids_for_summary = requested_cluster_ids or all_source_article_ids
        selection = _select_analysis_articles(
            articles=articles,
            representative_id=representative_id,
        )
        if selection["status"] == "mixed_cluster_no_majority":
            return _empty_summary(
                cluster_id=cluster_id,
                representative_id=representative_id,
                reason="mixed_cluster_no_majority",
                source_article_ids=[],
                cluster_article_ids=cluster_ids_for_summary,
                coverage=_coverage_info(
                    cluster_article_count=len(cluster_ids_for_summary),
                    analyzed_article_count=0,
                    warning=selection["warning"],
                    selection=selection,
                ),
            )

        articles_for_analysis = selection["articles"]
        source_article_ids = _article_ids(articles_for_analysis)
        cluster_article_count = len(cluster_ids_for_summary)
        analyzed_article_count = len(source_article_ids)
        coverage = _coverage_info(
            cluster_article_count=cluster_article_count,
            analyzed_article_count=analyzed_article_count,
            warning=_join_warnings(coverage_warning, selection["warning"]),
            selection=selection,
        )

        target_companies = _candidate_peer_companies(articles_for_analysis)
        if not target_companies:
            return _empty_summary(
                cluster_id=cluster_id,
                representative_id=representative_id,
                reason="self 회사를 제외한 피어사 후보를 찾지 못했습니다.",
                source_article_ids=source_article_ids,
                cluster_article_ids=cluster_ids_for_summary,
                coverage=coverage,
            )

        article_fact_notes, fact_extraction_warnings, fact_extraction_failed = (
            _extract_article_fact_notes_batch(
                cluster_id=cluster_id,
                articles=articles_for_analysis,
                target_companies=target_companies,
                representative_id=representative_id,
            )
        )
        merged_facts = _merge_article_facts(article_fact_notes)
        cluster_fact_intelligence = _build_cluster_fact_intelligence(merged_facts)
        cluster_event_type = _classify_cluster_event_type(
            cluster_fact_intelligence,
            articles_for_analysis,
        )
        extracted_facts = _build_extracted_facts(
            cluster_id=cluster_id,
            article_fact_notes=article_fact_notes,
            articles=articles_for_analysis,
            cluster_event_type=cluster_event_type,
            target_companies=target_companies,
        )
        selected_fact_ids = _select_fact_ids_for_summary_lines(
            extracted_facts=extracted_facts,
            cluster_event_type=cluster_event_type,
        )

        try:
            result = _summarize_from_fact_ids(
                cluster_event_type=cluster_event_type,
                extracted_facts=extracted_facts,
                selected_fact_ids=selected_fact_ids,
                target_companies=target_companies,
                main_company=target_companies[0],
            )
            result = _validate_fact_id_summary(
                result=result,
                extracted_facts=extracted_facts,
                source_article_ids=source_article_ids,
                main_company=result.get("main_company", ""),
            )
        except Exception as exc:
            log.error("피어사 뉴스 요약 실패 | cluster=%s error=%s", cluster_id, exc)
            return _empty_summary(
                cluster_id=cluster_id,
                representative_id=representative_id,
                reason=f"LLM 요약 실패: {type(exc).__name__}",
                source_article_ids=source_article_ids,
                cluster_article_ids=cluster_ids_for_summary,
                coverage=coverage,
            )

        summary = {
            "cluster_id": cluster_id,
            "representative_id": representative_id,
            "source_article_ids": source_article_ids,
            "cluster_article_ids": cluster_ids_for_summary,
            "analyzed_article_ids": source_article_ids,
            "summary_scope": "peer_company_fact_only",
            "excluded_company_tiers": ["self"],
            "target_peer_companies": target_companies,
            "cluster_fact_intelligence": cluster_fact_intelligence,
            "extracted_facts": extracted_facts,
            "selected_fact_ids": selected_fact_ids,
            "coverage": coverage,
            "model": _LLM_MODEL,
            **result,
        }
        summary = _enrich_peer_comparison_issue(
            summary,
            articles=articles_for_analysis,
            target_companies=target_companies,
        )
        if fact_extraction_warnings:
            summary["validation_warnings"] = _dedupe_keep_order(
                [
                    *_normalize_string_list(summary.get("validation_warnings")),
                    *fact_extraction_warnings,
                ]
            )
        if fact_extraction_failed:
            summary["fact_extraction_failed"] = True
        log.info(
            "피어사 뉴스 요약 완료 | cluster=%s valid=%s company=%s sources=%d",
            cluster_id,
            summary.get("is_valid_summary"),
            summary.get("main_company"),
            len(summary["source_article_ids"]),
        )
        return summary
