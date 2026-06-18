# 작성일: 2026-06-02
# 작성자: 박지원
# 변경이력:
#   2026-06-02 박지원 — 카드뉴스 생성/클러스터링·품질·근거(provenance) 보강 및 카드 검증 완화
#   2026-06-05 심유정 — 카드뉴스 전략 인사이트 근거 강화 및 frontend-ready 처리 개선
#   2026-06-11 최종민 — ChatOpenAI lazy-import 적용, card_news_composer LLM gen-search 이행 리팩터
"""카드뉴스 생성 에이전트.

AnalysisPackage를 사용자에게 보여줄 카드뉴스/API 응답 형태로 재가공한다.
기존 raw cluster 기반 생성 메서드는 호환용으로 유지한다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from src.analysis.models import AnalysisPackage
from src.composers.card_news.classification import (  # noqa: F401
    _filter_false_positive_sectors,
    _importance_band,
    _infer_event_type,
    _infer_sectors,
    _normalize_exposure_band,
    _normalize_sector,
    _normalized_importance_score,
)
from src.composers.card_news.display_assembly import (  # noqa: F401
    _article_title_summary_lines,
    _card_from_summary,
    _clean_display_section_text,
    _display_internal_term_violation,
    _display_meta,
    _display_sections_from_strategy_result,
    _display_sections_missing_frontend_ready,
    _display_subject_from_summary,
    _display_summary_lines,
    _fix_display_particle_spacing,
    _is_grounded_display_candidate,
    _key_number_display_map,
    _llm_display_summary_lines,
    _merge_summary_lines,
    _numbered_summary_lines,
    _section_role_violation,
    _select_display_items,
    _slides,
    _slides_from_display_sections,
    _summary_line_for_display,
    _summary_line_priority,
    _summary_line_role,
    _sync_frontend_implication_from_display_sections,
    _sync_implication_frontend_from_display_sections,
)
from src.composers.card_news.evidence_media import (  # noqa: F401
    _article_image_urls,
    _dedupe_card_fact_basis,
    _default_sources,
    _evidence_chain,
    _fact_basis,
    _image_asset_score,
    _image_urls_from_value,
    _is_image_reference,
    _load_source_articles,
    _media_assets,
    _media_assets_for_cluster,
    _normalize_fact_basis_evidence_type,
    _rich_sources,
    _sector_has_direct_evidence,
    _source_article_ids,
    _source_indexes,
)
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
from src.composers.card_news.text_utils0 import (  # noqa: F401
    _accent_color,
    _action_semantic_role_for_index,
    _action_semantic_role_for_linkage_field,
    _article_content_sentences,
    _bounded_detail_items,
    _build_cluster_fetch_ids,
    _card_news_id,
    _card_signals,
    _card_text,
    _card_text_from_card,
    _clean_display_truncated_fragment,
    _clean_issue_term,
    _clean_source_headline,
    _compact_ascii,
    _compact_card_title,
    _compact_repeated_subject_in_phrase,
    _company_surface_replacements,
    _db_record,
    _dedupe_keep_order,
    _detail_line_candidates,
    _detail_line_key,
    _display_token_pieces,
    _first_amount_like_term,
    _first_list_item,
    _first_non_empty,
    _first_sentence_is_too_thin,
    _first_text,
    _follow_up_action_from_statement,
    _format_articles,
    _format_numeric_text,
    _get_llm,
    _grounding_path_matches,
    _has_final_consonant,
    _has_non_money_quantity,
    _has_numeric_or_period_signal,
    _has_schedule_signal,
    _has_target_capacity_or_schedule,
    _insight_candidates_from_strategy,
    _is_financial_only_title,
    _is_internal_analysis_copy,
    _is_labeled_public_frontend_item,
    _is_market_reaction_summary_line,
    _is_near_duplicate_detail,
    _is_signature_noun_phrase,
    _is_untranslated_english_text,
    _issue_execution_detail_phrase,
    _issue_signature,
    _issue_terms,
    _join_context_terms,
    _json_like_text,
    _labeled_frontend_ready_items,
    _linkage_area_records,
    _linkage_record_terms,
    _linkage_text,
    _list_dicts,
    _list_string,
    _list_value,
    _llm,
    _looks_like_action,
    _looks_like_article_boilerplate,
    _looks_like_non_summary_line,
    _looks_like_sentence_title,
    _low_specificity_display_match_term,
    _metadata,
    _normalize_card_news_statement_style,
    _normalize_display_token,
    _normalize_event_type,
    _normalize_peer_id,
    _normalize_summary_similarity_token,
    _now_iso,
    _optional_float,
    _optional_int,
    _order_articles,
    _parse_datetime_for_display,
    _parse_json,
    _polish_frontend_ready_internal_terms,
    _profile_linkage_level,
    _profile_phrase_from_peer_copy,
    _public_copy_cleanup,
    _public_sentences,
    _published_datetime,
    _set_card_signals,
    _skax_business_phrase_from_implication,
    _split_public_conclusion_evidence,
    _statement_to_check_subject,
    _string_or_none,
    _strip_number_prefix,
    _strip_public_section_prefixes,
    _subject_company_names,
    _surface_alias_pattern,
    _unique_nonempty,
)
from src.composers.card_news.text_utils1 import (  # noqa: F401
    _action_candidates_from_strategy,
    _actionize_detail_lines,
    _bounded_detail_lines,
    _business_area_record_has_issue_overlap,
    _business_context_title,
    _business_focus_from_summary,
    _clean_card_editorial_text,
    _clean_frontend_ready_text,
    _clean_structured_blocks,
    _clean_subject_phrase,
    _cleanup_public_frontend_implication,
    _contains_key_number_text,
    _editorial_candidate_has_minimum_grounding,
    _ensure_card_sentence,
    _fatal_summary_validation_issues,
    _format_public_frontend_item,
    _frontend_implication,
    _frontend_implication_from_result,
    _frontend_ready_display_items,
    _frontend_ready_section_blocks,
    _grounding_tokens,
    _has_specific_skax_linkage,
    _has_token_overlap,
    _implication,
    _implication_from_result,
    _industry_frontend_ready_display_items,
    _is_summary_repeat_insight,
    _linkage_has_concrete_profile_detail,
    _linkage_priority_phrase,
    _literal_summary_lines,
    _missing_fact_basis_line_indexes,
    _normalize_company_surface_names,
    _plain_summary_lines,
    _profile_phrase_from_linkage,
    _profile_reason_from_linkage,
    _profile_specificity_note,
    _published_date,
    _record_specificity_level,
    _signals,
    _skax_business_phrase_from_linkage,
    _skax_reason_from_linkage,
    _skax_specificity_note,
    _split_main_detail_block,
    _structured_blocks_from_labeled_lines,
    _subtitle,
    _summary_candidate_lines,
    _summary_fact_text,
    _summary_focus_tokens,
    _summary_line_count_valid,
    _summary_lines_too_similar,
    _summary_similarity_tokens,
    _trust_score,
    _validation_missing,
    _validation_pass,
    _validation_sc_score,
    _with_particle,
    _with_structured_frontend_blocks,
    mark_near_duplicate_card_candidates,
    peer_company_label,
)
from src.db.article_store import get_articles_by_ids

log = logging.getLogger(__name__)


class CardNewsComposer:
    """뉴스 요약 결과를 우선 사용해 card_news 저장/API 스키마를 생성한다."""

    def generate(
        self,
        summary: dict[str, Any],
        analysis: dict[str, Any] | None = None,
        classification: dict[str, Any] | None = None,
        articles: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """요약·분석 결과를 card_news 저장/API 스키마 호환 dict로 변환한다.

        DB 저장은 하지 않는다.
        """
        analysis = analysis or {}
        classification = classification or {}
        articles = articles if articles is not None else _load_source_articles(summary)

        cluster_id = _optional_int(summary.get("cluster_id"))
        peer_id = _normalize_peer_id(summary.get("main_company")) or _normalize_peer_id(
            classification.get("company")
        )
        trust_score = _trust_score(articles)
        source_article_ids = _source_article_ids(summary, articles)
        generated_at = _now_iso()
        published_date = _published_date(articles, generated_at)
        created_at = _published_datetime(articles, generated_at)

        title = _first_non_empty(
            summary.get("display_headline"),
            summary.get("headline"),
            summary.get("display_one_line_summary"),
            summary.get("one_line_summary"),
            analysis.get("analysis_summary"),
            classification.get("title"),
            (articles[0] or {}).get("title") if articles else "",
            "피어사 주요 뉴스",
        )
        title = _compact_card_title(
            title,
            summary=summary,
            classification=classification,
            articles=articles,
        )
        summary_lines = _plain_summary_lines(summary, use_llm=True)
        if not summary_lines:
            summary_lines = _article_title_summary_lines(articles)
        elif len(summary_lines) < _SUMMARY_LINE_MIN:
            summary_lines = _merge_summary_lines(
                summary_lines, _article_title_summary_lines(articles)
            )
        card_text = _card_text(title, summary_lines, summary, articles)
        event_type = _infer_event_type(summary, classification, card_text)
        title = _business_context_title(title, summary=summary, event_type=event_type)
        card_text = _card_text(title, summary_lines, summary, articles)
        sectors = _infer_sectors(classification, card_text)
        sector = sectors[0] if sectors else "other"
        exposure_score = _normalized_importance_score(
            classification=classification,
            card_text=card_text,
        )
        exposure_band = _importance_band(exposure_score)
        signals = _signals(
            classification=classification,
            summary=summary,
            source_article_ids=source_article_ids,
            card_text=card_text,
        )
        sources = _rich_sources(articles)
        media_assets = _media_assets(articles)
        cover_image = media_assets[0]["url"] if media_assets else _DEFAULT_COVER_IMAGE_URL
        validation_pass = _validation_pass(summary, analysis)
        validation_sc_score = _validation_sc_score(summary, analysis, trust_score)
        evidence_chain = _evidence_chain(
            summary=summary,
            analysis=analysis,
            sources=sources,
            source_article_ids=source_article_ids,
            cluster_id=cluster_id,
            created_at=created_at,
        )

        db_record = _db_record(
            card_id=_card_news_id(cluster_id, published_date),
            company=peer_id,
            cluster_id=cluster_id,
            title=title,
            summary_lines=summary_lines,
            event_type=event_type,
            importance=exposure_band,
            importance_score=exposure_score,
            sector=sector,
            sectors=sectors,
            signals=signals,
            evidence_chain=evidence_chain,
            sources=sources,
            validation_pass=validation_pass,
            validation_sc_score=validation_sc_score,
        )

        insights = _list_string(analysis.get("strategic_meaning")) or _list_string(
            summary.get("fact_summary")
        )
        display_sections = _display_sections_from_strategy_result(
            summary=summary,
            analysis=analysis,
            implication={},
            strategic_root={"analysis": analysis},
            sentence_grounding={},
            sources=sources,
            summary_lines=summary_lines,
        )
        return {
            **db_record,
            "peer_id": peer_id,
            "subtitle": _subtitle(analysis, classification),
            "category_label": sector.upper() if sector == "ax" else sector,
            "published_date": published_date,
            "exposure_band": exposure_band,
            "exposure_score": exposure_score,
            "implication": _implication(analysis),
            "sources": sources,
            "trust_score": trust_score,
            "source_count": len(sources),
            "financial_context": None,
            "display_sections": display_sections,
            "slides": _slides_from_display_sections(
                title=title,
                display_sections=display_sections,
                sources=sources,
                media_assets=media_assets,
            )
            or _slides(title, summary_lines, insights, sources, media_assets),
            "display": _display_meta(sector, cover_image),
            "is_human_reviewed": False,
            "is_bookmarked": False,
            "bookmark_count": 0,
            "share_count": 0,
            "created_at": created_at,
            "frontend_implication": _frontend_implication(analysis),
            "validation": {
                "pass": validation_pass,
                "sc_score": validation_sc_score,
            },
            "db_record": db_record,
        }

    def generate_from_analysis_package(
        self,
        analysis_package: AnalysisPackage | dict[str, Any],
        classification: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """AnalysisPackage 를 사용자용 카드뉴스/API 스키마로 재가공.

        W2-4: implication 이중 처리 제거. ImplicationAgent v4.0 (또는 v5.0) 의
        is_valid_implication=true 결과가 있으면 그것을 단일 출처로 사용하고,
        실패/누락 시에만 analysis 기반 frontend implication 으로 fallback.
        """
        package = (
            analysis_package.to_dict()
            if isinstance(analysis_package, AnalysisPackage)
            else analysis_package
        )
        input_bundle = package.get("input_bundle") or {}
        validation = package.get("validation") or {}
        classification = classification or validation.get("classification") or {}
        integrated_issue = package.get("integrated_issue") or package.get("summary") or {}
        card = self.generate(
            summary=integrated_issue,
            analysis=package.get("analysis") or {},
            classification=classification,
            articles=input_bundle.get("items") or [],
        )
        if not card:
            return {}
        implication_result = package.get("implication") or {}
        sentence_grounding = (
            package.get("sentence_grounding")
            or (package.get("evidence_payload") or {})
            .get("analysis_package", {})
            .get("sentence_grounding")
            or {}
        )
        is_llm_valid = bool(
            isinstance(implication_result, dict)
            and implication_result.get("is_valid_implication")
            and (
                (implication_result.get("skax_implication") or {}).get("why_important")
                or (implication_result.get("peer_implication") or {}).get("peer_meaning")
            )
        )
        if is_llm_valid:
            frontend_implication = _frontend_implication_from_result(
                implication_result,
                fallback=card.get("frontend_implication"),
                analysis=package.get("analysis") or {},
            )
            card["implication_result"] = implication_result
            card["implication"] = _implication_from_result(
                implication_result,
                fallback=card.get("implication"),
                frontend=frontend_implication,
            )
            card["frontend_implication"] = frontend_implication
        else:
            # ImplicationAgent 결과 없음/무효 → analysis 기반 frontend fallback 유지.
            card.setdefault(
                "frontend_implication", _frontend_implication(package.get("analysis") or {})
            )
        literal_summary_lines = _literal_summary_lines(integrated_issue)
        if literal_summary_lines and len(literal_summary_lines) < _SUMMARY_LINE_MIN:
            literal_summary_lines = _merge_summary_lines(
                literal_summary_lines,
                _article_title_summary_lines(input_bundle.get("items") or []),
            )
        if literal_summary_lines:
            card["summary_lines"] = literal_summary_lines
            if isinstance(card.get("db_record"), dict):
                card["db_record"]["summary_lines"] = literal_summary_lines
        display_sections = _display_sections_from_strategy_result(
            summary=integrated_issue,
            analysis=package.get("analysis") or {},
            implication=implication_result,
            strategic_root=package,
            sentence_grounding=sentence_grounding,
            sources=card.get("sources") or [],
            summary_lines=literal_summary_lines or _list_string(card.get("summary_lines")),
        )
        missing_frontend_sections = _display_sections_missing_frontend_ready(display_sections)
        card["needs_review"] = bool(missing_frontend_sections)
        if missing_frontend_sections:
            card["needs_review_reason"] = (
                "frontend_ready 직접 생성 문장이 없거나 품질 기준을 통과하지 못했습니다: "
                + ", ".join(missing_frontend_sections)
            )
        card["frontend_implication"] = _sync_frontend_implication_from_display_sections(
            card.get("frontend_implication"),
            display_sections=display_sections,
        )
        card["implication"] = _sync_implication_frontend_from_display_sections(
            card.get("implication"),
            frontend=card["frontend_implication"],
        )
        card["display_sections"] = display_sections
        card["slides"] = _slides_from_display_sections(
            title=str(card.get("title") or ""),
            display_sections=display_sections,
            sources=card.get("sources") or [],
            media_assets=_media_assets(input_bundle.get("items") or []),
        ) or card.get("slides", [])
        card["analysis_package"] = {
            "bundle_id": package.get("bundle_id"),
            "integrated_issue": integrated_issue,
            "summary": integrated_issue,
            "issue_understanding": package.get("issue_understanding") or {},
            "profile_linkage": package.get("profile_linkage") or {},
            "skax_response_linkage": package.get("skax_response_linkage") or {},
            "grounding_summary": package.get("grounding_summary") or {},
            "claim_strength": package.get("claim_strength"),
            "analysis": package.get("analysis") or {},
            "implication": implication_result,
            "sentence_grounding": sentence_grounding,
            "validation": validation,
        }
        return card

    def generate_from_cluster(
        self,
        cluster_id: int,
        representative_id: int,
        company: str,
        classification: dict[str, Any],
        cluster_article_ids: list[int] | None = None,
        peer_id: str = "",
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """클러스터 정보와 요약 결과로 카드뉴스를 생성한다.

        Args:
            cluster_id: 클러스터 ID.
            representative_id: 대표 기사 ID (가장 신뢰도 높은 기사).
            company: 회사 ID.
            classification: ClusterClassifier 결과.
            cluster_article_ids: 클러스터 내 전체 기사 ID 목록 (없으면 대표 기사만 사용).
            summary: SourceSummarizer가 만든 클러스터 사실 요약.

        Returns:
            card_news dict (저장 전 validation 없는 상태) — DB card_news 테이블 row 와 1:1.
        """
        company = company or peer_id

        # 카드 요약/출처 모두 클러스터 전체 기사를 기준으로 한다.
        article_ids = _build_cluster_fetch_ids(representative_id, cluster_article_ids)
        articles = _order_articles(get_articles_by_ids(article_ids), article_ids)
        if not articles:
            return {}

        summary_card = _card_from_summary(
            summary=summary or {},
            articles=articles,
            company=company,
            cluster_id=cluster_id,
            representative_id=representative_id,
            classification=classification,
        )
        if summary_card:
            log.info(
                "카드뉴스 생성 완료(summary) | id=%s sector=%s band=%s event=%s sources=%d",
                summary_card["id"],
                summary_card["sector"],
                summary_card["exposure_band"],
                summary_card["event_type"],
                len(articles),
            )
            return summary_card

        articles_text = _format_articles(articles)
        prompt = _ISSUE_CARD_PROMPT.replace("{articles_text}", articles_text)
        generated_at = _now_iso()
        published_date = _published_date(articles, generated_at)

        try:
            from src.observability import tracing_config

            response = _get_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="CardNewsComposer",
                    prompt_version=_PROMPT_VERSION,
                    company=company,
                    cluster_id=cluster_id,
                ),
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            card_data = _parse_json(content)

            card = {
                "id": _card_news_id(cluster_id, published_date),
                "company": company,
                "cluster_id": cluster_id,
                "representative_id": representative_id,
                "published_date": published_date,
                "title": card_data.get("title", articles[0]["title"][:100]),
                "summary_lines": card_data.get("summary_lines", []),
                "event_type": card_data.get(
                    "event_type",
                    classification.get("event_type", "tech"),
                ),
                # v3: 분류 결과의 sector·exposure 정보를 카드에 그대로 전파
                "sector": classification.get("sector", "other"),
                "sectors": classification.get("sectors", ["other"]),
                "exposure_score": classification.get("exposure_score", 0.0),
                "exposure_band": classification.get("exposure_band", "low"),
                "signals": classification.get("signals", {}),
                # 등급 자체는 v3에서 폐기되었으나, DB 컬럼 호환을 위해 노출도 밴드를 저장
                "importance": classification.get("importance", "low"),
                "importance_score": classification.get("importance_score", 0.0),
                "sources": _default_sources(articles),
            }
            _attach_card_news_schema_fields(card)

            log.info(
                "카드뉴스 생성 완료 | id=%s sector=%s band=%s event=%s sources=%d",
                card["id"],
                card["sector"],
                card["exposure_band"],
                card["event_type"],
                len(articles),
            )
            return card

        except Exception as e:
            log.error("카드뉴스 생성 실패 | cluster=%d error=%s", cluster_id, e)
            return {}


__all__ = [
    "CardNewsComposer",
    "mark_near_duplicate_card_candidates",
    "peer_company_label",
]
