"""Issue integration policy values.

이 모듈은 IssueIntegrationAgent의 고정 정책값을 한곳에 모은다.
서비스 본문에는 임의 숫자를 직접 두지 않고, 필요한 값은 이 정책 객체를 통해
참조한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class IntegrationPolicy:
    """Runtime policy for deterministic issue integration.

    The budgets below protect the next LLM prompt from unbounded parser chunks.
    AXIS의 기존 AnalysisContextBuilder도 한국어+영문 혼합 텍스트를 약 2.5 chars/token
    으로 압축 추정한다. 같은 기준을 사용해 IntegratedIssue evidence section을
    AnalysisAgent 입력 예산 안에 유지한다.
    """

    char_per_token_estimate: float = 2.5
    target_evidence_tokens: int = 1800
    compact_evidence_tokens: int = 120
    fact_summary_tokens: int = 220
    minimum_coverage_per_group: int = 1
    sentence_window_tokens: int = 120
    uncertainty_markers: tuple[str, ...] = (
        "예정",
        "계획",
        "전망",
        "가능성",
        "기대",
        "추정",
        "목표",
        "will",
        "plan",
        "expected",
        "forecast",
        "guidance",
    )
    numeric_markers: tuple[str, ...] = (
        "%",
        "원",
        "조",
        "억",
        "만",
        "달러",
        "usd",
        "krw",
        "q",
        "분기",
        "매출",
        "영업이익",
        "ebitda",
    )
    materiality_terms: tuple[str, ...] = (
        "계약",
        "수주",
        "공시",
        "매출",
        "영업이익",
        "가이던스",
        "투자",
        "리스크",
        "규제",
        "전략",
        "사업",
        "고객",
        "플랫폼",
        "ai",
        "ax",
        "cloud",
        "msp",
        "security",
    )
    provenance_quality: dict[str, float] = field(
        default_factory=lambda: {
            "financial_metric": 1.0,
            "business_signal": 0.95,
            "risk_fact": 0.9,
            "parser_chunk": 0.85,
            "parser_section": 0.8,
            "title": 0.55,
            "content_sentence": 0.65,
            "unknown": 0.5,
        }
    )
    rank_component_weights: dict[str, float] = field(
        default_factory=lambda: {
            "provenance_score": 0.35,
            "materiality_density": 0.25,
            "numeric_density": 0.2,
            "section_specificity": 0.1,
            "evidence_balance": 0.1,
        }
    )
    quality_weights: dict[str, float] = field(
        default_factory=lambda: {
            "fact_presence": 0.25,
            "evidence_presence": 0.25,
            "source_coverage": 0.2,
            "structured_signal_presence": 0.2,
            "uncertainty_penalty": 0.1,
        }
    )
    # Content digest budgets are intentionally separate from fact/evidence budgets.
    # They carry source-body explanation for the next AnalysisAgent prompt while keeping
    # each raw article compressed enough to avoid unbounded raw content injection.
    source_content_digest_tokens: int = 220
    integrated_content_digest_tokens: int = 520
    source_content_key_point_limit: int = 5
    integrated_content_key_point_limit: int = 10
    source_body_extract_limit: int = 6
    integrated_body_extract_limit: int = 18
    content_section_limit: int = 8
    content_section_keyword_limit: int = 8
    content_sentence_score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "materiality_terms": 1.0,
            "numeric_density": 1.0,
            "title_overlap": 0.8,
            "parser_section": 0.7,
            "position_signal": 0.4,
            "uncertainty_marker": 0.3,
        }
    )
    frame_candidate_confidence: dict[str, float] = field(
        default_factory=lambda: {
            "input_bundle": 0.95,
            "source_metadata": 0.9,
            "article_metadata": 0.85,
            "parser_result": 0.82,
            "title_keyword": 0.78,
            "content_keyword": 0.62,
            "fact_keyword": 0.7,
            "fallback": 0.35,
        }
    )
    frame_topic_limit: int = 12
    frame_entity_limit: int = 20
    frame_relation_limit: int = 10
    frame_timeline_limit: int = 12
    frame_number_limit: int = 20
    # raw_articles row lifecycle values are policy-owned because they are pipeline contracts,
    # not extraction logic. These values mirror current crawler/relevance status semantics:
    # skipped/irrelevant/error rows should not drive analysis, while successful crawls may.
    ineligible_processing_statuses: tuple[str, ...] = ("skipped",)
    ineligible_relevance_labels: tuple[str, ...] = ("irrelevant",)
    eligible_crawl_statuses: tuple[str, ...] = ("success", "ok", "completed")

    @property
    def target_evidence_chars(self) -> int:
        return int(self.target_evidence_tokens * self.char_per_token_estimate)

    @property
    def compact_evidence_chars(self) -> int:
        return int(self.compact_evidence_tokens * self.char_per_token_estimate)

    @property
    def fact_summary_chars(self) -> int:
        return int(self.fact_summary_tokens * self.char_per_token_estimate)

    @property
    def sentence_window_chars(self) -> int:
        return int(self.sentence_window_tokens * self.char_per_token_estimate)

    @property
    def source_content_digest_chars(self) -> int:
        return int(self.source_content_digest_tokens * self.char_per_token_estimate)

    @property
    def integrated_content_digest_chars(self) -> int:
        return int(self.integrated_content_digest_tokens * self.char_per_token_estimate)


DEFAULT_POLICY = IntegrationPolicy()


__all__ = ["DEFAULT_POLICY", "IntegrationPolicy"]
