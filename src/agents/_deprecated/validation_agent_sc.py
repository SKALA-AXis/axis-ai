# 작성일: 2026-04-27
# 작성자: 최종민
# 변경이력:
#   2026-04-27 최종민 — v3 전환(Evidence Chain)으로 SC 검증 에이전트 추가 및 ruff 포맷 적용
"""SC 검증 에이전트 — Self-Consistency 환각 방지."""

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)

SC_THRESHOLD = 2 / 3  # 3회 중 2회 이상 일치
SC_RUNS = 3

# temperature=0.9: SC 검증용 다양성 확보
_llm_sc = ChatOpenAI(model="gpt-4o", temperature=0.9, max_tokens=800)
_llm_judge = ChatOpenAI(model="gpt-4o", temperature=0.0, max_tokens=300)

_SC_PROMPT = """\
당신은 SK AX 전략기획팀의 AI 어시스턴트입니다.
아래 이슈 카드를 바탕으로 SK AX 관점의 핵심 시사점을 1~2문장으로 작성해주세요.
출처에 없는 수치(금액·%·날짜)는 포함하지 마세요.

제목: {title}
요약: {summary}
출처: {sources_text}

핵심 시사점:"""

_JUDGE_PROMPT = """\
아래 세 개의 시사점이 핵심 주장에서 일치하는지 판단해주세요.
세부 표현은 달라도 괜찮으나, 주요 주장(사업 영향 방향, 핵심 우려사항)이 일치해야 합니다.

[시사점 1]: {s1}
[시사점 2]: {s2}
[시사점 3]: {s3}

일치하는 쌍의 수(0~3)와 이유를 JSON으로 응답하세요:
{{"consistent_pairs": 0, "reason": "..."}}"""

# 출처에 없는 수치 패턴 (자동 Fail)
_NUMERIC_PATTERN = re.compile(r"\d+[억조만천백십]?\s*원|\d+\.?\d*\s*%|\d{4}년\s*\d{1,2}월")


class ValidationAgent:
    """SC 검증 (3회 독립 생성 → 2/3 이상 일치 시 Pass)."""

    def validate(self, issue_card: dict[str, Any], implication: dict[str, Any]) -> dict[str, Any]:
        """Self-Consistency 검증을 수행한다.

        Args:
            issue_card: 이슈 카드 dict.
            implication: ImplicationAgent 결과.

        Returns:
            {'pass': bool, 'sc_score': float, 'reason': str}
        """
        card_id = issue_card.get("id", "")
        log.info("SC 검증 시작 | card_id=%s", card_id)

        # confidence < 0.6 → 즉시 Fail
        if implication.get("confidence", 0) < 0.6:
            log.info(
                "SC 검증 Fail — confidence 부족 | card_id=%s confidence=%.2f",
                card_id,
                implication.get("confidence", 0),
            )
            return {"pass": False, "sc_score": 0.0, "reason": "근거 불충분 (confidence < 0.6)"}

        # 출처에 없는 수치 포함 시 자동 Fail
        why = implication.get("why_important", "")
        impact = implication.get("potential_impact", "")
        if _NUMERIC_PATTERN.search(why + impact):
            log.warning("SC 검증 Fail — 출처 없는 수치 감지 | card_id=%s", card_id)
            return {"pass": False, "sc_score": 0.0, "reason": "출처에 없는 수치 포함"}

        # 3회 독립 생성
        summaries = _generate_sc_runs(issue_card)
        if len(summaries) < 2:
            return {"pass": False, "sc_score": 0.0, "reason": "SC 생성 실패"}

        # 일치율 판정
        sc_score, reason = _judge_consistency(summaries)
        passed = sc_score >= SC_THRESHOLD

        log.info(
            "SC 검증 완료 | card_id=%s pass=%s sc_score=%.2f",
            card_id,
            passed,
            sc_score,
        )
        return {"pass": passed, "sc_score": sc_score, "reason": reason}


def _generate_sc_runs(issue_card: dict[str, Any]) -> list[str]:
    """동일 이슈로 시사점 SC_RUNS회 독립 생성."""
    title = issue_card.get("title", "")
    summary = "\n".join(issue_card.get("summary_lines", []))
    sources_text = "\n".join(
        f"[{s.get('index', i + 1)}] {s.get('title', '')} ({s.get('source_name', '')})"
        for i, s in enumerate(issue_card.get("sources", []))
    )
    prompt = (
        _SC_PROMPT.replace("{title}", title)
        .replace("{summary}", summary)
        .replace("{sources_text}", sources_text)
    )

    results = []
    for i in range(SC_RUNS):
        try:
            resp = _llm_sc.invoke(prompt)
            results.append(resp.content.strip())
        except Exception as e:
            log.warning("SC 생성 %d회 실패 | error=%s", i + 1, e)
    return results


def _judge_consistency(summaries: list[str]) -> tuple[float, str]:
    """GPT-4o로 일치율을 판정한다."""
    padded = (summaries + [""] * SC_RUNS)[:SC_RUNS]
    prompt = (
        _JUDGE_PROMPT.replace("{s1}", padded[0])
        .replace("{s2}", padded[1])
        .replace("{s3}", padded[2])
    )
    try:
        resp = _llm_judge.invoke(prompt)
        data = _parse_json(resp.content)
        pairs = min(int(data.get("consistent_pairs", 0)), 3)
        score = pairs / 3.0
        return score, data.get("reason", "")
    except Exception as e:
        log.warning("SC 일치율 판정 실패 | error=%s", e)
        # fallback: 단순 문자열 유사도
        return _simple_consistency(summaries)


def _simple_consistency(summaries: list[str]) -> tuple[float, str]:
    """GPT 판정 실패 시 단순 키워드 오버랩으로 대체."""
    if len(summaries) < 2:
        return 0.0, "생성 부족"
    words0 = set(summaries[0].split())
    words1 = set(summaries[1].split())
    overlap = len(words0 & words1) / max(len(words0 | words1), 1)
    passed = overlap >= 0.3
    return (2 / 3 if passed else 0.0), f"키워드 오버랩: {overlap:.2f}"


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
