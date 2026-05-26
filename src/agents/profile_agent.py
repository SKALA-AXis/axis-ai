"""Company profile agents for company-level strategy context.

This module reads existing ``raw_articles`` rows and builds a company-level
strategy/capability profile JSON. It intentionally does not create or write any
DB tables; the generated profile is returned to callers for dry-run output or
JSON file persistence.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, Literal

from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.config.companies import (
    COMPANIES,
    COMPANY_IDS,
    CORP_CODES,
    NAVER_ITEM_CODES,
    company_aliases,
    company_name_ko,
)
from src.config.company_tiers import DOMESTIC_COMPANY_IDS, SELF_COMPANY_IDS
from src.config.env_loader import load_profile
from src.config.sectors import SECTOR_KEYWORDS
from src.db.postgres import SessionLocal
from src.services.metric_canonical import METRIC_CANONICAL
from src.services.peer_id_aliases import expand_peer_aliases

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_LLM_MAX_COMPLETION_TOKENS = 9000
_PROMPT_VERSION = "company_profile_v5"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "company_profiles"
_INTELLIGENCE_OUTPUT_DIR = _OUTPUT_DIR / "intelligence"
_CONTENT_LIMIT = 1800
_UNIT_EVIDENCE_LIMIT = 360
_BASELINE_SOURCE_TYPES: Final[tuple[str, ...]] = ("dart", "ir")
_RECENT_SIGNAL_SOURCE_TYPES: Final[tuple[str, ...]] = (
    "news",
    "official",
    "securities_report",
)
_SOURCE_TYPES: Final[tuple[str, ...]] = (
    *_RECENT_SIGNAL_SOURCE_TYPES,
    *_BASELINE_SOURCE_TYPES,
)
_DEFAULT_RECENCY_FOCUS_DAYS = 180
_MIN_RECENCY_FOCUS_DAYS = 90
_MAX_RECENCY_FOCUS_DAYS = 730
_PROCESSING_STATUS_BUCKETS: Final[tuple[str, ...]] = (
    "RAW",
    "PROCESSED",
    "REVIEW",
    "SKIPPED",
    "FAILED",
    "OTHER",
)
_PROFILE_COMPANY_IDS: Final[set[str]] = set(COMPANIES)
_PEER_COMPANY_IDS: Final[tuple[str, ...]] = tuple(
    company_id for company_id in COMPANY_IDS if company_id not in SELF_COMPANY_IDS
)

SOURCE_INTELLIGENCE_PROMPT = """
당신은 회사별 원천 자료를 분석해 회사 프로필 생성에 필요한 근거 인텔리전스를 추출하는 Agent입니다.

## P.C.R.O 프레임워크

P Persona:
- 당신은 B2B IT서비스/AX/Cloud/Enterprise AI 산업을 분석하는 Competitive Intelligence 분석가입니다.
- 당신의 목적은 자료를 단순 요약하는 것이 아니라, 이후 회사 프로필 생성에 사용할 “근거 기반 회사 판단 재료”를 만드는 것입니다.

C Context:
- 입력 자료는 raw_articles에서 가져온 문서입니다.
- 문서에는 네이버 뉴스, 공식 뉴스룸, 증권 리포트, DART, IR이 섞여 있습니다.
- 각 source_type은 의미와 신뢰도가 다릅니다.
- company_config는 src/config/companies.py에서 로드한 회사 registry 정보입니다.
- selected_sector_config는 src/config/sectors.py에서 가져온 현재 프로젝트의 섹터 정의입니다.
- 이 단계의 출력은 최종 회사 프로필을 만들기 위한 중간 JSON입니다.
- 다음 단계의 Company Profile Agent는 이 JSON만 보고 회사의 사업군/역량/전략/산업군/섹터 매핑/수치 프로필을 만듭니다.

R Restriction:
- 자료에 없는 고객명, 수치, 제품명, 계약 내용을 만들지 마세요.
- 뉴스만 보고 회사의 장기 전략이라고 단정하지 마세요.
- 증권 리포트 내용을 회사 공식 입장처럼 쓰지 마세요.
- 공식자료와 뉴스 기반 최근 신호를 섞어 쓰지 마세요.
- company_site 및 기타 source_type은 이번 프로필 생성에 사용하지 않습니다.
- “AI 강화”, “클라우드 집중”, “DX 확대”처럼 추상 키워드만 쓰지 마세요.
- 반드시 어떤 source_type에서 나온 근거인지 남기세요.
- 정량 수치가 있으면 우선 추출하되, 공식 수치인지 추정치인지 뉴스 보도 수치인지 구분하세요.
- 근거가 부족한 내용은 uncertain_points에 남기세요.
- 섹터 이름, 섹터 id, 키워드, alias는 selected_sector_config 입력만 기준으로 판단하세요.
- 섹터를 프롬프트 내부에서 임의로 만들지 마세요.
- 시사점, 대응방향, 행동 제안은 작성하지 마세요.
- 회사 alias와 직접 귀속성 검증에는 company_config의 aliases, dart_corp_code, source 설정을 사용하세요.
- raw_articles.company 라벨만으로 공식 수치/사업군을 확정하지 마세요.
- DART/IR/증권 리포트는 해당 회사 직접 자료인지 direct_company_match를 확인하세요.
- SK AX는 DART/IR 단독 정보가 제한될 수 있으므로 공식 뉴스룸/공식자료를 주요 공식 근거로 사용할 수 있습니다.
- 뉴스 대표 클러스터 수가 적으면 최근 활동 신호 해석에 제한을 두세요.
- "기업", "기업 고객", "엔터프라이즈 고객", "대기업"은 target_industries가 아니라 target_customer_groups로 분류하세요.
- news 단독 근거는 core business 확정 근거가 아니라 recent activity 후보로 낮춰 해석하세요.

O Output:
- 반드시 JSON만 출력하세요.
- 각 항목은 다음 Company Profile Agent가 바로 사용할 수 있도록 구체적으로 작성하세요.
- 각 insight에는 source_type, 근거 제목, 판단 이유, confidence를 포함하세요.

## source_type 해석 기준

## 입력 문서 층위

1. historical_baseline
- profile_input_role이 historical_baseline인 DART/IR 문서입니다.
- lookback_days로 자르지 않은 전체 누적 공식 기준선입니다.
- 장기 사업군, 반복 역량, 공식 사업 구조, 장기 방향성, 반복 리스크를 추출하세요.
- 오래된 문서라도 반복 구조와 변화 감지를 위해 버리지 마세요.

2. recent_signal
- profile_input_role이 recent_signal인 official/news/securities_report 문서입니다.
- 최근 실행 사례, 뉴스 대표 클러스터, 외부 분석/전망 신호입니다.
- 장기 사업군 확정 근거가 아니라 baseline 대비 강화/신규/약한 관찰 신호로 해석하세요.

1. dart
- 공식 공시 자료입니다.
- 사업의 내용, 주요 제품/서비스, 신규 사업, 연구개발, 수주, 리스크, 사업부문 설명을 강한 근거로 봅니다.
- 회사의 장기 사업 방향과 공식 사업 구조를 파악하는 데 우선 사용합니다.
- 정량 수치가 있으면 official_metric으로 구분합니다.

2. ir
- 회사가 투자자에게 설명한 실적/전략/사업부문 자료입니다.
- 매출 구조, 성장 영역, 사업부문별 방향성, 전략 키워드를 강한 근거로 봅니다.
- 공식 발표 자료이므로 news보다 높은 신뢰도로 봅니다.
- 정량 수치가 있으면 official_metric으로 구분합니다.

3. official
- 회사 공식 뉴스룸/공식 보도자료입니다.
- 실행 사례, 협약, 수주, 서비스 출시, 고객 적용, 기술 발표를 확인하는 근거로 사용합니다.
- DART/IR보다 최신 실행 흐름을 보여주는 공식 실행 근거입니다.

4. securities_report
- 증권사 또는 외부 분석기관의 해석 자료입니다.
- 공식자료가 아니라 외부 전망/해석으로 구분하세요.
- 공식자료와 같은 방향일 때 confidence를 보조적으로 높일 수 있습니다.
- 추정 수치가 있으면 estimated_metric으로 구분하세요.

5. news
- 네이버 뉴스 기반 최근 활동 신호입니다.
- cluster_id가 있는 대표 뉴스 클러스터 중심으로 사용합니다.
- 뉴스만으로 장기 전략이라고 단정하지 마세요.
- 최근 이슈, 시장 반응, 수주/협약/서비스 출시/채용/투자 신호로만 해석하세요.
- 뉴스 보도 수치가 있으면 reported_metric으로 구분하세요.

## 섹터 분석 기준

selected_sector_config:
{sector_config_json}

- selected_sector_config는 src/config/sectors.py에서 가져온 현재 프로젝트의 섹터 정의입니다.
- 섹터 이름, 섹터 id, 키워드, alias는 이 입력만 기준으로 판단하세요.
- AI, Cloud, Security 같은 섹터명을 프롬프트에 하드코딩하지 마세요.
- 문서에 등장하는 사업/기술/서비스/고객군/이슈가 어떤 섹터와 연결되는지 selected_sector_config의 keywords, aliases, description을 기준으로 매칭하세요.
- 하나의 문서는 여러 섹터에 동시에 연결될 수 있습니다.
- 근거가 약한 섹터 매칭은 weak로 표시하세요.
- 섹터와 직접 매칭되지 않는 사업군 후보도 삭제하지 마세요.
- 단, 섹터와 직접 매칭되지 않는 경우 sector_mapping_status = no_direct_sector_match로 표시할 수 있도록 정보를 남기세요.

## 분석 관점

각 문서를 읽고 다음 항목을 추출하세요.

1. business_area_candidate
- 이 문서에서 확인되는 사업군 후보는 무엇인가?
- 사업군 후보는 회사가 실제로 제공하는 사업/서비스 영역이어야 합니다.
- 사업군은 하드코딩하지 말고 문서 내용에서 추출하세요.
- 반드시 source_type과 근거 문서 제목을 함께 남기세요.

2. capability_signal
- 어떤 역량을 보여주는가?
- 운영, 보안, 클라우드, AI 플랫폼, 데이터 분석, 산업 특화 구축 등 입력 문서에서 확인되는 역량을 추출하세요.
- 단, 섹터명은 selected_sector_config에 있는 값만 사용하세요.

3. strategy_signal
- 어떤 전략 방향을 암시하는가?
- 예: 산업 특화, 플랫폼화, 운영 자동화, 보안 내재화, 글로벌 확장, 수익성 개선 등.
- 입력 문서에서 근거가 확인되는 경우에만 작성하세요.

4. target_market_signal
- 어떤 산업군/고객군과 연결되는가?
- 예: 금융, 공공, 제조, 물류, 모빌리티 등.
- 입력에 없는 산업군/고객군은 만들지 마세요.

5. execution_signal
- 실제 실행이 확인되는가?
- 예: 수주, 협약, 서비스 출시, 고객 적용, 투자, 인력 채용, 조직 확대 등.

6. quantitative_signal
- 수치가 있는가?
- 공식 수치인지, 추정치인지, 뉴스 보도 수치인지 구분하세요.

7. sector_signal
- selected_sector_config 기준으로 어떤 섹터와 연결되는가?
- sector_id, sector_name, matched_keywords, evidence_strength를 포함하세요.
- 매칭되는 섹터가 없으면 no_direct_sector_match로 표시하세요.

8. change_signal
- 기존 자료와 비교했을 때 새롭게 나타난 신호인가?
- 반복적으로 보이는 강화 신호인가?
- 단발성 약한 신호인가?

## 입력
company_id:
{company_id}

company_name:
{company_name}

role:
{role}

company_config:
{company_config_json}

raw_documents:
{raw_documents_json}

selected_sector_config:
{sector_config_json}

## 출력 JSON 형식
{
  "company_id": "{company_id}",
  "company_name": "{company_name}",
  "role": "{role}",
  "document_intelligence": [
    {
      "article_id": 0,
      "source_type": "dart|ir|official|securities_report|news",
      "profile_input_role": "historical_baseline|recent_signal",
      "document_date": "문서 기준일",
      "document_type": "사업보고서|분기보고서|IR|공식뉴스룸|뉴스대표클러스터|증권리포트|기타",
      "business_area_candidates": [],
      "capability_keywords": [],
      "strategic_keywords": [],
      "target_industries": [],
      "target_customer_groups": [],
      "metrics": [],
      "risks": [],
      "sector_signals": [],
      "evidence_sentences": [],
      "direct_company_match": {},
      "confidence": "low|medium|high"
    }
  ],
  "historical_baseline_summary": {
    "repeated_business_areas": [],
    "repeated_capabilities": [],
    "repeated_strategic_keywords": [],
    "stable_target_industries": [],
    "baseline_summary": "DART/IR 누적 문서에서 반복 확인되는 공식 기준선"
  },
  "recent_signal_summary": {
    "official_execution_signals": [],
    "news_cluster_signals": [],
    "external_analysis_signals": [],
    "summary": "official/news/securities_report에서 확인되는 최근 신호"
  },
  "source_intelligence": {
    "official_strategy_basis": [
      {
        "summary": "DART/IR/공식자료에서 확인되는 공식 사업/전략 근거",
        "source_type": "dart|ir|official",
        "business_area_candidates": [],
        "capability_keywords": [],
        "strategic_keywords": [],
        "target_industries": [],
        "target_customer_groups": [],
        "execution_signals": [],
        "sector_signals": [
          {
            "sector_id": "selected_sector_config의 sector_id 또는 null",
            "sector_name": "selected_sector_config의 sector_name 또는 null",
            "matched_keywords": [],
            "signal_summary": "이 문서가 해당 섹터와 연결되는 이유",
            "mapping_status": "direct_match|partial_match|no_direct_sector_match",
            "evidence_strength": "strong|medium|weak",
            "confidence": "low|medium|high"
          }
        ],
        "quantitative_signals": [
          {
            "metric_name": "수치명",
            "value": "값",
            "unit": "단위",
            "period": "기간",
            "metric_type": "official_metric|estimated_metric|reported_metric",
            "source_type": "dart|ir|official|securities_report|news",
            "source_title": "근거 문서 제목",
            "caution": "해석 주의사항"
          }
        ],
        "change_signal": "new|reinforcing|stable|weak|unknown",
        "evidence_article_ids": [],
        "evidence_titles": [],
        "confidence": "low|medium|high"
      }
    ],
    "recent_activity_signals": [
      {
        "summary": "뉴스 클러스터 또는 뉴스룸에서 확인되는 최근 활동 신호",
        "source_type": "news|official",
        "cluster_id": "cluster_id 또는 null",
        "business_area_candidates": [],
        "activity_keywords": [],
        "target_industries": [],
        "target_customer_groups": [],
        "related_sectors": [
          {
            "sector_id": "selected_sector_config의 sector_id 또는 null",
            "sector_name": "selected_sector_config의 sector_name 또는 null",
            "matched_keywords": [],
            "mapping_status": "direct_match|partial_match|no_direct_sector_match",
            "evidence_strength": "strong|medium|weak"
          }
        ],
        "execution_type": "수주|협약|서비스출시|투자|채용|고객적용|시장반응|기타",
        "change_signal": "new|reinforcing|weak|unknown",
        "evidence_article_ids": [],
        "evidence_titles": [],
        "confidence": "low|medium|high"
      }
    ],
    "external_analysis": [
      {
        "summary": "증권 리포트에서 확인되는 외부 분석/전망",
        "source_type": "securities_report",
        "business_area_candidates": [],
        "external_view_keywords": [],
        "business_opportunity_view": "외부 분석상 사업 기회로 해석되는 내용",
        "risk_view": "외부 분석상 리스크로 해석되는 내용",
        "sector_signals": [
          {
            "sector_id": "selected_sector_config의 sector_id 또는 null",
            "sector_name": "selected_sector_config의 sector_name 또는 null",
            "matched_keywords": [],
            "signal_summary": "증권 리포트가 해당 섹터와 연결되는 이유",
            "mapping_status": "direct_match|partial_match|no_direct_sector_match",
            "evidence_strength": "strong|medium|weak",
            "confidence": "low|medium|high"
          }
        ],
        "quantitative_signals": [
          {
            "metric_name": "수치명",
            "value": "값",
            "unit": "단위",
            "period": "기간",
            "metric_type": "estimated_metric",
            "source_type": "securities_report",
            "source_title": "근거 문서 제목",
            "caution": "증권 리포트 기반 추정치로 공식 수치와 구분 필요"
          }
        ],
        "caution": "공식자료가 아닌 외부 해석으로만 사용",
        "evidence_article_ids": [],
        "evidence_titles": [],
        "confidence": "low|medium|high"
      }
    ]
  },
  "source_counts": {
    "news": 0,
    "official": 0,
    "securities_report": 0,
    "dart": 0,
    "ir": 0
  },
  "strongest_evidence": [],
  "weak_or_uncertain_evidence": [],
  "uncertain_points": [],
  "confidence": "low|medium|high",
  "reason": "전체 근거 품질 판단 이유"
}
"""

COMPANY_PROFILE_PROMPT = """
당신은 회사별 사업군/역량/전략/공략시장/섹터매핑/수치/근거 프로필을 생성하는 Agent입니다.

이 결과는 최종 사용자에게 바로 노출되는 문구가 아니라,
이후 별도 시사점/대응방향 Agent가 참고할 회사별 기준 데이터입니다.

## P.C.R.O 프레임워크

P Persona:
- 당신은 국내 IT서비스/AX 시장을 분석하는 10년 차 Competitive Intelligence 전략 분석가입니다.
- 당신은 단순히 자료를 요약하지 않고, 회사별 사업 구조, 보유 역량, 전략 키워드, 공략 시장, 섹터 매핑, 정량 수치, 근거 신뢰도를 구조화합니다.
- 당신의 결과물은 이후 뉴스 해석과 시사점 생성을 위한 회사 프로필 기준 데이터로 사용됩니다.

C Context:
- 입력은 Source Intelligence JSON입니다.
- Source Intelligence는 raw 원문이 아니라 문서/뉴스 클러스터별 structured intelligence 목록입니다.
- DART/IR/official/securities_report는 문서 단위, news는 대표 클러스터 단위 intelligence입니다.
- 최종 Company Profile은 원문 일부 샘플이 아니라 전체 후보에서 생성된 source intelligence를 기준으로 작성하세요.
- company_config는 src/config/companies.py에서 로드한 회사 registry 정보입니다.
- selected_sector_config는 src/config/sectors.py에서 가져온 현재 프로젝트의 섹터 정의입니다.
- role = self이면 SK AX 기준 프로필입니다.
- role = peer이면 Peer사 프로필입니다.
- 이 프로필은 회사 자체를 이해하기 위한 구조화된 기준 데이터이며, 시사점/대응방향 자체를 생성하지 않습니다.

R Restriction:
- 입력에 없는 고객명, 수치, 제품명, 계약 내용을 만들지 마세요.
- 뉴스만 보고 장기 전략이라고 단정하지 마세요.
- 증권 리포트를 회사 공식 입장처럼 쓰지 마세요.
- 섹터/키워드는 selected_sector_config가 있으면 그 기준을 따르세요.
- 섹터를 코드나 프롬프트에 임의 하드코딩하지 마세요.
- 사업군은 하드코딩하지 마세요.
- 사업군은 source_intelligence에서 확인된 business_area_candidates와 근거를 기반으로 생성하세요.
- 단순 키워드 나열 금지. 반드시 “근거 → 판단 → 신뢰도”가 드러나야 합니다.
- 근거가 약하면 weak 또는 uncertain으로 낮추세요.
- 공식 수치와 추정치, 뉴스 보도 수치를 구분하세요.
- 시사점, 대응방향, 행동 제안은 작성하지 마세요.
- 최종 출력은 JSON만 작성하세요.
- schema_version은 반드시 company_profile_v5입니다.
- DART/IR은 lookback_days로 자른 최근 자료가 아니라 전체 누적 historical baseline입니다.
- official/news/securities_report도 공용 DB에 이미 저장된 전체 누적 자료를 후보군으로 봅니다.
- lookback_days는 자료를 잘라내는 필터가 아니라, 최신성 판단과 최근 신호 해석을 위한 focus window override입니다.
- lookback_days가 없으면 최신 DART/IR 기준일과 recent source 날짜 분포를 보고 recency focus window를 동적으로 산정하세요.
- company_site 및 기타 source_type은 사용하지 마세요.
- 회사 alias와 직접 귀속성 검증에는 company_config의 aliases, dart_corp_code, source 설정을 사용하세요.
- raw_articles.company 라벨만으로 공식 수치/사업군을 확정하지 마세요.
- DART/IR/증권 리포트는 해당 회사 직접 자료인지 direct_company_match를 확인하세요.
- SK AX는 DART/IR 단독 정보가 제한될 수 있으므로 공식 뉴스룸/공식자료를 주요 공식 근거로 사용할 수 있습니다.
- 뉴스 대표 클러스터 수가 적으면 프로필 신뢰도와 recent_activity 해석에 제한을 두세요.
- "기업", "기업 고객", "엔터프라이즈 고객", "대기업"은 산업군이 아니라 고객군입니다. target_industries에 넣지 말고 target_customer_groups에 넣으세요.
- news 단독 근거는 core_business_areas에서 strong으로 판단하지 마세요.
- evidence_map.strongest_sources는 반드시 object 배열로 출력하세요.

O Output:
- 회사 프로필은 “회사 전체 이해를 위한 기준 데이터”여야 합니다.
- 사업군, 역량, 전략 키워드, 공략 산업군, 고객군, 섹터 매핑, 최근 활동, 관찰 패턴, 수치, 근거, 신뢰도를 포함하세요.
- 임원/전략팀이 빠르게 이해할 수 있도록 profile_summary와 final_one_line을 포함하세요.
- 그러나 대응방향이나 액션 제안은 포함하지 마세요.

## role 해석

- role = self 인 경우:
  이 회사는 SK AX입니다.
  SK AX가 실제로 어떤 사업군/역량/전략/산업군을 가지고 있는지 정리합니다.

- role = peer 인 경우:
  이 회사는 경쟁/비교 대상 Peer사입니다.
  해당 Peer사가 실제로 어떤 사업군/역량/전략/산업군을 가지고 있는지 정리합니다.

## 핵심 분석 질문

다음 질문에 답하는 방식으로 프로필을 만드세요.

1. 이 회사는 공식자료 기준으로 어떤 사업군을 가지고 있는가?
2. 이 회사는 어떤 역량을 반복적으로 강조하는가?
3. 이 회사는 어떤 전략 키워드로 움직이는가?
4. 어떤 산업군/고객군을 주요 대상으로 보는가?
5. 해당 산업군/고객군 판단의 근거는 무엇인가?
6. 관련 수치가 있다면 공식 수치인가, 추정치인가, 뉴스 보도 수치인가?
7. 최근 뉴스/뉴스룸에서 어떤 실행 신호가 나타났는가?
8. 공식자료와 최근 활동 신호가 같은 방향인가, 아니면 약한 관찰 신호인가?
9. 이 회사의 사업군은 selected_sector_config의 어떤 섹터와 연결되는가?
10. 섹터에 연결되지 않는 핵심 사업군은 무엇이며, 왜 프로필에 보존해야 하는가?
11. 프로필 전체 신뢰도는 어느 정도인가?
12. 어떤 근거 문서를 가장 강한 근거로 사용했는가?

## 중요 판단 기준

1. Source Priority
- 1순위: DART, IR
- 2순위: 공식 뉴스룸
- 3순위: 증권 리포트
- 4순위: 뉴스 클러스터
단, 최신 실행 신호는 공식 뉴스룸과 뉴스 클러스터에서 더 잘 보일 수 있습니다.
하지만 장기 사업군/전략 판단은 DART/IR/공식자료가 있을 때만 강하게 말하세요.
- basis_period.historical_baseline의 DART/IR 문서는 전체 누적 공식 기준선입니다.
- basis_period.recent_signal_period의 official/news/securities_report 문서는 전체 누적 후보군 중 최신성/반복성/변화 신호를 분리해 해석하세요.
- 최신 DART/IR 기준선과 같은 방향이면 reinforcing_signal, 기준선에는 없지만 최근 새롭게 나타나면 new_or_watch_signal로 표시하세요.

2. Business Area Extraction
- 사업군은 selected_sector_config에서 가져오지 않습니다.
- 사업군은 source_intelligence의 business_area_candidates에서 추출합니다.
- 동일하거나 유사한 사업군 후보는 통합하세요.
- 통합할 때 반드시 근거를 남기세요.
- DART/IR/공식자료에 반복 등장하는 사업군은 evidence_strength를 높게 둡니다.
- 뉴스에서만 등장하는 사업군은 recent_signal 또는 weak로 처리합니다.
- core_business_areas에서 strong은 DART/IR/official 중 최소 하나 이상의 직접 근거가 있을 때만 사용하세요.
- news 단독 근거를 core_business_areas에 넣어야 한다면 evidence_strength는 medium 또는 weak로 제한하고, basis에 "뉴스 기반 최근 적용 사례이므로 장기 공식 사업군으로 단정하지 않음"을 남기세요.
- 증권 리포트 단독 근거는 external_analysis이며 core_business_areas strong 근거로 쓰지 마세요.

3. Sector Mapping
- 섹터는 selected_sector_config에서 가져온 분석 기준입니다.
- 각 사업군이 selected_sector_config의 어떤 섹터와 연결되는지 매핑하세요.
- 매핑 상태는 direct_match, partial_match, no_direct_sector_match 중 하나로 표시하세요.
- 섹터에 직접 매핑되지 않는 사업군도 회사 핵심 사업이면 삭제하지 마세요.
- 그런 사업군은 analysis_scope = profile_context_only로 표시하세요.
- 섹터와 연결되는 사업군은 analysis_scope = sector_analysis_scope로 표시하세요.
- 섹터 밖이지만 실적 영향, 대규모 수주, AI/클라우드/데이터/AX 결합 가능성이 확인되는 사업군은 analysis_scope = exception_watch로 표시할 수 있습니다.
- 단, exception_watch는 관찰 표시일 뿐 대응방향을 만들지 않습니다.
- sector_profile에는 mapping_status를 포함하세요.
- mapping_status 값은 direct_match|partial_match|no_match 중 하나입니다.
- 섹터와 직접 매칭되는 근거가 있으면 direct_match, 키워드/활동상 일부 연결되지만 공식 근거가 약하면 partial_match, 전혀 연결 근거가 없으면 no_match로 표시하세요.
- partial_match는 strong으로 두지 말고 evidence_strength=weak 또는 medium 이하로 두세요.

4. Metrics
- 수치가 있으면 metrics_profile에 모으세요.
- DART/IR/공식자료 수치는 official_metrics로 분류하세요.
- 증권 리포트 추정치는 estimated_metrics로 분류하세요.
- 뉴스 보도 수치는 reported_metrics로 분류하세요.
- SK AX 직접 수치인지 확인되지 않는 영업이익/매출/전망 수치는 estimated_metrics에 넣지 말고 uncertain_metrics에 넣으세요.
- SK그룹 전체, SK㈜, 지주, 다른 계열사 수치일 가능성이 있으면 uncertain_metrics로 분리하세요.
- 수치의 period, unit, source_title, caution을 포함하세요.
- 수치가 없으면 억지로 만들지 말고 빈 배열로 두세요.

5. Recent Activity
- 뉴스와 공식 뉴스룸의 최근 실행 신호는 recent_activity_profile에 넣으세요.
- 뉴스 기반 활동은 장기 전략으로 단정하지 마세요.
- cluster_id가 있으면 반드시 남기세요.

6. Observed Signal Patterns
- observed_signal_patterns에는 반복적으로 관찰되는 사업/역량/전략/산업 신호를 넣으세요.
- 대응방향이나 행동 제안을 쓰지 마세요.

7. Quality
- DART/IR/공식자료 근거가 충분하면 confidence를 높게 둘 수 있습니다.
- 뉴스 중심이면 confidence를 낮추거나 medium 이하로 두세요.
- 증권 리포트 중심이면 external_analysis 성격임을 quality.data_limitations에 남기세요.
- 데이터가 부족하면 uncertain_points에 남기세요.
- confidence는 raw_matched 전체 건수가 아니라 used_for_final_profile intelligence와 strongest_sources의 질로 판단하세요.
- DART/IR/official 기반으로 여러 사업군이 뒷받침되면 high 가능, official 1건 + news 1건 중심이면 medium 이하입니다.
- news 단독 recent_activity가 많아도 high로 두지 마세요.
- securities_report가 많아도 직접 귀속성이 약하면 confidence 상승에 사용하지 마세요.

## 공통 basis 객체

모든 basis 배열에는 가능한 한 아래 필드를 포함하세요.
- source_type: dart|ir|official|securities_report|news
- title: 근거 문서 제목
- article_id: raw_articles.id
- url: 원문 URL 또는 null
- evidence_text: 판단에 사용한 근거 요약
- why_used: 이 근거를 해당 사업군/역량/전략/산업군 판단에 사용한 이유

모든 metrics 배열에는 가능한 한 아래 필드를 포함하세요.
- metric_name: 매출액|수주액|성장률|고객 수|사업 비중 등
- value: 수치
- unit: 억원|%|건|명 등
- period: 기간
- metric_type: official_metric|estimated_metric|reported_metric
- source_type: dart|ir|official|securities_report|news
- source_title: 근거 문서 제목
- article_id: raw_articles.id
- caution: 공식 수치/추정치/보도 수치 구분과 해석 주의사항

## 입력
company_id:
{company_id}

company_name:
{company_name}

role:
{role}

company_config:
{company_config_json}

basis_period:
{basis_period_json}

source_intelligence:
{source_intelligence_json}

selected_sector_config:
{sector_config_json}

## 출력 JSON 형식
{
  "schema_version": "company_profile_v5",
  "company_id": "{company_id}",
  "company_name": "{company_name}",
  "role": "{role}",
  "generated_at": "{now}",
  "basis_period": {basis_period_json},

    "profile_summary": {
    "one_line": "회사의 현재 사업 방향을 한 문장으로 요약",
    "positioning": "이 회사가 IT서비스/AX 시장에서 어떤 포지션인지",
    "current_direction": "현재 어떤 방향으로 사업과 역량을 전개하는지",
    "baseline_vs_recent": "DART/IR 누적 기준선의 날짜 흐름과 최근 official/news/securities_report 신호가 목표 방향에 맞게 이어지는지",
    "confidence": "low|medium|high"
  },

  "historical_baseline_profile": {
    "repeated_business_areas": [],
    "repeated_capabilities": [],
    "repeated_strategic_keywords": [],
    "stable_target_industries": [],
    "baseline_summary": "DART/IR 전체 누적 기준으로 반복 확인되는 공식 사업 구조"
  },

  "business_profile": {
    "core_business_areas": [
      {
        "area": "자료에서 추출된 사업군명",
        "description": "회사가 해당 사업 영역에서 무엇을 하는지",
        "evidence_strength": "strong|medium|weak",
        "sector_mapping": {
          "matched_sector_ids": [],
          "mapping_status": "direct_match|partial_match|no_direct_sector_match",
          "mapping_reason": "selected_sector_config 기준으로 왜 이렇게 매핑했는지"
        },
        "analysis_scope": "sector_analysis_scope|profile_context_only|exception_watch",
        "basis": []
      }
    ],
    "capabilities": [
      {
        "keyword": "역량 키워드",
        "description": "이 역량이 어떤 서비스/운영/기술/고객군과 연결되는지",
        "evidence_strength": "strong|medium|weak",
        "basis": []
      }
    ],
    "strategic_keywords": [
      {
        "keyword": "전략 키워드",
        "meaning": "이 회사의 전략 방향에서 이 키워드가 의미하는 바",
        "evidence_strength": "strong|medium|weak",
        "basis": []
      }
    ]
  },

  "target_market_profile": {
    "target_industries": [
      {
        "industry": "산업군명",
        "reason": "이 산업군을 공략한다고 볼 수 있는 이유",
        "evidence_strength": "strong|medium|weak",
        "basis": [],
        "metrics": [],
        "confidence": "low|medium|high"
      }
    ],
    "target_customer_groups": [
      {
        "customer_group": "대기업|금융기관|공공기관|제조기업 등",
        "reason": "이 고객군과 연결되는 근거",
        "evidence_strength": "strong|medium|weak",
        "basis": []
      }
    ]
  },

  "sector_profile": [
    {
      "sector_id": "sectors.py의 sector_id",
      "sector_name": "sectors.py의 sector_name",
      "mapping_status": "direct_match|partial_match|no_match",
      "sector_definition_basis": {
        "keywords": ["sectors.py에서 가져온 키워드"],
        "aliases": ["sectors.py에서 가져온 alias"],
        "description": "sectors.py에서 가져온 섹터 설명"
      },
      "company_stance": "이 회사가 해당 섹터에서 어떤 사업/역량/전략 방향을 보이는지",
      "evidence_strength": "strong|medium|weak",
      "basis_by_source_type": {
        "dart_ir_basis": "DART/IR에서 확인되는 공식 근거",
        "official_basis": "공식 뉴스룸에서 확인되는 실행 근거",
        "news_basis": "뉴스 클러스터에서 확인되는 최근 활동 신호",
        "external_analysis_basis": "증권 리포트에서 확인되는 외부 해석"
      },
      "related_business_areas": ["사업군명"],
      "related_capabilities": ["역량 키워드"],
      "related_target_industries": ["산업군"],
      "metrics": [],
      "confidence": "low|medium|high",
      "caution": "근거가 약하거나 추가 확인이 필요한 내용"
    }
  ],

  "recent_activity_profile": {
    "summary": "official/news/securities_report에서 확인되는 최근 활동 흐름 요약",
    "key_activities": [
      {
        "activity": "최근 활동명",
        "source_type": "official|news|securities_report",
        "cluster_id": "cluster_id 또는 null",
        "related_keywords": [],
        "related_sectors": [],
        "related_industries": [],
        "interpretation_level": "official_execution|recent_signal|external_analysis",
        "evidence_strength": "strong|medium|weak",
        "basis": []
      }
    ],
    "caution": "뉴스 기반 활동은 최근 신호이며 장기 전략으로 단정하지 않음"
  },

  "change_signals": [
    {
      "signal": "과거 기준선 대비 최근 새롭게 강화되거나 등장한 신호",
      "change_type": "new|reinforcing|declining|stable|uncertain",
      "baseline_basis": "DART/IR 누적 기준선에서의 기존 상태",
      "recent_basis": "official/news/securities_report에서 확인된 최근 상태",
      "date_check": {
        "baseline_dates": [],
        "recent_signal_dates": [],
        "interpretation": "목표/전략 기준선 이후 실제 실행 신호가 이어지는지 날짜 흐름으로 판단"
      },
      "related_business_areas": [],
      "related_sectors": [],
      "evidence_strength": "strong|medium|weak",
      "confidence": "low|medium|high"
    }
  ],

  "observed_signal_patterns": [
    {
      "pattern": "공식자료와 최근 뉴스에서 반복적으로 관찰되는 사업/역량/전략/산업 신호",
      "related_keywords": [],
      "related_sectors": [],
      "related_industries": [],
      "source_types": ["dart", "ir", "official", "news", "securities_report"],
      "evidence_strength": "strong|medium|weak",
      "basis": []
    }
  ],

  "metrics_profile": {
    "official_metrics": [],
    "estimated_metrics": [],
    "reported_metrics": [],
    "uncertain_metrics": []
  },

  "evidence_map": {
    "source_counts": {
      "raw_matched": {"news": 0, "official": 0, "securities_report": 0, "dart": 0, "ir": 0},
      "analysis_units": {
        "news_clusters": 0,
        "official_documents": 0,
        "securities_report_documents": 0,
        "dart_documents": 0,
        "ir_documents": 0
      },
      "document_or_cluster_intelligence_created": {
        "news_cluster_intelligence": 0,
        "official_document_intelligence": 0,
        "securities_report_document_intelligence": 0,
        "dart_document_intelligence": 0,
        "ir_document_intelligence": 0
      },
      "used_for_final_profile": {
        "news_cluster_intelligence": 0,
        "official_document_intelligence": 0,
        "securities_report_document_intelligence": 0,
        "dart_document_intelligence": 0,
        "ir_document_intelligence": 0
      },
      "news_pipeline": {
        "raw_news_articles": 0,
        "relevance_pass_articles": 0,
        "clustered_news_articles": 0,
        "representative_news_clusters": 0,
        "unclustered_news_articles": 0
      },
      "by_processing_status": {
        "RAW": 0,
        "PROCESSED": 0,
        "SKIPPED": 0,
        "FAILED": 0,
        "OTHER": 0
      }
    },
    "strongest_sources": [
      {
        "source_type": "dart|ir|official|securities_report|news",
        "article_id": 0,
        "title": "근거 제목",
        "url": "원문 URL 또는 null",
        "why_strong": "왜 강한 근거인지"
      }
    ],
    "source_article_ids": [],
    "source_cluster_ids": []
  },

  "quality": {
    "confidence": "low|medium|high",
    "reason": "프로필 신뢰도 판단 이유",
    "uncertain_points": [],
    "data_limitations": []
  },

  "final_one_line": "이 회사는 현재 어떤 사업/역량/전략 방향으로 움직이는 회사로 볼 수 있는지 최종 한 줄 결론",
  "prompt_version": "company_profile_v5"
}
"""

COMPACT_COMPANY_PROFILE_PROMPT = """
당신은 국내 IT서비스/AX 시장을 분석하는 Competitive Intelligence 전략 분석가입니다.

입력은 raw 원문이 아니라 전체 후보 데이터를 문서/뉴스 클러스터 단위로 full-read 분석한 structured intelligence rollup입니다.
DART/IR은 historical baseline, official/news/securities_report는 최근 실행/외부 분석 신호로 해석하세요.

제약:
- JSON만 출력하세요.
- schema_version과 prompt_version은 company_profile_v5입니다.
- 시사점, 대응방향, 행동 제안은 작성하지 마세요.
- company_site는 사용하지 마세요.
- 사업군은 source_intelligence의 business_area/capability/strategy rollup, 전체 document_or_cluster_intelligence_manifest, evidence_rich_intelligence_focus 근거에서만 추출하세요.
- 뉴스 단독 근거는 장기 사업군 strong 근거로 쓰지 마세요.
- securities_report 수치는 직접 귀속성이 약하면 uncertain_metrics로 보내세요.
- "기업", "엔터프라이즈", "대기업"은 산업군이 아니라 고객군입니다.
- sector_profile은 selected_sector_config의 모든 섹터를 포함하세요.
- change_signals에는 date_check를 포함해 baseline 날짜와 recent signal 날짜 흐름을 비교하세요.
- news는 news_cluster_intelligence, DART는 dart_document_intelligence, IR은 ir_document_intelligence, official은 official_document_intelligence, securities_report는 securities_report_document_intelligence 기준으로 해석하세요.
- DART/IR/official/securities_report는 클러스터링하지 말고 문서 intelligence 단위로 판단하세요.

입력:
company_id: {company_id}
company_name: {company_name}
role: {role}
company_config: {company_config_json}
basis_period: {basis_period_json}
selected_sector_config: {sector_config_json}
source_intelligence_rollup: {source_intelligence_json}

반드시 아래 최상위 키를 포함한 JSON 객체로 출력하세요:
{
  "schema_version": "company_profile_v5",
  "company_id": "{company_id}",
  "company_name": "{company_name}",
  "role": "{role}",
  "generated_at": "{now}",
  "basis_period": {basis_period_json},
  "profile_summary": {
    "one_line": "",
    "positioning": "",
    "current_direction": "",
    "baseline_vs_recent": "",
    "confidence": "low|medium|high"
  },
  "historical_baseline_profile": {
    "repeated_business_areas": [],
    "repeated_capabilities": [],
    "repeated_strategic_keywords": [],
    "stable_target_industries": [],
    "baseline_summary": ""
  },
  "business_profile": {
    "core_business_areas": [
      {
        "area": "사업군명. 섹터명 자체 금지",
        "description": "회사가 이 사업에서 무엇을 하는지 1~2문장",
        "evidence_strength": "strong|medium|weak",
        "sector_mapping": {
          "matched_sector_ids": [],
          "mapping_status": "direct_match|partial_match|no_direct_sector_match",
          "mapping_reason": "sectors.py 기준 매핑 이유"
        },
        "analysis_scope": "sector_analysis_scope|profile_context_only|exception_watch",
        "basis": []
      }
    ],
    "capabilities": [
      {
        "keyword": "역량 키워드",
        "description": "어떤 서비스/운영/기술/고객군과 연결되는지",
        "evidence_strength": "strong|medium|weak",
        "basis": []
      }
    ],
    "strategic_keywords": [
      {
        "keyword": "전략 키워드",
        "meaning": "전략 방향에서 의미",
        "evidence_strength": "strong|medium|weak",
        "basis": []
      }
    ]
  },
  "target_market_profile": {
    "target_industries": [],
    "target_customer_groups": []
  },
  "sector_profile": [
    {
      "sector_id": "selected_sector_config의 sector_id",
      "sector_name": "selected_sector_config의 sector_name",
      "mapping_status": "direct_match|partial_match|no_match",
      "company_stance": "이 섹터에서 확인되는 회사의 사업/역량/전략 방향. 근거 없으면 명확히 없음",
      "evidence_strength": "strong|medium|weak",
      "basis_by_source_type": {
        "dart_ir_basis": null,
        "official_basis": null,
        "news_basis": null,
        "external_analysis_basis": null
      },
      "related_business_areas": [],
      "related_capabilities": [],
      "related_target_industries": [],
      "metrics": [],
      "confidence": "low|medium|high",
      "caution": ""
    }
  ],
  "recent_activity_profile": {
    "summary": "",
    "key_activities": [],
    "caution": "뉴스 기반 활동은 최근 신호이며 장기 전략으로 단정하지 않음"
  },
  "change_signals": [],
  "observed_signal_patterns": [],
  "metrics_profile": {
    "official_metrics": [],
    "estimated_metrics": [],
    "reported_metrics": [],
    "uncertain_metrics": []
  },
  "evidence_map": {
    "source_counts": {},
    "strongest_sources": [],
    "source_article_ids": [],
    "source_cluster_ids": []
  },
  "quality": {
    "confidence": "low|medium|high",
    "reason": "",
    "uncertain_points": [],
    "data_limitations": []
  },
  "final_one_line": "",
  "prompt_version": "company_profile_v5"
}

작성 규칙:
- core_business_areas는 반드시 객체 배열입니다. 문자열 배열 금지.
- capabilities와 strategic_keywords도 반드시 객체 배열입니다. 문자열 배열 금지.
- sector_profile은 반드시 selected_sector_config의 모든 섹터를 포함합니다.
- "AX", "보안", "인프라", "수주" 같은 섹터명 자체를 core_business_areas.area로 쓰지 마세요.
- sector_profile.mapping_status는 입력 unit_rollup/sector_unit_counts에 근거가 있으면 direct_match 또는 partial_match로 판단하세요. 모든 섹터를 no_match로 두지 마세요.
- profile_summary와 final_one_line은 빈 문자열 금지. 근거가 약하면 약하다고 쓰되 비워두지 마세요.
- strongest_sources는 object 배열입니다. "news" 같은 문자열만 넣지 마세요.
- strongest_sources는 evidence_rich_intelligence_focus 또는 document_or_cluster_intelligence_manifest의 article_id/title/evidence_text를 가진 실제 근거에서 고르세요.
- historical_baseline_profile의 repeated_* 항목은 문자열 배열 금지. 반드시 source_refs, evidence_summary, evidence_strength를 가진 object 배열로 쓰세요.
- target_industries에는 산업군만 넣으세요. 대기업/금융기관/공공기관/엔터프라이즈 고객/고객사는 target_customer_groups입니다.
- 금융기관은 target_industries="금융" + target_customer_groups="금융기관"으로 분리하세요.
- 공공기관은 target_industries="공공" + target_customer_groups="공공기관"으로 분리하세요.
- recent_activity_profile.key_activities는 source_type, evidence_strength, basis를 비워두지 마세요. news_cluster면 cluster_id를 넣으세요.
- metrics_profile의 metric_name에 "문서 내 수치"를 쓰지 마세요. metric_context, source_type, article_id, direct_company_match를 포함하세요.
- sector_profile direct_match/strong은 실제 basis_by_source_type 근거 요약이 있을 때만 사용하세요. 단순 id만 있으면 partial 또는 no_match입니다.
"""

_JSON_REPAIR_PROMPT = """\
아래 응답을 유효한 JSON 객체로만 고치세요.
스키마 키와 원래 의미를 유지하고, 설명 문장이나 markdown fence는 제거하세요.

오류:
{error}

원문:
{raw_text}
"""

_PROFILE_REPAIR_PROMPT = """\
아래 회사 프로필 JSON이 요구사항을 만족하도록 보완하세요.
- 반드시 JSON만 출력하세요.
- schema_version은 company_profile_v5입니다.
- profile_summary에는 baseline_vs_recent를 포함하세요.
- historical_baseline_profile과 change_signals를 포함하세요.
- change_signals 각 항목에는 date_check를 포함해 DART/IR 기준선 날짜와 recent signal 날짜 흐름을 비교하세요.
- sector_profile에는 selected_sector_config의 모든 sector_id가 포함되어야 합니다.
- sector_profile[].mapping_status는 direct_match|partial_match|no_match 중 하나입니다.
- 근거가 약한 섹터는 evidence_strength="weak", confidence="low"로 표시하세요.
- 시사점, 대응방향, 행동 제안, 카드뉴스 사용 규칙은 작성하지 마세요.
- card_usage_rules, response_rules, action_suggestions, news_signal_rules, opportunity_risk_profile 필드는 만들지 마세요.
- quality.data_limitations에 누락/한계를 기록하세요.

selected_sector_config:
{sector_config_json}

현재 JSON:
{profile_json}
"""

SOURCE_STAGE_SUMMARY_PROMPT = """
당신은 회사 프로필 생성을 위한 단계별 source intelligence 요약 Agent입니다.

입력은 raw 원문이 아니라 특정 source stage에 속한 document/cluster intelligence입니다.
이 단계에서는 최종 회사 프로필을 만들지 말고, 다음 final reducer가 사용할 stage summary JSON만 만드세요.

규칙:
- 반드시 JSON만 출력하세요.
- 시사점, 대응방향, 행동 제안은 작성하지 마세요.
- 입력된 stage_source_types 밖의 source를 추정하지 마세요.
- DART/IR은 historical baseline, official은 공식 실행 신호, news는 최근 클러스터 신호, securities_report는 외부 분석/전망으로 구분하세요.
- securities_report 직접 귀속성이 낮은 수치/해석은 uncertain 또는 external_analysis로 낮추세요.
- "기업", "대기업", "금융기관", "공공기관", "엔터프라이즈 고객"은 고객군입니다. 금융기관은 산업군 금융과 고객군 금융기관으로 분리하세요.
- 근거는 가능한 article_id/title/unit_id/cluster_id를 유지하세요.

입력:
company_id: {company_id}
company_name: {company_name}
role: {role}
stage_id: {stage_id}
stage_source_types: {stage_source_types_json}
selected_sector_config: {sector_config_json}
stage_intelligence: {stage_intelligence_json}

출력 JSON 형식:
{
  "stage_id": "{stage_id}",
  "source_types": {stage_source_types_json},
  "processed_intelligence_counts": {},
  "business_area_findings": [
    {
      "area": "사업군명",
      "evidence_summary": "근거 요약",
      "source_refs": [],
      "evidence_strength": "strong|medium|weak"
    }
  ],
  "capability_findings": [],
  "strategic_keyword_findings": [],
  "target_industry_findings": [],
  "target_customer_group_findings": [],
  "activity_findings": [
    {
      "activity": "실행 신호",
      "source_type": "official|news|securities_report|dart|ir",
      "cluster_id": null,
      "basis": []
    }
  ],
  "metric_findings": {
    "official_metrics": [],
    "estimated_metrics": [],
    "reported_metrics": [],
    "uncertain_metrics": []
  },
  "sector_findings": [
    {
      "sector_id": "sectors.py sector_id",
      "mapping_status": "direct_match|partial_match|no_match",
      "evidence_summary": "실제 근거 요약",
      "evidence_strength": "strong|medium|weak",
      "source_refs": []
    }
  ],
  "strongest_sources": [
    {
      "source_type": "dart|ir|official|news|securities_report",
      "article_id": 0,
      "title": "근거 제목",
      "why_strong": "왜 강한 근거인지"
    }
  ],
  "uncertain_points": [],
  "data_limitations": [],
  "confidence": "low|medium|high"
}
"""

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=0.2,
            max_completion_tokens=_LLM_MAX_COMPLETION_TOKENS,
        )
    return _llm


class ProfileAgent:
    """Build a profile JSON for SK AX or a peer company."""

    role: Literal["peer", "self"] | None = None

    def build_context(
        self,
        *,
        companies: list[str],
        sectors: list[str] | None = None,
        event_type: str | None = None,
        peer_profile_context: dict[str, Any] | None = None,
        skax_profile_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return lightweight ProfileContext for the analysis workflow.

        이 메서드는 DB schema나 저장 구조를 만들지 않는다. DataAnalysisSupervisorAgent가
        시사점 생성을 위해 필요한 SK AX / Peer / sector context를 런타임 dict로 묶는
        용도다. 무거운 회사별 프로필 생성은 ``build_profile``을 명시적으로 호출할 때만
        수행한다.
        """
        normalized_companies = [
            _normalize_company_id(company_id) for company_id in companies if company_id
        ]
        context_peer_ids = _profile_context_peer_ids(normalized_companies)
        peer_profiles = dict(peer_profile_context or {})
        for company_id in context_peer_ids:
            if company_id in SELF_COMPANY_IDS or company_id in peer_profiles:
                continue
            try:
                config = load_company_config(company_id)
            except Exception:
                continue
            peer_profiles[company_id] = {
                "peer_id": company_id,
                "company_name": config.get("company_name", company_id),
                "role": config.get("role"),
                "aliases": config.get("aliases", []),
                "event_type": event_type,
                "matched_sectors": sectors or [],
            }

        skax_profile = dict(skax_profile_context or {})
        sector_context: dict[str, Any] = {"selected_sector_ids": list(dict.fromkeys(sectors or []))}
        if not skax_profile:
            try:
                from src.services.skax_profile_context_loader import SKAXProfileLoader

                loaded = SKAXProfileLoader().load(sectors or [], use_db=False)
                skax_profile = loaded.get("skax_profile", loaded)
                sector_context.update(
                    {
                        "sector_config": loaded.get("sector_config", []),
                        "skax_contexts": loaded.get("skax_contexts", []),
                    }
                )
            except Exception as exc:
                log.warning("SK AX profile context 로드 실패 | error=%s", exc)
        if not skax_profile:
            try:
                skax_config = load_company_config("sk_ax")
            except Exception:
                skax_config = {"company_id": "sk_ax", "company_name": "SK AX", "role": "self"}
            skax_profile = {
                "company_id": "sk_ax",
                "company_name": skax_config.get("company_name", "SK AX"),
                "role": skax_config.get("role", "self"),
                "aliases": skax_config.get("aliases", []),
                "matched_sectors": sectors or [],
            }

        return {
            "skax_profile": skax_profile,
            "peer_profiles": peer_profiles,
            "profile_company_ids": ["sk_ax", *context_peer_ids],
            "sector_context": sector_context,
        }

    def build_profile(self, company_id: str, lookback_days: int | None = None) -> dict[str, Any]:
        company_id = _normalize_company_id(company_id)
        company_config = load_company_config(company_id)
        role = str(company_config["role"])
        if self.role is not None and role != self.role:
            raise ValueError(f"{type(self).__name__} cannot build role={role} profile")

        company_name = str(company_config["company_name"])
        sector_config = build_selected_sector_config()
        source_units, source_counts, retrieval_notes, basis_period = load_company_profile_documents(
            company_id=company_id,
            lookback_days=lookback_days,
            company_config=company_config,
            sector_config=sector_config,
        )

        source_intelligence = self.extract_source_intelligence(
            company_id=company_id,
            company_name=company_name,
            role=role,
            company_config=company_config,
            source_units=source_units,
            source_counts=source_counts,
            sector_config=sector_config,
            retrieval_notes=retrieval_notes,
        )
        return self.synthesize_profile(
            company_id=company_id,
            company_name=company_name,
            role=role,
            company_config=company_config,
            basis_period=basis_period,
            source_intelligence=source_intelligence,
            sector_config=sector_config,
        )

    def extract_source_intelligence(
        self,
        *,
        company_id: str,
        company_name: str,
        role: str,
        company_config: dict[str, Any],
        source_units: list[dict[str, Any]],
        source_counts: dict[str, Any],
        sector_config: list[dict[str, Any]],
        retrieval_notes: list[str],
    ) -> dict[str, Any]:
        return {
            "company_id": company_id,
            "company_name": company_name,
            "role": role,
            "source_intelligence_units": source_units,
            "source_counts": source_counts,
            "retrieval_notes": retrieval_notes,
        }

    def synthesize_profile(
        self,
        *,
        company_id: str,
        company_name: str,
        role: str,
        company_config: dict[str, Any],
        basis_period: dict[str, Any],
        source_intelligence: dict[str, Any],
        sector_config: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        source_counts = source_intelligence.get("source_counts", _empty_source_count_bundle())
        final_source_intelligence = _compact_source_intelligence_for_final_profile(
            source_intelligence,
            sector_config=sector_config,
        )
        stage_summaries = self.summarize_source_stages(
            company_id=company_id,
            company_name=company_name,
            role=role,
            source_intelligence=final_source_intelligence,
            sector_config=sector_config,
        )
        final_source_intelligence = _build_final_reducer_input(
            final_source_intelligence,
            stage_summaries=stage_summaries,
        )
        final_source_intelligence = _fit_final_source_intelligence(final_source_intelligence)
        prompt = (
            COMPACT_COMPANY_PROFILE_PROMPT.replace("{company_id}", company_id)
            .replace("{company_name}", company_name)
            .replace("{role}", role)
            .replace("{now}", now)
            .replace(
                "{company_config_json}",
                json.dumps(
                    _compact_company_config(company_config), ensure_ascii=False, default=str
                ),
            )
            .replace("{basis_period_json}", json.dumps(basis_period, ensure_ascii=False))
            .replace(
                "{source_intelligence_json}",
                json.dumps(final_source_intelligence, ensure_ascii=False, indent=2),
            )
            .replace(
                "{sector_config_json}",
                json.dumps(sector_config, ensure_ascii=False, indent=2),
            )
        )
        profile = _invoke_json_prompt(prompt, phase="company_profile")
        profile = _sanitize_company_profile(profile)
        profile = _inject_profile_diagnostics(
            profile=profile,
            source_counts=source_counts,
            retrieval_notes=_limitations_from_counts(source_counts, company_id=company_id),
            sector_config=sector_config,
            final_source_intelligence=final_source_intelligence,
        )
        profile["schema_version"] = _PROMPT_VERSION
        profile.setdefault("company_id", company_id)
        profile.setdefault("company_name", company_name)
        profile.setdefault("role", role)
        profile.setdefault("generated_at", now)
        profile.setdefault("basis_period", basis_period)
        profile["prompt_version"] = _PROMPT_VERSION
        profile["_source_intelligence"] = source_intelligence
        return profile

    def summarize_source_stages(
        self,
        *,
        company_id: str,
        company_name: str,
        role: str,
        source_intelligence: dict[str, Any],
        sector_config: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        stage_summaries: list[dict[str, Any]] = []
        for stage in _source_stage_plan():
            stage_input = _stage_intelligence_input(source_intelligence, stage)
            if not _stage_has_units(stage_input):
                continue
            prompt = (
                SOURCE_STAGE_SUMMARY_PROMPT.replace("{company_id}", company_id)
                .replace("{company_name}", company_name)
                .replace("{role}", role)
                .replace("{stage_id}", stage["stage_id"])
                .replace(
                    "{stage_source_types_json}",
                    json.dumps(stage["source_types"], ensure_ascii=False),
                )
                .replace(
                    "{sector_config_json}",
                    json.dumps(sector_config, ensure_ascii=False, indent=2),
                )
                .replace(
                    "{stage_intelligence_json}",
                    json.dumps(stage_input, ensure_ascii=False, indent=2, default=str),
                )
            )
            summary = _invoke_json_prompt(prompt, phase=f"source_stage_{stage['stage_id']}")
            summary.setdefault("stage_id", stage["stage_id"])
            summary.setdefault("source_types", stage["source_types"])
            stage_summaries.append(_sanitize_stage_summary(summary))
        return stage_summaries


class PeerProfileAgent(ProfileAgent):
    """Backward-compatible peer profile agent facade."""

    role: Literal["peer", "self"] | None = "peer"


class SKAXProfileAgent(ProfileAgent):
    """Backward-compatible SK AX profile agent facade."""

    role: Literal["peer", "self"] | None = "self"

    def build_profile(
        self,
        company_id: str = "sk_ax",
        lookback_days: int | None = None,
    ) -> dict[str, Any]:
        return super().build_profile(company_id=company_id, lookback_days=lookback_days)


def _profile_context_peer_ids(company_ids: list[str]) -> list[str]:
    """Return all domestic peer ids, with issue-related peers first."""
    issue_peer_ids = [
        company_id
        for company_id in company_ids
        if company_id in DOMESTIC_COMPANY_IDS and company_id not in SELF_COMPANY_IDS
    ]
    return list(dict.fromkeys([*issue_peer_ids, *sorted(DOMESTIC_COMPANY_IDS)]))


def build_selected_sector_config() -> list[dict[str, Any]]:
    """Convert src.config.sectors definitions into prompt-friendly JSON."""
    sectors: list[dict[str, Any]] = []
    for sector_id, info in SECTOR_KEYWORDS.items():
        sectors.append(
            {
                "sector_id": sector_id,
                "sector_name": info.get("name_ko", sector_id),
                "description": "",
                "keywords": list(info.get("keywords", [])),
                "aliases": [],
            }
        )
    return sectors


def load_company_config(company_id: str) -> dict[str, Any]:
    """Normalize company registry data from src.config.companies."""
    normalized = _normalize_company_id(company_id)
    raw_config = dict(COMPANIES.get(normalized, {}))
    return {
        "company_id": normalized,
        "company_name": company_name_ko(normalized),
        "role": _role_for_company(normalized),
        "aliases": company_aliases(normalized),
        "dart_corp_code": CORP_CODES.get(normalized),
        "naver_item_code": NAVER_ITEM_CODES.get(normalized),
        "raw_config": raw_config,
    }


def load_company_profile_documents(
    *,
    company_id: str,
    lookback_days: int | None,
    company_config: dict[str, Any] | None = None,
    sector_config: list[dict[str, Any]] | None = None,
    include_qdrant_chunks: bool | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str], dict[str, Any]]:
    end = datetime.now(timezone.utc)
    all_data_start = datetime(1970, 1, 1, tzinfo=timezone.utc)
    company_config = company_config or load_company_config(company_id)
    sector_config = sector_config or build_selected_sector_config()
    company_json = json.dumps([company_id], ensure_ascii=False)
    source_counts = _empty_source_count_bundle()
    source_units: list[dict[str, Any]] = []
    notes: list[str] = []
    qdrant_enabled = (
        include_qdrant_chunks
        if include_qdrant_chunks is not None
        else os.getenv("PROFILE_AGENT_ENABLE_QDRANT_CHUNKS", "false").lower() == "true"
    )

    with SessionLocal() as db:
        dart_rows = _fetch_rows_by_source(db=db, source_type="dart", company_json=company_json)
        ir_rows = _fetch_rows_by_source(db=db, source_type="ir", company_json=company_json)
        official_rows = _fetch_rows_by_source(
            db=db, source_type="official", company_json=company_json
        )
        securities_rows = _fetch_rows_by_source(
            db=db,
            source_type="securities_report",
            company_json=company_json,
        )
        business_signal_rows = _fetch_business_signal_rows(
            db=db,
            company_id=company_id,
            lookback_days=lookback_days or _DEFAULT_RECENCY_FOCUS_DAYS,
        )
        financial_metric_rows = _fetch_financial_metric_rows(db=db, company_id=company_id)
        representative_news_rows = _fetch_representative_news_rows(db=db, company_json=company_json)
        news_cluster_articles = _fetch_news_cluster_articles(
            db=db,
            company_json=company_json,
            cluster_ids=[
                row._mapping.get("cluster_id")
                for row in representative_news_rows
                if row._mapping.get("cluster_id") is not None
            ],
        )

        historical_documents = [
            _row_to_document(row._mapping, company_config, role_hint="historical_baseline")
            for row in [*dart_rows, *ir_rows]
        ]
        historical_period = _historical_baseline_period(historical_documents)
        fallback_policy = {"used": False, "reason": ""}

        raw_matched = collect_source_counts(
            company_id,
            all_data_start,
            db=db,
            source_types=_RECENT_SIGNAL_SOURCE_TYPES,
        )
        historical_raw_matched = collect_source_counts(
            company_id,
            all_data_start,
            db=db,
            source_types=_BASELINE_SOURCE_TYPES,
        )
        for source_type in _BASELINE_SOURCE_TYPES:
            raw_matched[source_type] = historical_raw_matched.get(source_type, 0)
        source_counts["raw_matched"].update(raw_matched)
        source_counts["news_pipeline"].update(
            collect_news_pipeline_counts(company_id, all_data_start, db=db)
        )
        source_counts["by_processing_status"].update(
            collect_processing_status_counts(company_id, all_data_start, db=db)
        )
        recent_signal_period = collect_source_period(
            company_id,
            all_data_start,
            db=db,
            source_types=_RECENT_SIGNAL_SOURCE_TYPES,
        )
        recency_focus = _derive_recency_focus(
            requested_days=lookback_days,
            historical_period=historical_period,
            recent_signal_period=recent_signal_period,
            end=end,
        )

        source_units.extend(
            _document_unit_from_row(
                row._mapping,
                company_config=company_config,
                sector_config=sector_config,
                unit_type="official_document",
            )
            for row in [*dart_rows, *ir_rows, *official_rows]
        )
        source_units.extend(
            _document_unit_from_row(
                row._mapping,
                company_config=company_config,
                sector_config=sector_config,
                unit_type="securities_report_document",
            )
            for row in securities_rows
        )
        source_units.extend(
            _news_cluster_unit_from_rows(
                rep_row._mapping,
                news_cluster_articles.get(str(rep_row._mapping.get("cluster_id")), []),
                company_config=company_config,
                sector_config=sector_config,
            )
            for rep_row in representative_news_rows
        )
        source_units.extend(
            _business_signal_unit_from_row(row._mapping, company_config=company_config)
            for row in business_signal_rows
        )
        source_units.extend(
            _financial_metric_unit_from_row(row._mapping, company_config=company_config)
            for row in financial_metric_rows
        )

        _populate_analysis_unit_counts(
            source_counts=source_counts,
            dart_count=len(dart_rows),
            ir_count=len(ir_rows),
            official_count=len(official_rows),
            securities_count=len(securities_rows),
            news_cluster_count=len(representative_news_rows),
            business_signal_count=len(business_signal_rows),
            financial_metric_count=len(financial_metric_rows),
            qdrant_chunk_count=0,
        )
        source_counts["structured_evidence"].update(
            {
                "business_signals": len(business_signal_rows),
                "financial_metrics": len(financial_metric_rows),
                "qdrant_chunks": 0,
            }
        )

    qdrant_units = _load_qdrant_document_chunk_units(
        company_id=company_id,
        company_config=company_config,
        enabled=bool(qdrant_enabled),
    )
    if qdrant_enabled and not qdrant_units:
        notes.append("Qdrant DART/IR chunk 근거 없음 또는 조회 실패")
    if qdrant_units:
        source_units.extend(qdrant_units)
        qdrant_count = len(qdrant_units)
        source_counts["structured_evidence"]["qdrant_chunks"] = qdrant_count
        source_counts["analysis_units"]["qdrant_document_chunks"] = qdrant_count
        source_counts["document_or_cluster_intelligence_created"][
            "qdrant_document_chunk_intelligence"
        ] = qdrant_count
        source_counts["used_for_final_profile"]["qdrant_document_chunk_intelligence"] = qdrant_count

    if not source_units:
        notes.append("조회 기간 내 raw_articles 근거 문서 없음")
    notes.extend(_limitations_from_counts(source_counts, company_id=company_id))
    basis_period = _basis_period(
        historical_period=historical_period,
        recent_signal_period=recent_signal_period,
        recency_focus=recency_focus,
        end=end,
        fallback_policy=fallback_policy,
    )
    return source_units, source_counts, notes, basis_period


def _fetch_rows_by_source(*, db: Any, source_type: str, company_json: str) -> list[Any]:
    return db.execute(
        text("""
            SELECT id, source_type, source_name, publisher, title, content, url,
                   published_at, collected_at, company, metadata, cluster_id,
                   crawl_status, processing_status, is_representative
            FROM raw_articles
            WHERE source_type = :source_type
              AND company @> CAST(:company_json AS jsonb)
              AND COALESCE(crawl_status, 'success') <> 'failed'
            ORDER BY
              COALESCE(published_at, collected_at) DESC NULLS LAST,
              collected_at DESC NULLS LAST,
              id DESC
        """),
        {"source_type": source_type, "company_json": company_json},
    ).fetchall()


def _fetch_representative_news_rows(*, db: Any, company_json: str) -> list[Any]:
    return db.execute(
        text("""
            SELECT id, source_type, source_name, publisher, title, content, url,
                   published_at, collected_at, company, metadata, cluster_id,
                   crawl_status, processing_status, is_representative
            FROM raw_articles
            WHERE source_type = 'news'
              AND company @> CAST(:company_json AS jsonb)
              AND cluster_id IS NOT NULL
              AND is_representative = true
              AND COALESCE(crawl_status, 'success') <> 'failed'
            ORDER BY
              COALESCE(published_at, collected_at) DESC NULLS LAST,
              collected_at DESC NULLS LAST,
              id DESC
        """),
        {"company_json": company_json},
    ).fetchall()


def _fetch_news_cluster_articles(
    *,
    db: Any,
    company_json: str,
    cluster_ids: list[Any],
) -> dict[str, list[dict[str, Any]]]:
    if not cluster_ids:
        return {}
    rows = db.execute(
        text("""
            SELECT id, source_type, source_name, publisher, title, content, url,
                   published_at, collected_at, company, metadata, cluster_id,
                   crawl_status, processing_status, is_representative
            FROM raw_articles
            WHERE source_type = 'news'
              AND company @> CAST(:company_json AS jsonb)
              AND cluster_id = ANY(:cluster_ids)
              AND COALESCE(crawl_status, 'success') <> 'failed'
            ORDER BY
              cluster_id,
              is_representative DESC NULLS LAST,
              COALESCE(published_at, collected_at) DESC NULLS LAST,
              id DESC
        """),
        {"company_json": company_json, "cluster_ids": cluster_ids},
    ).fetchall()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row_map = dict(row._mapping)
        grouped.setdefault(str(row_map.get("cluster_id")), []).append(row_map)
    return grouped


def _fetch_business_signal_rows(
    *,
    db: Any,
    company_id: str,
    lookback_days: int,
    limit: int = 80,
) -> list[Any]:
    aliases = expand_peer_aliases(company_id)
    return db.execute(
        text("""
            SELECT id, peer_id, business_area, signal_type, sentiment, summary,
                   evidence_text, confidence, period_year, period_quarter,
                   raw_article_id, created_at
              FROM raw_article_business_signals
             WHERE peer_id = ANY(:aliases)
               AND created_at >= NOW() - (:days || ' days')::interval
             ORDER BY confidence DESC NULLS LAST,
                      period_year DESC NULLS LAST,
                      period_quarter DESC NULLS LAST,
                      created_at DESC NULLS LAST
             LIMIT :limit
        """),
        {"aliases": aliases, "days": int(lookback_days), "limit": int(limit)},
    ).fetchall()


def _fetch_financial_metric_rows(
    *,
    db: Any,
    company_id: str,
    limit: int = 48,
) -> list[Any]:
    aliases = expand_peer_aliases(company_id)
    return db.execute(
        text("""
            SELECT id, peer_id, metric_name, metric_label, business_area,
                   value_numeric, value_krwbn, unit, currency, period_year,
                   period_quarter, period, confidence, raw_article_id, created_at
              FROM raw_article_financial_metrics
             WHERE peer_id = ANY(:aliases)
               AND period_year IS NOT NULL
             ORDER BY period_year DESC NULLS LAST,
                      period_quarter DESC NULLS LAST,
                      confidence DESC NULLS LAST,
                      created_at DESC NULLS LAST
             LIMIT :limit
        """),
        {"aliases": aliases, "limit": int(limit)},
    ).fetchall()


def _historical_baseline_period(baseline_documents: list[dict[str, Any]]) -> dict[str, Any]:
    dates_by_source: dict[str, list[datetime]] = {
        source_type: [] for source_type in _BASELINE_SOURCE_TYPES
    }
    for doc in baseline_documents:
        source_type = str(doc.get("source_type") or "")
        parsed = _parse_iso_datetime(str(doc.get("document_date") or ""))
        if source_type in dates_by_source and parsed is not None:
            dates_by_source[source_type].append(parsed)

    latest_by_source = {
        source_type: max(values).date().isoformat()
        for source_type, values in dates_by_source.items()
        if values
    }
    baseline_document_ids = [
        int(doc["article_id"]) for doc in baseline_documents if doc.get("article_id") is not None
    ]
    all_document_dates = [value for values in dates_by_source.values() for value in values]
    if not all_document_dates:
        return {
            "source_types": list(_BASELINE_SOURCE_TYPES),
            "start_date": None,
            "end_date": None,
            "latest_dates_by_source": latest_by_source,
            "document_ids": baseline_document_ids,
            "purpose": "장기 사업군/공식 사업 구조/반복 키워드 기준선",
        }

    return {
        "source_types": list(_BASELINE_SOURCE_TYPES),
        "start_date": min(all_document_dates).date().isoformat(),
        "end_date": max(all_document_dates).date().isoformat(),
        "latest_dates_by_source": latest_by_source,
        "document_ids": baseline_document_ids,
        "purpose": "장기 사업군/공식 사업 구조/반복 키워드 기준선",
    }


def collect_source_counts(
    company_id: str,
    start_date: datetime,
    *,
    db: Any,
    source_types: tuple[str, ...] = _SOURCE_TYPES,
) -> dict[str, int]:
    company_json = json.dumps([company_id], ensure_ascii=False)
    counts = _empty_source_counts()
    rows = db.execute(
        text("""
            SELECT source_type, COUNT(*) AS count
            FROM raw_articles
            WHERE source_type = ANY(:source_types)
              AND company @> CAST(:company_json AS jsonb)
              AND COALESCE(published_at, collected_at) >= :start_date
              AND COALESCE(crawl_status, 'success') <> 'failed'
            GROUP BY source_type
        """),
        {
            "source_types": list(source_types),
            "company_json": company_json,
            "start_date": start_date,
        },
    ).fetchall()
    for row in rows:
        counts[str(row.source_type)] = int(row.count or 0)
    return counts


def collect_news_pipeline_counts(
    company_id: str,
    start_date: datetime,
    *,
    db: Any,
) -> dict[str, int]:
    company_json = json.dumps([company_id], ensure_ascii=False)
    row = db.execute(
        text("""
            SELECT
                COUNT(*) AS raw_news_articles,
                COUNT(*) FILTER (WHERE relevance_label = 'relevant') AS relevance_pass_articles,
                COUNT(*) FILTER (WHERE cluster_id IS NOT NULL) AS clustered_news_articles,
                COUNT(*) FILTER (
                    WHERE cluster_id IS NOT NULL AND is_representative = true
                ) AS representative_news_clusters,
                COUNT(*) FILTER (WHERE cluster_id IS NULL) AS unclustered_news_articles
            FROM raw_articles
            WHERE source_type = 'news'
              AND company @> CAST(:company_json AS jsonb)
              AND COALESCE(published_at, collected_at) >= :start_date
              AND COALESCE(crawl_status, 'success') <> 'failed'
        """),
        {"company_json": company_json, "start_date": start_date},
    ).one()
    return {
        "raw_news_articles": int(row.raw_news_articles or 0),
        "relevance_pass_articles": int(row.relevance_pass_articles or 0),
        "clustered_news_articles": int(row.clustered_news_articles or 0),
        "representative_news_clusters": int(row.representative_news_clusters or 0),
        "unclustered_news_articles": int(row.unclustered_news_articles or 0),
    }


def collect_processing_status_counts(
    company_id: str,
    start_date: datetime,
    *,
    db: Any,
) -> dict[str, int]:
    company_json = json.dumps([company_id], ensure_ascii=False)
    rows = db.execute(
        text("""
            SELECT
                CASE
                    WHEN processing_status IN ('RAW', 'PROCESSED', 'REVIEW', 'SKIPPED', 'FAILED')
                        THEN processing_status
                    ELSE 'OTHER'
                END AS status_bucket,
                COUNT(*) AS count
            FROM raw_articles
            WHERE source_type = 'news'
              AND company @> CAST(:company_json AS jsonb)
              AND COALESCE(published_at, collected_at) >= :start_date
              AND COALESCE(crawl_status, 'success') <> 'failed'
            GROUP BY status_bucket
        """),
        {"company_json": company_json, "start_date": start_date},
    ).fetchall()
    counts = _empty_processing_status_counts()
    for row in rows:
        counts[str(row.status_bucket)] = int(row.count or 0)
    return counts


def collect_source_period(
    company_id: str,
    start_date: datetime,
    *,
    db: Any,
    source_types: tuple[str, ...],
) -> dict[str, Any]:
    company_json = json.dumps([company_id], ensure_ascii=False)
    row = db.execute(
        text("""
            SELECT
                MIN(COALESCE(published_at, collected_at)) AS start_date,
                MAX(COALESCE(published_at, collected_at)) AS end_date
            FROM raw_articles
            WHERE source_type = ANY(:source_types)
              AND company @> CAST(:company_json AS jsonb)
              AND COALESCE(published_at, collected_at) >= :start_date
              AND COALESCE(crawl_status, 'success') <> 'failed'
        """),
        {
            "source_types": list(source_types),
            "company_json": company_json,
            "start_date": start_date,
        },
    ).one()
    start_value = row.start_date if isinstance(row.start_date, datetime) else None
    end_value = row.end_date if isinstance(row.end_date, datetime) else None
    return {
        "start_date": _ensure_aware(start_value).date().isoformat() if start_value else None,
        "end_date": _ensure_aware(end_value).date().isoformat() if end_value else None,
    }


def _document_unit_from_row(
    row: Any,
    *,
    company_config: dict[str, Any],
    sector_config: list[dict[str, Any]],
    unit_type: str,
) -> dict[str, Any]:
    source_type = str(row.get("source_type") or "")
    article_id = int(row["id"])
    title = str(row.get("title") or "")
    content = str(row.get("content") or "")
    compact = _compact_text(content, limit=_UNIT_EVIDENCE_LIMIT)
    sector_signals = _sector_signals_for_text(f"{title} {content}", sector_config)
    direct_match = _direct_company_match(row, company_config)
    return {
        "unit_id": f"{source_type}:{article_id}",
        "unit_type": unit_type,
        "source_type": source_type,
        "company_id": company_config.get("company_id"),
        "title": title,
        "article_ids": [article_id],
        "cluster_id": None,
        "source_count": 1,
        "published_at": _iso_or_empty(row.get("published_at")),
        "collected_at": _iso_or_empty(row.get("collected_at")),
        "latest_collected_at": _iso_or_empty(row.get("collected_at")),
        "direct_company_match": direct_match,
        "business_area_candidates": _business_area_candidates(title),
        "capability_keywords": _matched_keywords_from_sector_signals(sector_signals),
        "strategic_keywords": _strategic_keywords_for_text(f"{title} {content}"),
        "target_industries": [],
        "target_customer_groups": _target_customer_groups_for_text(f"{title} {content}"),
        "sector_signals": sector_signals,
        "quantitative_signals": _quantitative_signals_for_text(
            f"{title} {content}",
            source_type=source_type,
            source_title=title,
            direct_company_match=direct_match,
            article_id=article_id,
        ),
        "activity_signals": _activity_signals_for_text(title, source_type=source_type),
        "evidence_texts": [{"text": compact, "why_used": "문서 intelligence 생성을 위한 핵심 근거"}]
        if compact
        else [],
        "confidence": _unit_confidence(source_type, direct_match),
    }


def _business_signal_unit_from_row(
    row: Any,
    *,
    company_config: dict[str, Any],
) -> dict[str, Any]:
    signal_id = row.get("id")
    business_area = str(row.get("business_area") or "").strip()
    signal_type = str(row.get("signal_type") or "").strip()
    summary = str(row.get("summary") or "").strip()
    evidence_text = str(row.get("evidence_text") or "").strip()
    title = " / ".join(item for item in [business_area, signal_type] if item) or "business signal"
    article_id = row.get("raw_article_id")
    article_ids = [int(article_id)] if isinstance(article_id, int) else []
    return {
        "unit_id": f"business_signal:{signal_id}",
        "unit_type": "business_signal",
        "source_type": "business_signal",
        "company_id": company_config.get("company_id"),
        "title": title,
        "article_ids": article_ids,
        "cluster_id": None,
        "source_count": 1,
        "published_at": "",
        "collected_at": _iso_or_empty(row.get("created_at")),
        "latest_collected_at": _iso_or_empty(row.get("created_at")),
        "direct_company_match": {
            "label_match": True,
            "alias_match": False,
            "corp_code_match": False,
            "match_confidence": "high",
            "match_reason": f"raw_article_business_signals.peer_id={row.get('peer_id')}",
        },
        "business_area_candidates": [business_area] if business_area else [],
        "capability_keywords": [business_area] if business_area else [],
        "strategic_keywords": [signal_type] if signal_type else [],
        "target_industries": [],
        "target_customer_groups": [],
        "sector_signals": [],
        "quantitative_signals": [],
        "activity_signals": [signal_type] if signal_type else [],
        "evidence_texts": [
            {
                "text": _compact_text(evidence_text or summary, limit=_UNIT_EVIDENCE_LIMIT),
                "why_used": "정규화된 business signal 근거",
            }
        ]
        if (evidence_text or summary)
        else [],
        "confidence": _confidence_band(row.get("confidence")),
        "structured_payload": {
            "signal_id": signal_id,
            "peer_id": row.get("peer_id"),
            "business_area": row.get("business_area"),
            "signal_type": row.get("signal_type"),
            "sentiment": row.get("sentiment"),
            "confidence": _safe_float_or_none(row.get("confidence")),
            "period_year": row.get("period_year"),
            "period_quarter": row.get("period_quarter"),
            "raw_article_id": row.get("raw_article_id"),
        },
    }


def _financial_metric_unit_from_row(
    row: Any,
    *,
    company_config: dict[str, Any],
) -> dict[str, Any]:
    metric_id = row.get("id")
    metric_name_raw = str(row.get("metric_name") or "")
    metric_name = METRIC_CANONICAL.get(metric_name_raw, metric_name_raw)
    metric_label = row.get("metric_label") or metric_name
    period = row.get("period") or _metric_period(row)
    article_id = row.get("raw_article_id")
    article_ids = [int(article_id)] if isinstance(article_id, int) else []
    metric_signal = {
        "metric_name": metric_name,
        "metric_name_raw": metric_name_raw,
        "metric_label": metric_label,
        "metric_type": "official_metric",
        "value_numeric": _safe_float_or_none(row.get("value_numeric")),
        "value_krwbn": _safe_float_or_none(row.get("value_krwbn")),
        "unit": row.get("unit"),
        "currency": row.get("currency"),
        "period": period,
        "source_type": "financial_metric",
        "article_id": row.get("raw_article_id"),
        "confidence": _safe_float_or_none(row.get("confidence")),
        "caution": "raw_article_financial_metrics 정규화 지표",
    }
    return {
        "unit_id": f"financial_metric:{metric_id}",
        "unit_type": "financial_metric",
        "source_type": "financial_metric",
        "company_id": company_config.get("company_id"),
        "title": f"{metric_label} {period}".strip(),
        "article_ids": article_ids,
        "cluster_id": None,
        "source_count": 1,
        "published_at": "",
        "collected_at": _iso_or_empty(row.get("created_at")),
        "latest_collected_at": _iso_or_empty(row.get("created_at")),
        "direct_company_match": {
            "label_match": True,
            "alias_match": False,
            "corp_code_match": False,
            "match_confidence": "high",
            "match_reason": f"raw_article_financial_metrics.peer_id={row.get('peer_id')}",
        },
        "business_area_candidates": [str(row.get("business_area"))]
        if row.get("business_area")
        else [],
        "capability_keywords": [],
        "strategic_keywords": [],
        "target_industries": [],
        "target_customer_groups": [],
        "sector_signals": [],
        "quantitative_signals": [metric_signal],
        "activity_signals": [],
        "evidence_texts": [
            {
                "text": _metric_evidence_text(metric_signal),
                "why_used": "정규화된 financial metric 근거",
            }
        ],
        "confidence": _confidence_band(row.get("confidence")),
        "structured_payload": {
            "financial_metric_id": metric_id,
            "peer_id": row.get("peer_id"),
            "metric_name": metric_name,
            "metric_name_raw": metric_name_raw,
            "period_year": row.get("period_year"),
            "period_quarter": row.get("period_quarter"),
            "raw_article_id": row.get("raw_article_id"),
        },
    }


def _load_qdrant_document_chunk_units(
    *,
    company_id: str,
    company_config: dict[str, Any],
    enabled: bool,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    if not enabled:
        return []
    try:
        from src.rag.document_index import search_dart_chunks

        chunks = search_dart_chunks(
            "사업의 내용 주요 서비스 신규 사업 연구개발 리스크",
            top_k=top_k,
            peer_id=company_id,
        )
    except Exception as exc:  # noqa: BLE001 — optional retrieval.
        log.debug("profile qdrant chunk retrieval failed | company=%s error=%s", company_id, exc)
        return []
    return [
        _qdrant_chunk_unit_from_payload(chunk, company_config=company_config)
        for chunk in chunks
        if isinstance(chunk, dict)
    ]


def _qdrant_chunk_unit_from_payload(
    payload: dict[str, Any],
    *,
    company_config: dict[str, Any],
) -> dict[str, Any]:
    source_type = str(payload.get("source_type") or "document_chunk")
    point_id = str(payload.get("point_id") or payload.get("chunk_id") or "")
    article_id = payload.get("raw_article_id")
    article_ids = [int(article_id)] if isinstance(article_id, int) else []
    text_value = str(payload.get("text") or "")
    return {
        "unit_id": f"qdrant_chunk:{point_id}",
        "unit_type": "qdrant_document_chunk",
        "source_type": f"qdrant_{source_type}",
        "company_id": company_config.get("company_id"),
        "title": str(payload.get("section_title") or payload.get("report_name") or "document chunk"),
        "article_ids": article_ids,
        "cluster_id": None,
        "source_count": 1,
        "published_at": str(payload.get("published_at") or ""),
        "collected_at": "",
        "latest_collected_at": "",
        "direct_company_match": {
            "label_match": True,
            "alias_match": False,
            "corp_code_match": False,
            "match_confidence": "high",
            "match_reason": f"qdrant peer_id={payload.get('peer_id')}",
        },
        "business_area_candidates": [],
        "capability_keywords": [
            str(item)
            for item in (payload.get("matched_keywords") or [])
            if str(item).strip()
        ],
        "strategic_keywords": [],
        "target_industries": [],
        "target_customer_groups": [],
        "sector_signals": [],
        "quantitative_signals": [],
        "activity_signals": [],
        "evidence_texts": [
            {
                "text": _compact_text(text_value, limit=_UNIT_EVIDENCE_LIMIT),
                "why_used": "Qdrant DART/IR 본문 chunk 근거",
            }
        ]
        if text_value
        else [],
        "confidence": "high",
        "structured_payload": {
            "point_id": point_id,
            "raw_article_id": payload.get("raw_article_id"),
            "section_key": payload.get("section_key"),
            "section_title": payload.get("section_title"),
            "period": payload.get("period"),
            "score": _safe_float_or_none(payload.get("score")),
        },
    }


def _compact_source_intelligence_for_final_profile(
    source_intelligence: dict[str, Any],
    *,
    sector_config: list[dict[str, Any]],
) -> dict[str, Any]:
    units = source_intelligence.get("source_intelligence_units", [])
    if not isinstance(units, list):
        units = []

    source_counter: Counter[str] = Counter()
    unit_counter: Counter[str] = Counter()
    direct_counter: Counter[str] = Counter()
    sector_counter: Counter[str] = Counter()
    mapping_counter: Counter[str] = Counter()
    business_counter: Counter[str] = Counter()
    capability_counter: Counter[str] = Counter()
    strategy_counter: Counter[str] = Counter()
    industry_counter: Counter[str] = Counter()
    customer_counter: Counter[str] = Counter()
    activity_counter: Counter[str] = Counter()
    metric_counter: Counter[str] = Counter()
    document_or_cluster_manifest: list[dict[str, Any]] = []
    evidence_rich_intelligence_focus: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    source_article_ids: list[int] = []
    source_cluster_ids: list[str] = []

    for unit in units:
        if not isinstance(unit, dict):
            continue
        source_type = str(unit.get("source_type") or "unknown")
        unit_type = str(unit.get("unit_type") or "unknown")
        source_counter[source_type] += 1
        unit_counter[unit_type] += 1
        direct_match = unit.get("direct_company_match", {})
        if isinstance(direct_match, dict):
            direct_counter[str(direct_match.get("match_confidence") or "unknown")] += 1
        _update_counter(business_counter, unit.get("business_area_candidates"))
        _update_counter(capability_counter, unit.get("capability_keywords"))
        _update_counter(strategy_counter, unit.get("strategic_keywords"))
        _update_counter(industry_counter, unit.get("target_industries"))
        _update_counter(customer_counter, unit.get("target_customer_groups"))
        _update_counter(activity_counter, unit.get("activity_signals"))
        for signal in unit.get("sector_signals", []) or []:
            if not isinstance(signal, dict):
                continue
            sector_id = signal.get("sector_id")
            mapping_status = str(signal.get("mapping_status") or "unknown")
            if sector_id:
                sector_counter[str(sector_id)] += 1
            mapping_counter[mapping_status] += 1
            _update_counter(capability_counter, signal.get("matched_keywords"))
        for metric in unit.get("quantitative_signals", []) or []:
            if not isinstance(metric, dict):
                continue
            metric_type = str(metric.get("metric_type") or "uncertain_metric")
            metric_counter[metric_type] += 1
            if len(metrics) < 35:
                metrics.append(_compact_metric(metric, unit))
        for article_id in unit.get("article_ids", []) or []:
            if isinstance(article_id, int):
                source_article_ids.append(article_id)
        if unit.get("cluster_id") is not None:
            source_cluster_ids.append(str(unit.get("cluster_id")))

        document_or_cluster_manifest.append(_unit_manifest_for_final(unit))
        evidence_rich_intelligence_focus.append(_compact_unit_intelligence_for_final(unit))

    evidence_rich_intelligence_focus.sort(
        key=lambda item: (
            _source_priority(str(item.get("source_type") or "")),
            str(item.get("latest_collected_at") or ""),
        ),
        reverse=False,
    )

    sector_summary = []
    sector_lookup = {str(item.get("sector_id")): item for item in sector_config}
    for sector_id, count in sector_counter.most_common():
        sector = sector_lookup.get(sector_id, {})
        sector_summary.append(
            {
                "sector_id": sector_id,
                "sector_name": sector.get("sector_name", sector_id),
                "unit_count": count,
            }
        )

    return {
        "company_id": source_intelligence.get("company_id"),
        "company_name": source_intelligence.get("company_name"),
        "role": source_intelligence.get("role"),
        "source_counts": source_intelligence.get("source_counts", _empty_source_count_bundle()),
        "retrieval_notes": source_intelligence.get("retrieval_notes", []),
        "unit_rollup": {
            "total_units": sum(unit_counter.values()),
            "by_source_type": dict(source_counter),
            "by_unit_type": dict(unit_counter),
            "direct_company_match_confidence": dict(direct_counter),
            "sector_unit_counts": sector_summary,
            "sector_mapping_status_counts": dict(mapping_counter),
            "metric_type_counts": dict(metric_counter),
        },
        "keyword_rollup": {
            "business_area_candidates": _counter_to_items(business_counter, limit=40),
            "capability_keywords": _counter_to_items(capability_counter, limit=50),
            "strategic_keywords": _counter_to_items(strategy_counter, limit=40),
            "target_industries": _counter_to_items(industry_counter, limit=40),
            "target_customer_groups": _counter_to_items(customer_counter, limit=40),
            "activity_signals": _counter_to_items(activity_counter, limit=40),
        },
        "metrics_rollup": metrics,
        "document_or_cluster_intelligence_manifest": document_or_cluster_manifest,
        "evidence_rich_intelligence_focus": evidence_rich_intelligence_focus,
        "evidence_index": {
            "source_article_count": len(set(source_article_ids)),
            "source_cluster_count": len(set(source_cluster_ids)),
            "sample_article_ids": sorted(set(source_article_ids))[:120],
            "sample_cluster_ids": sorted(set(source_cluster_ids))[:80],
        },
    }


def _compact_company_config(company_config: dict[str, Any]) -> dict[str, Any]:
    aliases = company_config.get("aliases", [])
    return {
        "company_id": company_config.get("company_id"),
        "company_name": company_config.get("company_name"),
        "role": company_config.get("role"),
        "aliases": aliases[:20] if isinstance(aliases, list) else [],
        "alias_count": len(aliases) if isinstance(aliases, list) else 0,
        "dart_corp_code": company_config.get("dart_corp_code"),
        "naver_item_code": company_config.get("naver_item_code"),
    }


def _source_stage_plan() -> list[dict[str, Any]]:
    return [
        {"stage_id": "historical_baseline_dart_ir", "source_types": ["dart", "ir"]},
        {"stage_id": "official_execution", "source_types": ["official"]},
        {"stage_id": "news_cluster_recent_signal", "source_types": ["news"]},
        {"stage_id": "securities_report_external_analysis", "source_types": ["securities_report"]},
    ]


def _stage_intelligence_input(
    source_intelligence: dict[str, Any],
    stage: dict[str, Any],
) -> dict[str, Any]:
    source_types = {str(source_type) for source_type in stage.get("source_types", [])}
    manifest = [
        item
        for item in source_intelligence.get("document_or_cluster_intelligence_manifest", [])
        if isinstance(item, dict) and str(item.get("source_type")) in source_types
    ]
    focus = [
        item
        for item in source_intelligence.get("evidence_rich_intelligence_focus", [])
        if isinstance(item, dict) and str(item.get("source_type")) in source_types
    ]
    focus.sort(
        key=lambda item: str(item.get("latest_collected_at") or item.get("published_at") or ""),
        reverse=True,
    )
    metrics = [
        item
        for item in source_intelligence.get("metrics_rollup", [])
        if isinstance(item, dict) and str(item.get("source_type")) in source_types
    ]
    return _fit_stage_intelligence(
        {
            "stage_id": stage.get("stage_id"),
            "source_types": list(source_types),
            "source_counts": _stage_source_counts(
                source_intelligence.get("source_counts", {}),
                source_types=source_types,
            ),
            "unit_rollup": source_intelligence.get("unit_rollup", {}),
            "keyword_rollup": source_intelligence.get("keyword_rollup", {}),
            "manifest": manifest,
            "evidence_focus": focus,
            "metrics_rollup": metrics,
            "evidence_index": source_intelligence.get("evidence_index", {}),
        }
    )


def _stage_has_units(stage_input: dict[str, Any]) -> bool:
    return bool(stage_input.get("manifest") or stage_input.get("evidence_focus"))


def _stage_source_counts(
    source_counts: dict[str, Any],
    *,
    source_types: set[str],
) -> dict[str, Any]:
    raw = source_counts.get("raw_matched", {}) if isinstance(source_counts, dict) else {}
    return {
        "raw_matched": {
            source_type: int(raw.get(source_type) or 0) for source_type in sorted(source_types)
        },
        "analysis_units": source_counts.get("analysis_units", {})
        if isinstance(source_counts, dict)
        else {},
        "document_or_cluster_intelligence_created": source_counts.get(
            "document_or_cluster_intelligence_created", {}
        )
        if isinstance(source_counts, dict)
        else {},
    }


def _fit_stage_intelligence(stage_input: dict[str, Any]) -> dict[str, Any]:
    candidate = json.loads(json.dumps(stage_input, ensure_ascii=False, default=str))
    manifest = candidate.get("manifest", [])
    if isinstance(manifest, list):
        candidate["manifest"] = [
            _manifest_line(item) if isinstance(item, dict) else str(item) for item in manifest
        ]
    focus = candidate.get("evidence_focus", [])
    if isinstance(focus, list):
        candidate["evidence_focus"] = focus[:12]
    metrics = candidate.get("metrics_rollup", [])
    if isinstance(metrics, list):
        candidate["metrics_rollup"] = metrics[:12]
    keywords = candidate.get("keyword_rollup", {})
    if isinstance(keywords, dict):
        for key, values in list(keywords.items()):
            if isinstance(values, list):
                keywords[key] = values[:20]
    if len(json.dumps(candidate, ensure_ascii=False, default=str)) <= 70_000:
        return candidate
    if isinstance(candidate.get("evidence_focus"), list):
        candidate["evidence_focus"] = candidate["evidence_focus"][:6]
    if isinstance(candidate.get("metrics_rollup"), list):
        candidate["metrics_rollup"] = candidate["metrics_rollup"][:6]
    return candidate


def _sanitize_stage_summary(summary: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "stage_id",
        "source_types",
        "processed_intelligence_counts",
        "business_area_findings",
        "capability_findings",
        "strategic_keyword_findings",
        "target_industry_findings",
        "target_customer_group_findings",
        "activity_findings",
        "metric_findings",
        "sector_findings",
        "strongest_sources",
        "uncertain_points",
        "data_limitations",
        "confidence",
    }
    sanitized = {key: value for key, value in summary.items() if key in allowed_keys}
    sanitized["strongest_sources"] = _normalize_strongest_sources(
        sanitized.get("strongest_sources", [])
    )
    return sanitized


def _build_final_reducer_input(
    source_intelligence: dict[str, Any],
    *,
    stage_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "company_id": source_intelligence.get("company_id"),
        "company_name": source_intelligence.get("company_name"),
        "role": source_intelligence.get("role"),
        "source_counts": source_intelligence.get("source_counts", _empty_source_count_bundle()),
        "retrieval_notes": source_intelligence.get("retrieval_notes", []),
        "unit_rollup": source_intelligence.get("unit_rollup", {}),
        "keyword_rollup": source_intelligence.get("keyword_rollup", {}),
        "metrics_rollup": source_intelligence.get("metrics_rollup", []),
        "document_or_cluster_intelligence_manifest": source_intelligence.get(
            "document_or_cluster_intelligence_manifest", []
        ),
        "evidence_rich_intelligence_focus": source_intelligence.get(
            "evidence_rich_intelligence_focus", []
        ),
        "source_stage_summaries": stage_summaries,
        "evidence_index": source_intelligence.get("evidence_index", {}),
        "llm_flow": {
            "mode": "sequential_stage_then_final_reducer",
            "stage_order": [stage["stage_id"] for stage in _source_stage_plan()],
            "note": "DART/IR -> official -> news_cluster -> securities_report 순서로 stage summary를 만들고 final reducer가 종합",
        },
    }


def _fit_final_source_intelligence(source_intelligence: dict[str, Any]) -> dict[str, Any]:
    """Keep final LLM input bounded while preserving aggregate counts."""
    fitted = json.loads(json.dumps(source_intelligence, ensure_ascii=False, default=str))
    budgets = [
        {
            "evidence_focus": 16,
            "metrics": 15,
            "keywords": 20,
            "article_ids": 50,
            "cluster_ids": 35,
            "target_chars": 100_000,
        },
        {
            "evidence_focus": 8,
            "metrics": 8,
            "keywords": 12,
            "article_ids": 20,
            "cluster_ids": 15,
            "target_chars": 80_000,
        },
        {
            "evidence_focus": 4,
            "metrics": 4,
            "keywords": 8,
            "article_ids": 10,
            "cluster_ids": 8,
            "target_chars": 60_000,
        },
    ]
    for budget in budgets:
        candidate = _apply_intelligence_budget(fitted, budget)
        if len(json.dumps(candidate, ensure_ascii=False, default=str)) <= int(
            budget["target_chars"]
        ):
            return candidate
        fitted = candidate
    return fitted


def _apply_intelligence_budget(data: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    candidate = json.loads(json.dumps(data, ensure_ascii=False, default=str))
    focus = candidate.get("evidence_rich_intelligence_focus", [])
    if isinstance(focus, list):
        candidate["evidence_rich_intelligence_focus"] = focus[
            : int(budget.get("evidence_focus", 0))
        ]
    metrics = candidate.get("metrics_rollup", [])
    if isinstance(metrics, list):
        candidate["metrics_rollup"] = metrics[: int(budget.get("metrics", 0))]
    keywords = candidate.get("keyword_rollup", {})
    if isinstance(keywords, dict):
        for key, values in list(keywords.items()):
            if isinstance(values, list):
                keywords[key] = values[: int(budget.get("keywords", 0))]
    evidence = candidate.get("evidence_index", {})
    if isinstance(evidence, dict):
        article_ids = evidence.get("sample_article_ids", [])
        cluster_ids = evidence.get("sample_cluster_ids", [])
        if isinstance(article_ids, list):
            evidence["sample_article_ids"] = article_ids[: int(budget.get("article_ids", 0))]
        if isinstance(cluster_ids, list):
            evidence["sample_cluster_ids"] = cluster_ids[: int(budget.get("cluster_ids", 0))]
    candidate["llm_input_budget"] = {
        "applied": True,
        "target_chars": budget.get("target_chars"),
        "note": "최종 LLM 입력은 raw 원문이 아니라 전체 document/cluster intelligence manifest, 집계 rollup, 근거 focus를 사용",
    }
    if len(json.dumps(candidate, ensure_ascii=False, default=str)) > int(
        budget.get("target_chars", 0)
    ):
        candidate = _minimize_manifest(candidate)
    return candidate


def _minimize_manifest(data: dict[str, Any]) -> dict[str, Any]:
    candidate = json.loads(json.dumps(data, ensure_ascii=False, default=str))
    manifest = candidate.get("document_or_cluster_intelligence_manifest", [])
    if isinstance(manifest, list):
        minimized_manifest: list[str] = []
        for item in manifest:
            if isinstance(item, str):
                minimized_manifest.append(item)
            elif isinstance(item, dict):
                minimized_manifest.append(_manifest_line(item))
        candidate["document_or_cluster_intelligence_manifest"] = minimized_manifest
    candidate.setdefault("llm_input_budget", {})["manifest_minimized"] = True
    return candidate


def _manifest_line(item: dict[str, Any]) -> str:
    sector_ids = ",".join(str(value) for value in item.get("sector_ids", []) if value)
    parts = [
        str(item.get("unit_id") or ""),
        str(item.get("intelligence_type") or ""),
        str(item.get("date") or ""),
        f"article={item.get('article_id') or ''}",
        f"cluster={item.get('cluster_id') or ''}",
        f"sectors={sector_ids}",
        f"metrics={item.get('metric_count') or 0}",
        f"activities={item.get('activity_count') or 0}",
        f"match={item.get('direct_company_match_confidence') or ''}",
    ]
    return "|".join(parts)


def _source_priority(source_type: str) -> int:
    return {
        "dart": 0,
        "qdrant_dart": 0,
        "ir": 1,
        "qdrant_ir": 1,
        "official": 2,
        "business_signal": 3,
        "financial_metric": 4,
        "news": 5,
        "securities_report": 6,
    }.get(source_type, 9)


def _update_counter(counter: Counter[str], values: Any) -> None:
    if not isinstance(values, list):
        return
    for value in values:
        text_value = str(value or "").strip()
        if text_value:
            counter[text_value] += 1


def _counter_to_items(counter: Counter[str], *, limit: int) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def _compact_unit_intelligence_for_final(unit: dict[str, Any]) -> dict[str, Any]:
    evidence_text = ""
    evidence = unit.get("evidence_texts")
    if isinstance(evidence, list) and evidence:
        first = evidence[0]
        if isinstance(first, dict):
            evidence_text = str(first.get("text") or "")
    direct_match = unit.get("direct_company_match", {})
    sector_signals = []
    for signal in (
        unit.get("sector_signals", [])[:4] if isinstance(unit.get("sector_signals"), list) else []
    ):
        if not isinstance(signal, dict):
            continue
        sector_signals.append(
            {
                "sector_id": signal.get("sector_id"),
                "sector_name": signal.get("sector_name"),
                "mapping_status": signal.get("mapping_status"),
                "matched_keywords": signal.get("matched_keywords", [])[:5]
                if isinstance(signal.get("matched_keywords"), list)
                else [],
                "evidence_strength": signal.get("evidence_strength"),
            }
        )
    return {
        "unit_id": unit.get("unit_id"),
        "intelligence_type": _intelligence_type_for_unit(unit),
        "unit_type": unit.get("unit_type"),
        "source_type": unit.get("source_type"),
        "title": _compact_text(str(unit.get("title") or ""), limit=100),
        "article_ids": unit.get("article_ids", [])[:6],
        "cluster_id": unit.get("cluster_id"),
        "source_count": unit.get("source_count"),
        "published_at": unit.get("published_at"),
        "latest_collected_at": unit.get("latest_collected_at"),
        "direct_company_match_confidence": direct_match.get("match_confidence")
        if isinstance(direct_match, dict)
        else None,
        "business_area_candidates": unit.get("business_area_candidates", [])[:2],
        "capability_keywords": unit.get("capability_keywords", [])[:4],
        "strategic_keywords": unit.get("strategic_keywords", [])[:4],
        "target_industries": unit.get("target_industries", [])[:3],
        "target_customer_groups": unit.get("target_customer_groups", [])[:3],
        "sector_signals": sector_signals,
        "quantitative_signals": [
            _compact_metric(metric, unit)
            for metric in unit.get("quantitative_signals", [])[:1]
            if isinstance(metric, dict)
        ],
        "activity_signals": unit.get("activity_signals", [])[:3],
        "evidence_text": _compact_text(evidence_text, limit=90),
        "confidence": unit.get("confidence"),
    }


def _unit_manifest_for_final(unit: dict[str, Any]) -> dict[str, Any]:
    direct_match = unit.get("direct_company_match", {})
    article_ids = unit.get("article_ids", []) if isinstance(unit.get("article_ids"), list) else []
    sector_ids: list[str] = []
    for signal in unit.get("sector_signals", []) or []:
        if isinstance(signal, dict) and signal.get("sector_id"):
            sector_ids.append(str(signal.get("sector_id")))
    return {
        "unit_id": unit.get("unit_id"),
        "intelligence_type": _intelligence_type_for_unit(unit),
        "source_type": unit.get("source_type"),
        "article_id": article_ids[0] if article_ids else None,
        "cluster_id": unit.get("cluster_id"),
        "title": _compact_text(str(unit.get("title") or ""), limit=70),
        "date": unit.get("published_at") or unit.get("latest_collected_at"),
        "direct_company_match_confidence": direct_match.get("match_confidence")
        if isinstance(direct_match, dict)
        else None,
        "sector_ids": list(dict.fromkeys(sector_ids)),
        "business_area_count": len(unit.get("business_area_candidates", []) or []),
        "capability_count": len(unit.get("capability_keywords", []) or []),
        "metric_count": len(unit.get("quantitative_signals", []) or []),
        "activity_count": len(unit.get("activity_signals", []) or []),
        "confidence": unit.get("confidence"),
    }


def _intelligence_type_for_unit(unit: dict[str, Any]) -> str:
    source_type = str(unit.get("source_type") or "")
    unit_type = str(unit.get("unit_type") or "")
    if unit_type == "news_cluster":
        return "news_cluster_intelligence"
    if source_type == "dart":
        return "dart_document_intelligence"
    if source_type == "ir":
        return "ir_document_intelligence"
    if source_type == "official":
        return "official_document_intelligence"
    if source_type == "securities_report":
        return "securities_report_document_intelligence"
    return f"{source_type}_document_intelligence"


def _compact_metric(metric: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    article_ids = unit.get("article_ids", [])
    article_id = article_ids[0] if isinstance(article_ids, list) and article_ids else None
    return {
        "metric_name": metric.get("metric_name"),
        "value": metric.get("value"),
        "unit": metric.get("unit"),
        "period": metric.get("period"),
        "metric_context": metric.get("metric_context"),
        "metric_type": metric.get("metric_type"),
        "source_type": unit.get("source_type"),
        "source_title": _compact_text(str(unit.get("title") or ""), limit=120),
        "article_id": metric.get("article_id") or article_id,
        "unit_id": unit.get("unit_id"),
        "direct_company_match": metric.get("direct_company_match")
        or unit.get("direct_company_match")
        or {},
        "caution": metric.get("caution"),
    }


def _news_cluster_unit_from_rows(
    representative_row: Any,
    rows: list[dict[str, Any]],
    *,
    company_config: dict[str, Any],
    sector_config: list[dict[str, Any]],
) -> dict[str, Any]:
    rep = dict(representative_row)
    if not rows:
        rows = [rep]
    cluster_id = str(rep.get("cluster_id"))
    title = str(rep.get("title") or "")
    article_ids = [int(row["id"]) for row in rows if row.get("id") is not None]
    related_titles = [
        str(row.get("title") or "")
        for row in rows
        if row.get("id") != rep.get("id") and row.get("title")
    ][:8]
    combined_text = " ".join(
        [title, *related_titles, *[str(row.get("content") or "")[:700] for row in rows[:5]]]
    )
    sector_signals = _sector_signals_for_text(combined_text, sector_config)
    direct_match = _direct_company_match(rep, company_config)
    row_datetimes = [value for row in rows if (value := _document_datetime(row)) is not None]
    latest_collected_at = max(row_datetimes) if row_datetimes else _document_datetime(rep)
    return {
        "unit_id": f"news_cluster:{cluster_id}",
        "unit_type": "news_cluster",
        "source_type": "news",
        "company_id": company_config.get("company_id"),
        "title": title,
        "representative_title": title,
        "related_titles": related_titles,
        "article_ids": article_ids,
        "cluster_id": cluster_id,
        "source_count": len(rows),
        "published_at": _iso_or_empty(rep.get("published_at")),
        "collected_at": _iso_or_empty(rep.get("collected_at")),
        "latest_collected_at": _iso_or_empty(latest_collected_at),
        "direct_company_match": direct_match,
        "business_area_candidates": _business_area_candidates(title),
        "capability_keywords": _matched_keywords_from_sector_signals(sector_signals),
        "strategic_keywords": _strategic_keywords_for_text(combined_text),
        "target_industries": [],
        "target_customer_groups": _target_customer_groups_for_text(combined_text),
        "sector_signals": sector_signals,
        "quantitative_signals": _quantitative_signals_for_text(
            combined_text,
            source_type="news",
            source_title=title,
            direct_company_match=direct_match,
            article_id=int(rep["id"]) if rep.get("id") is not None else None,
        ),
        "activity_signals": _activity_signals_for_text(combined_text, source_type="news"),
        "evidence_texts": [
            {
                "text": _compact_text(combined_text, limit=_UNIT_EVIDENCE_LIMIT),
                "why_used": "대표 클러스터와 관련 기사 묶음을 함께 읽어 생성한 뉴스 클러스터 근거",
            }
        ],
        "confidence": "medium" if direct_match.get("match_confidence") != "low" else "low",
    }


def _sector_signals_for_text(
    text_value: str, sector_config: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    lowered = text_value.lower()
    signals: list[dict[str, Any]] = []
    for sector in sector_config:
        keywords = [str(keyword) for keyword in sector.get("keywords", []) if keyword]
        aliases = [str(alias) for alias in sector.get("aliases", []) if alias]
        matched = [
            keyword for keyword in [*keywords, *aliases] if keyword and keyword.lower() in lowered
        ]
        if not matched:
            continue
        strength = "strong" if len(matched) >= 2 else "medium"
        signals.append(
            {
                "sector_id": sector.get("sector_id"),
                "sector_name": sector.get("sector_name"),
                "mapping_status": "direct_match" if strength == "strong" else "partial_match",
                "matched_keywords": matched[:12],
                "evidence_strength": strength,
                "reason": "문서 텍스트가 sectors.py 키워드/alias와 매칭됨",
            }
        )
    if not signals:
        signals.append(
            {
                "sector_id": None,
                "sector_name": None,
                "mapping_status": "no_direct_sector_match",
                "matched_keywords": [],
                "evidence_strength": "weak",
                "reason": "sectors.py 기준 직접 매칭 키워드 없음",
            }
        )
    return signals


def _matched_keywords_from_sector_signals(signals: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for signal in signals:
        values.extend(str(item) for item in signal.get("matched_keywords", []) if item)
    return list(dict.fromkeys(values))[:20]


def _business_area_candidates(title: str) -> list[str]:
    cleaned = re.sub(r"[\[\]\"'‘’“”]", "", title).strip()
    if not cleaned:
        return []
    return [_compact_text(cleaned, limit=80)]


def _strategic_keywords_for_text(text_value: str) -> list[str]:
    candidates = (
        "AI",
        "AX",
        "클라우드",
        "보안",
        "자동화",
        "글로벌",
        "플랫폼",
        "데이터",
        "수익성",
        "운영",
    )
    lowered = text_value.lower()
    return [keyword for keyword in candidates if keyword.lower() in lowered]


def _target_customer_groups_for_text(text_value: str) -> list[str]:
    groups = ("기업 고객", "엔터프라이즈 고객", "대기업", "공공기관", "금융기관")
    return [group for group in groups if group in text_value]


def _activity_signals_for_text(text_value: str, *, source_type: str) -> list[str]:
    if source_type not in {"official", "news"}:
        return []
    signals = []
    for keyword in ("수주", "협약", "MOU", "출시", "도입", "적용", "투자", "선정", "구축"):
        if keyword.lower() in text_value.lower():
            signals.append(keyword)
    return signals


def _quantitative_signals_for_text(
    text_value: str,
    *,
    source_type: str,
    source_title: str,
    direct_company_match: dict[str, Any],
    article_id: int | None = None,
) -> list[dict[str, Any]]:
    metric_type = {
        "dart": "official_metric",
        "ir": "official_metric",
        "official": "official_metric",
        "securities_report": "estimated_metric",
        "news": "reported_metric",
    }.get(source_type, "uncertain_metric")
    if direct_company_match.get("match_confidence") == "low":
        metric_type = "uncertain_metric"
    metrics: list[dict[str, Any]] = []
    for match in re.finditer(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(억원|조원|%|건|명)", text_value):
        context_start = max(0, match.start() - 70)
        context_end = min(len(text_value), match.end() + 70)
        metric_context = _compact_text(text_value[context_start:context_end], limit=180)
        metrics.append(
            {
                "metric_name": _infer_metric_name_from_context(metric_context, match.group(2)),
                "value": match.group(1),
                "unit": match.group(2),
                "period": "",
                "metric_context": metric_context,
                "metric_type": metric_type,
                "source_type": source_type,
                "source_title": source_title,
                "article_id": article_id,
                "direct_company_match": direct_company_match,
                "caution": "직접 귀속성 및 맥락은 최종 프로필 단계에서 재검증 필요",
            }
        )
        if len(metrics) >= 8:
            break
    return metrics


def _infer_metric_name_from_context(context: str, unit: str) -> str:
    checks = (
        ("영업이익", "영업이익"),
        ("매출액", "매출"),
        ("매출", "매출"),
        ("수주", "수주"),
        ("계약", "계약"),
        ("투자", "투자"),
        ("고객", "고객"),
        ("인력", "인력"),
        ("채용", "채용"),
        ("증가율", "증가율"),
        ("성장률", "성장률"),
    )
    matched = [label for keyword, label in checks if keyword in context]
    if "영업이익" in matched and unit == "%":
        return "영업이익 증가율"
    if "매출" in matched and unit == "%":
        return "매출 증가율"
    if "수주" in matched:
        return "수주 관련 수치"
    if "계약" in matched:
        return "계약 관련 수치"
    if "투자" in matched:
        return "투자 관련 수치"
    if "고객" in matched:
        return "고객 관련 수치"
    if "인력" in matched or "채용" in matched:
        return "인력/채용 관련 수치"
    if "매출" in matched:
        return "매출 관련 수치"
    if "영업이익" in matched:
        return "영업이익 관련 수치"
    if "증가율" in matched or "성장률" in matched:
        return "증가율/성장률 수치"
    return "맥락 확인 필요 수치"


def _unit_confidence(source_type: str, direct_match: dict[str, Any]) -> str:
    direct_confidence = str(direct_match.get("match_confidence") or "low")
    if source_type in {"dart", "ir", "official"} and direct_confidence in {"high", "medium"}:
        return "high"
    if source_type == "securities_report" and direct_confidence == "high":
        return "medium"
    if source_type == "news" and direct_confidence != "low":
        return "medium"
    return "low"


def _populate_analysis_unit_counts(
    *,
    source_counts: dict[str, Any],
    dart_count: int,
    ir_count: int,
    official_count: int,
    securities_count: int,
    news_cluster_count: int,
    business_signal_count: int = 0,
    financial_metric_count: int = 0,
    qdrant_chunk_count: int = 0,
) -> None:
    analysis_units = {
        "news_clusters": news_cluster_count,
        "official_documents": official_count,
        "securities_report_documents": securities_count,
        "dart_documents": dart_count,
        "ir_documents": ir_count,
        "business_signal_units": business_signal_count,
        "financial_metric_units": financial_metric_count,
        "qdrant_document_chunks": qdrant_chunk_count,
    }
    extracted = {
        "news_cluster_intelligence": news_cluster_count,
        "official_document_intelligence": official_count,
        "securities_report_document_intelligence": securities_count,
        "dart_document_intelligence": dart_count,
        "ir_document_intelligence": ir_count,
        "business_signal_intelligence": business_signal_count,
        "financial_metric_intelligence": financial_metric_count,
        "qdrant_document_chunk_intelligence": qdrant_chunk_count,
    }
    source_counts["analysis_units"].update(analysis_units)
    source_counts["document_or_cluster_intelligence_created"].update(extracted)
    source_counts["used_for_final_profile"].update(extracted)


def _derive_recency_focus(
    *,
    requested_days: int | None,
    historical_period: dict[str, Any],
    recent_signal_period: dict[str, Any],
    end: datetime,
) -> dict[str, Any]:
    if requested_days is not None:
        days = max(1, int(requested_days))
        return {
            "days": days,
            "start_date": (end - timedelta(days=days)).date().isoformat(),
            "policy": "manual_override",
            "reason": f"--lookback-days {days} 값으로 최신성 판단 focus window를 수동 지정",
        }

    baseline_end = _parse_iso_datetime(str(historical_period.get("end_date") or ""))
    recent_start = _parse_iso_datetime(str(recent_signal_period.get("start_date") or ""))
    recent_end = _parse_iso_datetime(str(recent_signal_period.get("end_date") or "")) or end

    if baseline_end is not None and recent_end > baseline_end:
        days = max(1, (recent_end.date() - baseline_end.date()).days)
        days = min(max(days, _MIN_RECENCY_FOCUS_DAYS), _MAX_RECENCY_FOCUS_DAYS)
        return {
            "days": days,
            "start_date": (end - timedelta(days=days)).date().isoformat(),
            "policy": "auto_from_latest_dart_ir",
            "reason": (
                "최신 DART/IR 기준일 이후 official/news/securities_report 신호를 "
                f"보기 위해 {days}일 focus window를 자동 산정"
            ),
        }

    if recent_start is not None and recent_end is not None:
        days = max(1, (recent_end.date() - recent_start.date()).days)
        days = min(max(days, _MIN_RECENCY_FOCUS_DAYS), _MAX_RECENCY_FOCUS_DAYS)
        return {
            "days": days,
            "start_date": (end - timedelta(days=days)).date().isoformat(),
            "policy": "auto_from_recent_signal_span",
            "reason": (
                "DART/IR 기준일을 확정하기 어려워 recent source 날짜 분포 기준으로 "
                f"{days}일 focus window를 자동 산정"
            ),
        }

    return {
        "days": _DEFAULT_RECENCY_FOCUS_DAYS,
        "start_date": (end - timedelta(days=_DEFAULT_RECENCY_FOCUS_DAYS)).date().isoformat(),
        "policy": "default",
        "reason": "기준 날짜가 부족해 기본 180일 focus window 사용",
    }


def _row_to_document(
    row: Any,
    company_config: dict[str, Any],
    *,
    role_hint: str,
) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    document_datetime = _document_datetime(row)
    return {
        "article_id": int(row["id"]),
        "source_type": row.get("source_type"),
        "profile_input_role": role_hint,
        "source_name": row.get("source_name"),
        "publisher": row.get("publisher"),
        "title": row.get("title"),
        "url": row.get("url"),
        "document_date": _iso_or_empty(document_datetime),
        "document_date_basis": "published_at" if row.get("published_at") else "collected_at",
        "published_at": _iso_or_empty(row.get("published_at")),
        "collected_at": _iso_or_empty(row.get("collected_at")),
        "cluster_id": row.get("cluster_id"),
        "processing_status": row.get("processing_status"),
        "is_representative": bool(row.get("is_representative"))
        if row.get("is_representative") is not None
        else None,
        "direct_company_match": _direct_company_match(row, company_config),
        "metadata": _metadata_for_prompt(metadata),
        "content_excerpt": _compact_text(str(row.get("content") or ""), limit=_CONTENT_LIMIT),
    }


def _invoke_json_prompt(prompt: str, *, phase: str) -> dict[str, Any]:
    try:
        from src.observability import tracing_config

        response = _get_llm().invoke(
            prompt,
            config=tracing_config(
                agent="CompanyProfileAgent",
                phase=phase,
                prompt_version=_PROMPT_VERSION,
            ),
        )
    except Exception:
        response = _get_llm().invoke(prompt)

    raw_text = response.content if isinstance(response.content, str) else str(response.content)
    try:
        return _parse_json(raw_text)
    except Exception as exc:
        log.warning("회사 프로필 JSON 파싱 실패, repair 시도 | phase=%s error=%s", phase, exc)
        repaired = _repair_json(raw_text, str(exc))
        return _parse_json(repaired)


def _repair_json(raw_text: str, error: str) -> str:
    prompt = _JSON_REPAIR_PROMPT.replace("{error}", error).replace("{raw_text}", raw_text)
    response = _get_llm().invoke(prompt)
    return response.content if isinstance(response.content, str) else str(response.content)


def _repair_profile_if_needed(
    profile: dict[str, Any],
    sector_config: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = {str(item.get("sector_id")) for item in sector_config if item.get("sector_id")}
    current = {
        str(item.get("sector_id"))
        for item in profile.get("sector_profile", [])
        if isinstance(item, dict) and item.get("sector_id")
    }
    if not expected or expected.issubset(current):
        if not expected:
            quality = profile.setdefault("quality", {})
            limitations = quality.setdefault("data_limitations", [])
            if isinstance(limitations, list) and "섹터 설정 없음" not in limitations:
                limitations.append("섹터 설정 없음")
        return profile

    prompt = _PROFILE_REPAIR_PROMPT.replace(
        "{sector_config_json}",
        json.dumps(sector_config, ensure_ascii=False, indent=2),
    ).replace("{profile_json}", json.dumps(profile, ensure_ascii=False, indent=2))
    repaired = _invoke_json_prompt(prompt, phase="company_profile_repair")
    return repaired


def _sanitize_company_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Remove fields that belong to downstream implication/response agents."""
    disallowed_keys = (
        "card_usage_rules",
        "response_rules",
        "action_suggestions",
        "news_signal_rules",
        "opportunity_risk_profile",
        "implication_usage_rules",
        "response_usage_rules",
    )
    for key in disallowed_keys:
        profile.pop(key, None)
    profile["schema_version"] = _PROMPT_VERSION
    profile["prompt_version"] = _PROMPT_VERSION
    return profile


def _normalize_structured_profile_sections(profile: dict[str, Any]) -> None:
    business_profile = profile.setdefault("business_profile", {})
    business_profile["core_business_areas"] = _normalize_named_object_list(
        business_profile.get("core_business_areas"),
        name_key="area",
        extra_defaults={
            "description": "",
            "evidence_strength": "weak",
            "sector_mapping": {
                "matched_sector_ids": [],
                "mapping_status": "no_direct_sector_match",
                "mapping_reason": "LLM 출력 보정 필요",
            },
            "analysis_scope": "profile_context_only",
            "basis": [],
        },
    )
    business_profile["capabilities"] = _normalize_named_object_list(
        business_profile.get("capabilities"),
        name_key="keyword",
        extra_defaults={"description": "", "evidence_strength": "weak", "basis": []},
    )
    business_profile["strategic_keywords"] = _normalize_named_object_list(
        business_profile.get("strategic_keywords"),
        name_key="keyword",
        extra_defaults={"meaning": "", "evidence_strength": "weak", "basis": []},
    )

    recent = profile.setdefault("recent_activity_profile", {})
    recent["key_activities"] = _normalize_named_object_list(
        recent.get("key_activities"),
        name_key="activity",
        extra_defaults={
            "source_type": None,
            "cluster_id": None,
            "related_keywords": [],
            "related_sectors": [],
            "related_industries": [],
            "interpretation_level": "recent_signal",
            "evidence_strength": "weak",
            "basis": [],
        },
    )

    summary = profile.setdefault("profile_summary", {})
    for key in ("one_line", "positioning", "current_direction", "baseline_vs_recent"):
        if not str(summary.get(key) or "").strip():
            summary[key] = "입력된 source intelligence 기준 추가 구체화 필요"
    if not str(profile.get("final_one_line") or "").strip():
        profile["final_one_line"] = summary.get("one_line")


def _normalize_named_object_list(
    value: Any,
    *,
    name_key: str,
    extra_defaults: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            merged = dict(extra_defaults)
            merged.update(item)
            normalized.append(merged)
        elif isinstance(item, str) and item.strip():
            merged = dict(extra_defaults)
            merged[name_key] = item.strip()
            normalized.append(merged)
    return normalized


def _ensure_sector_profile(profile: dict[str, Any], sector_config: list[dict[str, Any]]) -> None:
    existing = {
        str(item.get("sector_id")): item
        for item in profile.get("sector_profile", [])
        if isinstance(item, dict) and item.get("sector_id")
    }
    completed: list[dict[str, Any]] = []
    for sector in sector_config:
        sector_id = str(sector.get("sector_id") or "")
        current = dict(existing.get(sector_id, {}))
        current.setdefault("sector_id", sector_id)
        current.setdefault("sector_name", sector.get("sector_name", sector_id))
        current.setdefault("mapping_status", "no_match")
        current.setdefault("company_stance", "해당 섹터와 직접 연결되는 강한 근거는 추가 확인 필요")
        current.setdefault("evidence_strength", "weak")
        current.setdefault(
            "basis_by_source_type",
            {
                "dart_ir_basis": None,
                "official_basis": None,
                "news_basis": None,
                "external_analysis_basis": None,
            },
        )
        current.setdefault("related_business_areas", [])
        current.setdefault("related_capabilities", [])
        current.setdefault("related_target_industries", [])
        current.setdefault("metrics", [])
        current.setdefault("confidence", "low")
        current.setdefault("caution", "")
        completed.append(current)
    profile["sector_profile"] = completed


def validate_profile(profile: dict[str, Any]) -> None:
    quality = profile.setdefault("quality", {})
    uncertain = quality.setdefault("uncertain_points", [])
    limitations = quality.setdefault("data_limitations", [])
    if not isinstance(uncertain, list):
        uncertain = [str(uncertain)]
        quality["uncertain_points"] = uncertain
    if not isinstance(limitations, list):
        limitations = [str(limitations)]
        quality["data_limitations"] = limitations

    _validate_historical_baseline_profile(profile, limitations)
    _split_target_industries_and_customer_groups(profile)
    _validate_recent_activities(profile, uncertain)
    _validate_strongest_sources(profile, limitations)
    _validate_metrics_profile(profile)
    _validate_sector_profile(profile)


def _validate_historical_baseline_profile(profile: dict[str, Any], limitations: list[str]) -> None:
    baseline = profile.setdefault("historical_baseline_profile", {})
    field_name_key = {
        "repeated_business_areas": "area",
        "repeated_capabilities": "capability",
        "repeated_strategic_keywords": "keyword",
        "stable_target_industries": "industry",
    }
    for field, name_key in field_name_key.items():
        value = baseline.get(field, [])
        normalized: list[dict[str, Any]] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    item.setdefault(name_key, item.get("value") or item.get(name_key) or "")
                    item.setdefault("periods_observed", [])
                    item.setdefault("source_refs", [])
                    item.setdefault("evidence_summary", "근거 요약 추가 확인 필요")
                    item.setdefault("evidence_strength", "weak")
                    normalized.append(item)
                elif isinstance(item, str) and item.strip():
                    normalized.append(
                        {
                            name_key: item.strip(),
                            "periods_observed": [],
                            "source_refs": [],
                            "evidence_summary": "LLM이 문자열로 반환하여 원천 근거 요약 추가 확인 필요",
                            "evidence_strength": "weak",
                        }
                    )
                    note = f"historical_baseline_profile.{field} 문자열 항목을 object로 보정함"
                    if note not in limitations:
                        limitations.append(note)
        baseline[field] = normalized


_CUSTOMER_GROUP_TERMS: Final[set[str]] = {
    "기업",
    "기업 고객",
    "대기업",
    "금융기관",
    "공공기관",
    "엔터프라이즈 고객",
    "고객사",
    "기관",
}
_CUSTOMER_TO_INDUSTRY: Final[dict[str, str]] = {
    "금융기관": "금융",
    "공공기관": "공공",
}


def _split_target_industries_and_customer_groups(profile: dict[str, Any]) -> None:
    target = profile.setdefault("target_market_profile", {})
    raw_industries = _extract_named_objects(target.get("target_industries"), key="industry")
    raw_groups = _extract_named_objects(target.get("target_customer_groups"), key="customer_group")
    clean_industries: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {
        str(item.get("customer_group")): item for item in raw_groups if item.get("customer_group")
    }
    for item in raw_industries:
        industry = str(item.get("industry") or "").strip()
        if not industry:
            continue
        if _is_customer_group_value(industry):
            groups.setdefault(
                industry,
                {
                    "customer_group": industry,
                    "reason": item.get("reason") or "산업군이 아닌 고객군 표현으로 보정",
                    "evidence_strength": item.get("evidence_strength") or "weak",
                    "basis": item.get("basis") or [],
                },
            )
            mapped = _CUSTOMER_TO_INDUSTRY.get(industry)
            if mapped:
                clean_industries.setdefault(
                    mapped,
                    {
                        "industry": mapped,
                        "reason": f"{industry} 표현에서 산업군 '{mapped}'을 분리",
                        "evidence_strength": item.get("evidence_strength") or "weak",
                        "basis": item.get("basis") or [],
                        "metrics": item.get("metrics") or [],
                        "confidence": item.get("confidence") or "low",
                    },
                )
            continue
        item.setdefault("reason", "")
        item.setdefault("evidence_strength", "weak")
        item.setdefault("basis", [])
        item.setdefault("metrics", [])
        item.setdefault("confidence", "low")
        clean_industries.setdefault(industry, item)
    target["target_industries"] = list(clean_industries.values())
    target["target_customer_groups"] = list(groups.values())


def _extract_named_values(value: Any, *, key: str) -> list[str]:
    values: list[str] = []
    if not isinstance(value, list):
        return values
    for item in value:
        if isinstance(item, dict):
            candidate = str(item.get(key) or item.get("value") or "").strip()
        else:
            candidate = str(item or "").strip()
        if candidate:
            values.append(candidate)
    return values


def _extract_named_objects(value: Any, *, key: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            candidate = str(item.get(key) or item.get("value") or "").strip()
            if candidate:
                item[key] = candidate
                normalized.append(item)
        else:
            candidate = str(item or "").strip()
            if candidate:
                normalized.append({key: candidate})
    return normalized


def _is_customer_group_value(value: str) -> bool:
    stripped = value.strip()
    if stripped in _CUSTOMER_GROUP_TERMS:
        return True
    return any(
        term in stripped for term in ("기업 고객", "엔터프라이즈 고객", "고객사")
    ) or stripped in {"대기업", "기관"}


def _validate_recent_activities(profile: dict[str, Any], uncertain: list[str]) -> None:
    recent = profile.setdefault("recent_activity_profile", {})
    valid_items: list[dict[str, Any]] = []
    for item in recent.get("key_activities", []):
        if not isinstance(item, dict):
            continue
        basis = _valid_basis_items(item.get("basis"))
        source_type = item.get("source_type")
        evidence_strength = str(item.get("evidence_strength") or "").strip()
        if not source_type or not basis or not evidence_strength:
            activity = item.get("activity") or "unknown_activity"
            uncertain.append(f"recent_activity 근거 부족으로 제외: {activity}")
            continue
        if str(source_type) in {"news", "news_cluster"} and item.get("cluster_id") in {None, ""}:
            activity = item.get("activity") or "unknown_activity"
            uncertain.append(f"recent_activity 뉴스 클러스터 id 누락으로 제외: {activity}")
            continue
        item["basis"] = basis
        valid_items.append(item)
    recent["key_activities"] = valid_items


def _valid_basis_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    valid: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        has_title = bool(str(item.get("title") or item.get("source_title") or "").strip())
        has_article = item.get("article_id") is not None
        has_reason = bool(
            str(
                item.get("why_used")
                or item.get("evidence_text")
                or item.get("evidence_summary")
                or ""
            ).strip()
        )
        if (has_title or has_article) and has_reason:
            valid.append(item)
    return valid


def _validate_strongest_sources(profile: dict[str, Any], limitations: list[str]) -> None:
    evidence_map = profile.setdefault("evidence_map", {})
    normalized: list[dict[str, Any]] = []
    for item in evidence_map.get("strongest_sources", []):
        if not isinstance(item, dict):
            continue
        if not item.get("source_type") or not (item.get("article_id") or item.get("title")):
            continue
        if not item.get("why_strong"):
            continue
        normalized.append(item)
    if not normalized:
        note = "strongest_sources에 article_id/title/why_strong을 갖춘 실제 근거가 없어 빈 배열로 정리함"
        if note not in limitations:
            limitations.append(note)
    evidence_map["strongest_sources"] = normalized


def _validate_metrics_profile(profile: dict[str, Any]) -> None:
    metrics_profile = profile.setdefault("metrics_profile", {})
    uncertain_metrics = [
        _normalize_uncertain_metric(metric)
        for metric in list(metrics_profile.get("uncertain_metrics", []) or [])
        if isinstance(metric, dict)
    ]
    for bucket in ("official_metrics", "estimated_metrics", "reported_metrics"):
        kept: list[dict[str, Any]] = []
        for metric in metrics_profile.get(bucket, []) or []:
            if not isinstance(metric, dict):
                continue
            metric.setdefault("metric_context", "")
            metric.setdefault("direct_company_match", {})
            ambiguous = str(metric.get("metric_name") or "").strip() in {"", "문서 내 수치"}
            missing_context = not str(metric.get("metric_context") or "").strip()
            missing_direct = not isinstance(
                metric.get("direct_company_match"), dict
            ) or not metric.get("direct_company_match")
            if ambiguous or missing_context or missing_direct:
                uncertain_metrics.append(_normalize_uncertain_metric(metric))
            else:
                kept.append(metric)
        metrics_profile[bucket] = kept
    metrics_profile["uncertain_metrics"] = uncertain_metrics


def _normalize_uncertain_metric(metric: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metric)
    if str(normalized.get("metric_name") or "").strip() in {"", "문서 내 수치"}:
        normalized["metric_name"] = "맥락 확인 필요 수치"
    normalized.setdefault("value", "")
    normalized.setdefault("unit", "")
    normalized.setdefault("period", "")
    if not str(normalized.get("metric_context") or "").strip():
        normalized["metric_context"] = "수치 의미와 직접 귀속성 추가 확인 필요"
    normalized["metric_type"] = "uncertain_metric"
    normalized.setdefault("source_type", normalized.get("source_type"))
    normalized.setdefault("source_title", normalized.get("source_title"))
    normalized.setdefault("article_id", normalized.get("article_id"))
    direct = normalized.get("direct_company_match")
    if not isinstance(direct, dict) or not direct:
        normalized["direct_company_match"] = {
            "match_confidence": "low",
            "match_reason": "수치 직접 귀속성 정보 누락",
        }
    caution = str(normalized.get("caution") or "").strip()
    suffix = "수치명/맥락/직접 귀속성 확인이 부족해 uncertain_metrics로 이동."
    normalized["caution"] = f"{caution} {suffix}".strip() if suffix not in caution else caution
    return normalized


def _validate_sector_profile(profile: dict[str, Any]) -> None:
    for sector in profile.get("sector_profile", []):
        if not isinstance(sector, dict):
            continue
        evidence_text = _sector_evidence_text(sector)
        mapping_status = str(sector.get("mapping_status") or "")
        strength = str(sector.get("evidence_strength") or "")
        if (
            mapping_status == "direct_match"
            and (strength == "strong" or not evidence_text)
            and not evidence_text
        ):
            sector["mapping_status"] = "no_match"
            sector["evidence_strength"] = "weak"
            sector["confidence"] = "low"
            sector["caution"] = (
                "실제 근거 요약 없이 direct_match/strong으로 반환되어 no_match로 보정"
            )
            continue
        sector_id = str(sector.get("sector_id") or "")
        if mapping_status == "direct_match" and sector_id in {"deal", "security"}:
            required = _sector_required_terms(sector_id)
            if not any(term.lower() in evidence_text.lower() for term in required):
                sector["mapping_status"] = "partial_match"
                sector["evidence_strength"] = "weak"
                sector["confidence"] = "low"
                sector["caution"] = f"{sector_id} 직접 근거 키워드가 부족해 partial_match로 보정"


def _sector_evidence_text(sector: dict[str, Any]) -> str:
    basis = sector.get("basis_by_source_type", {})
    parts: list[str] = []
    if isinstance(basis, dict):
        for value in basis.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        parts.extend(
                            str(item.get(key) or "")
                            for key in ("evidence_summary", "why_used", "source_ref")
                        )
                    else:
                        parts.append(str(item or ""))
    meaningful = [part for part in parts if part and not _is_id_only_or_generic_basis(part)]
    return " ".join(meaningful)


def _is_id_only_or_generic_basis(value: str) -> bool:
    stripped = value.strip()
    if re.fullmatch(r"(dart|ir|news|official|securities_report):[A-Za-z0-9_-]+", stripped):
        return True
    generic_phrases = (
        "DART/IR에서 확인되는 공식 근거",
        "공식 뉴스룸에서 확인되는 실행 근거",
        "뉴스 클러스터에서 확인되는 최근 활동 신호",
        "증권 리포트에서 확인되는 외부 해석",
        "근거 요약 추가 확인 필요",
    )
    return stripped in generic_phrases


def _sector_required_terms(sector_id: str) -> tuple[str, ...]:
    if sector_id == "deal":
        return (
            "수주",
            "계약",
            "공급계약",
            "우선협상대상자",
            "MOU",
            "협약",
            "프로젝트",
            "지분투자",
            "사업자 선정",
        )
    if sector_id == "security":
        return ("보안", "정보보호", "사이버보안", "클라우드 보안", "침해", "랜섬웨어", "관제")
    return ()


def _inject_profile_diagnostics(
    *,
    profile: dict[str, Any],
    source_counts: dict[str, Any],
    retrieval_notes: list[str],
    sector_config: list[dict[str, Any]],
    final_source_intelligence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_map = profile.setdefault("evidence_map", {})
    evidence_map["source_counts"] = source_counts
    evidence_map["strongest_sources"] = _normalize_strongest_sources(
        evidence_map.get("strongest_sources", [])
    )
    if not evidence_map["strongest_sources"] and final_source_intelligence:
        evidence_map["strongest_sources"] = _fallback_strongest_sources_from_intelligence(
            final_source_intelligence
        )
    evidence_map.setdefault("source_article_ids", [])
    evidence_map.setdefault("source_cluster_ids", [])

    metrics_profile = profile.setdefault("metrics_profile", {})
    metrics_profile.setdefault("official_metrics", [])
    metrics_profile.setdefault("estimated_metrics", [])
    metrics_profile.setdefault("reported_metrics", [])
    metrics_profile.setdefault("uncertain_metrics", [])

    profile_summary = profile.setdefault("profile_summary", {})
    profile_summary.setdefault(
        "baseline_vs_recent",
        "DART/IR 누적 기준선과 최근 official/news/securities_report 신호를 분리해 비교해야 함",
    )
    profile.setdefault(
        "historical_baseline_profile",
        {
            "repeated_business_areas": [],
            "repeated_capabilities": [],
            "repeated_strategic_keywords": [],
            "stable_target_industries": [],
            "baseline_summary": "DART/IR 전체 누적 기준으로 반복 확인되는 공식 사업 구조",
        },
    )
    profile.setdefault("change_signals", [])
    _normalize_structured_profile_sections(profile)
    _ensure_sector_profile(profile, sector_config)
    if final_source_intelligence:
        _backfill_recent_activities_from_stage_summaries(profile, final_source_intelligence)
    validate_profile(profile)

    for sector in profile.get("sector_profile", []):
        if isinstance(sector, dict):
            sector["mapping_status"] = _normalize_sector_mapping_status(
                sector.get("mapping_status") or _sector_mapping_status(sector)
            )

    quality = profile.setdefault("quality", {})
    _cap_profile_confidence(quality, source_counts=source_counts, retrieval_notes=retrieval_notes)
    limitations = quality.setdefault("data_limitations", [])
    if not isinstance(limitations, list):
        limitations = [str(limitations)]
        quality["data_limitations"] = limitations
    for note in retrieval_notes:
        if note and note not in limitations:
            limitations.append(note)
    return profile


def _backfill_recent_activities_from_stage_summaries(
    profile: dict[str, Any],
    final_source_intelligence: dict[str, Any],
) -> None:
    recent = profile.setdefault("recent_activity_profile", {})
    current = recent.get("key_activities", [])
    if isinstance(current, list) and any(
        isinstance(item, dict) and item.get("source_type") and item.get("basis") for item in current
    ):
        return

    lookup = _intelligence_basis_lookup(final_source_intelligence)
    stage_summaries = final_source_intelligence.get("source_stage_summaries", [])
    if not isinstance(stage_summaries, list):
        return

    activities: list[dict[str, Any]] = []
    for stage in stage_summaries:
        if not isinstance(stage, dict):
            continue
        stage_source_type = _primary_stage_source_type(stage)
        if stage_source_type not in {"official", "news"}:
            continue
        stage_basis = _basis_objects_from_refs(
            stage.get("strongest_sources", []),
            lookup,
            fallback_source_type=stage_source_type,
        )
        for activity in stage.get("activity_findings", []) or []:
            if not isinstance(activity, dict):
                continue
            basis = _basis_objects_from_refs(
                activity.get("basis") or activity.get("source_refs") or [],
                lookup,
                fallback_source_type=stage_source_type,
            )
            if not basis:
                basis = stage_basis
            if not basis:
                continue
            activities.append(
                {
                    "activity": activity.get("activity")
                    or activity.get("summary")
                    or "최근 실행 신호",
                    "source_type": activity.get("source_type") or stage_source_type,
                    "cluster_id": activity.get("cluster_id"),
                    "related_keywords": activity.get("related_keywords", []),
                    "related_sectors": activity.get("related_sectors", []),
                    "related_industries": activity.get("related_industries", []),
                    "interpretation_level": "official_execution"
                    if stage_source_type == "official"
                    else "recent_signal",
                    "evidence_strength": activity.get("evidence_strength") or "medium",
                    "basis": basis,
                }
            )
    if activities:
        recent["key_activities"] = activities[:8]


def _primary_stage_source_type(stage: dict[str, Any]) -> str:
    source_types = stage.get("source_types", [])
    if isinstance(source_types, list) and source_types:
        return str(source_types[0])
    return ""


def _intelligence_basis_lookup(
    final_source_intelligence: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for unit in final_source_intelligence.get("evidence_rich_intelligence_focus", []) or []:
        if not isinstance(unit, dict):
            continue
        source_type = str(unit.get("source_type") or "")
        title = str(unit.get("title") or "")
        article_ids = (
            unit.get("article_ids", []) if isinstance(unit.get("article_ids"), list) else []
        )
        article_id = article_ids[0] if article_ids else None
        if source_type and article_id is not None:
            lookup[f"{source_type}:{article_id}"] = {
                "source_type": source_type,
                "article_id": article_id,
                "title": title,
                "why_used": "stage summary에서 복원한 document/cluster intelligence 근거",
            }
        cluster_id = unit.get("cluster_id")
        if source_type == "news" and cluster_id is not None:
            lookup[f"news_cluster:{cluster_id}"] = {
                "source_type": "news",
                "article_id": article_id,
                "title": title,
                "cluster_id": cluster_id,
                "why_used": "stage summary에서 복원한 news_cluster_intelligence 근거",
            }
    for line in (
        final_source_intelligence.get("document_or_cluster_intelligence_manifest", []) or []
    ):
        if not isinstance(line, str):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        unit_id = parts[0]
        source_type = unit_id.split(":", 1)[0] if ":" in unit_id else ""
        article_id = None
        cluster_id = None
        for part in parts:
            if part.startswith("article=") and part.removeprefix("article=").isdigit():
                article_id = int(part.removeprefix("article="))
            if part.startswith("cluster=") and part.removeprefix("cluster="):
                cluster_id = part.removeprefix("cluster=")
        if unit_id and source_type:
            lookup.setdefault(
                unit_id,
                {
                    "source_type": "news" if unit_id.startswith("news_cluster:") else source_type,
                    "article_id": article_id,
                    "title": unit_id,
                    "cluster_id": cluster_id,
                    "why_used": "manifest에서 복원한 document/cluster intelligence 근거",
                },
            )
    return lookup


def _basis_objects_from_refs(
    refs: Any,
    lookup: dict[str, dict[str, Any]],
    *,
    fallback_source_type: str,
) -> list[dict[str, Any]]:
    if not isinstance(refs, list):
        refs = [refs] if refs else []
    basis: list[dict[str, Any]] = []
    for ref in refs:
        if isinstance(ref, dict):
            normalized = dict(ref)
            normalized.setdefault(
                "why_used", normalized.get("evidence_summary") or "stage summary 근거"
            )
            if normalized.get("article_id") is not None or normalized.get("title"):
                basis.append(normalized)
            continue
        ref_text = str(ref or "").strip()
        if not ref_text:
            continue
        found = lookup.get(ref_text)
        if found:
            basis.append(dict(found))
            continue
        article_id = None
        source_type = fallback_source_type
        if ":" in ref_text:
            prefix, suffix = ref_text.split(":", 1)
            source_type = "news" if prefix == "news_cluster" else prefix
            if suffix.isdigit():
                article_id = int(suffix)
        basis.append(
            {
                "source_type": source_type,
                "article_id": article_id,
                "title": ref_text,
                "why_used": "stage summary가 반환한 source_ref 근거",
            }
        )
    return basis


def _fallback_strongest_sources_from_intelligence(
    final_source_intelligence: dict[str, Any],
) -> list[dict[str, Any]]:
    units = final_source_intelligence.get("evidence_rich_intelligence_focus", [])
    if not isinstance(units, list) or not units:
        units = final_source_intelligence.get("document_or_cluster_intelligence_manifest", [])
    if not isinstance(units, list):
        return []
    priority = {"dart": 0, "ir": 1, "official": 2, "news": 3, "securities_report": 4}
    candidates: list[dict[str, Any]] = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        title = str(unit.get("title") or "").strip()
        article_ids = unit.get("article_ids") if isinstance(unit.get("article_ids"), list) else []
        article_id = article_ids[0] if article_ids else None
        evidence_text = str(unit.get("evidence_text") or "").strip()
        source_type = str(unit.get("source_type") or "").strip()
        if not source_type or not title or not article_id or not evidence_text:
            continue
        confidence = str(unit.get("direct_company_match_confidence") or "")
        if confidence == "low" and source_type == "securities_report":
            continue
        candidates.append(
            {
                "source_type": source_type,
                "article_id": article_id,
                "title": title,
                "url": None,
                "why_strong": (
                    f"{unit.get('intelligence_type')} 기반 실제 근거. "
                    f"{_compact_text(evidence_text, limit=100)}"
                ),
                "_sort": (
                    priority.get(source_type, 99),
                    0 if confidence in {"high", "medium"} else 1,
                ),
            }
        )
    candidates.sort(key=lambda item: item.get("_sort", (99, 99)))
    for item in candidates:
        item.pop("_sort", None)
    return candidates[:5]


def _cap_profile_confidence(
    quality: dict[str, Any],
    *,
    source_counts: dict[str, Any],
    retrieval_notes: list[str],
) -> None:
    current = str(quality.get("confidence") or "").lower()
    if current != "high":
        return
    news = source_counts.get("news_pipeline", {}) if isinstance(source_counts, dict) else {}
    unclustered = int(news.get("unclustered_news_articles") or 0)
    representative = int(news.get("representative_news_clusters") or 0)
    raw_matched = source_counts.get("raw_matched", {}) if isinstance(source_counts, dict) else {}
    official_basis = int(raw_matched.get("official") or 0)
    dart_ir_basis = int(raw_matched.get("dart") or 0) + int(raw_matched.get("ir") or 0)
    has_limitations = bool(retrieval_notes)
    if unclustered > 0 or representative < 5 or has_limitations or official_basis < 2:
        quality["confidence"] = "medium"
        reason = str(quality.get("reason") or "")
        quality["reason"] = (
            f"{reason} 단, 뉴스 미클러스터 {unclustered}건, 대표 뉴스 {representative}건, "
            f"DART/IR 후보 {dart_ir_basis}건, 공식자료 후보 {official_basis}건 및 "
            "data_limitations를 고려해 최종 신뢰도를 medium으로 보정함."
        ).strip()


def _normalize_strongest_sources(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            candidate = {
                "source_type": item.get("source_type"),
                "article_id": item.get("article_id"),
                "title": item.get("title"),
                "url": item.get("url"),
                "why_strong": item.get("why_strong") or item.get("reason"),
            }
            if (
                candidate.get("source_type")
                and (candidate.get("article_id") or candidate.get("title"))
                and candidate.get("why_strong")
            ):
                normalized.append(candidate)
    return normalized


def _sector_mapping_status(sector: dict[str, Any]) -> str:
    evidence_strength = str(sector.get("evidence_strength") or "").lower()
    confidence = str(sector.get("confidence") or "").lower()
    if evidence_strength == "strong" and confidence in {"medium", "high"}:
        return "direct_match"
    if evidence_strength in {"weak", "medium"}:
        return "partial_match"
    return "no_match"


def _normalize_sector_mapping_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"direct_match", "partial_match", "no_match"}:
        return normalized
    if normalized in {"no_evidence", "no_direct_sector_match", "none", "no"}:
        return "no_match"
    return "partial_match" if normalized else "no_match"


def _limitations_from_counts(source_counts: dict[str, Any], *, company_id: str) -> list[str]:
    notes: list[str] = []
    news = source_counts.get("news_pipeline", {}) if isinstance(source_counts, dict) else {}
    raw_news = int(news.get("raw_news_articles") or 0)
    representative = int(news.get("representative_news_clusters") or 0)
    unclustered = int(news.get("unclustered_news_articles") or 0)
    if representative < 5:
        notes.append("뉴스 대표 클러스터 수가 적어 최근 활동 신호가 제한적임")
    if raw_news > 0 and representative == 0:
        notes.append(
            "raw 뉴스는 있으나 대표 클러스터가 없어 ArticleDeduplicator/전처리 파이프라인 미실행 가능성"
        )
    if unclustered > 0:
        notes.append("cluster_id가 없는 뉴스가 많아 프로필 입력에서 제외됨")

    raw_matched = source_counts.get("raw_matched", {}) if isinstance(source_counts, dict) else {}
    if company_id == "sk_ax" and not (raw_matched.get("dart") or raw_matched.get("ir")):
        notes.append("SK AX 단독 DART/IR 근거가 제한적임")
    return notes


def _parse_json(text_value: str) -> dict[str, Any]:
    text_value = text_value.strip()
    if text_value.startswith("```"):
        text_value = re.sub(r"^```(?:json)?", "", text_value.strip(), flags=re.IGNORECASE)
        text_value = re.sub(r"```$", "", text_value.strip())
    data = json.loads(text_value.strip())
    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object")
    return data


def _basis_period(
    *,
    historical_period: dict[str, Any],
    recent_signal_period: dict[str, Any],
    recency_focus: dict[str, Any],
    end: datetime,
    fallback_policy: dict[str, Any],
) -> dict[str, Any]:
    return {
        "historical_baseline": historical_period,
        "recent_signal_period": {
            "source_types": list(_RECENT_SIGNAL_SOURCE_TYPES),
            "start_date": recent_signal_period.get("start_date"),
            "end_date": recent_signal_period.get("end_date") or end.date().isoformat(),
            "recency_focus_days": recency_focus.get("days"),
            "recency_focus_start_date": recency_focus.get("start_date"),
            "recency_focus_policy": recency_focus.get("policy"),
            "recency_focus_reason": recency_focus.get("reason"),
            "purpose": "전체 누적 official/news/securities_report 후보 중 최신 실행 신호/외부 분석/뉴스 활동을 구분",
        },
        "date_basis": "published_at 우선, 없으면 collected_at fallback",
        "fallback_policy": fallback_policy,
    }


def _empty_source_counts() -> dict[str, int]:
    return {source_type: 0 for source_type in _SOURCE_TYPES}


def _empty_processing_status_counts() -> dict[str, int]:
    return {status: 0 for status in _PROCESSING_STATUS_BUCKETS}


def _empty_analysis_unit_counts() -> dict[str, int]:
    return {
        "news_clusters": 0,
        "official_documents": 0,
        "securities_report_documents": 0,
        "dart_documents": 0,
        "ir_documents": 0,
        "business_signal_units": 0,
        "financial_metric_units": 0,
        "qdrant_document_chunks": 0,
    }


def _empty_intelligence_counts() -> dict[str, int]:
    return {
        "news_cluster_intelligence": 0,
        "official_document_intelligence": 0,
        "securities_report_document_intelligence": 0,
        "dart_document_intelligence": 0,
        "ir_document_intelligence": 0,
        "business_signal_intelligence": 0,
        "financial_metric_intelligence": 0,
        "qdrant_document_chunk_intelligence": 0,
    }


def _empty_source_count_bundle() -> dict[str, Any]:
    return {
        "raw_matched": _empty_source_counts(),
        "analysis_units": _empty_analysis_unit_counts(),
        "document_or_cluster_intelligence_created": _empty_intelligence_counts(),
        "used_for_final_profile": _empty_intelligence_counts(),
        "news_pipeline": {
            "raw_news_articles": 0,
            "relevance_pass_articles": 0,
            "clustered_news_articles": 0,
            "representative_news_clusters": 0,
            "unclustered_news_articles": 0,
        },
        "by_processing_status": _empty_processing_status_counts(),
        "structured_evidence": {
            "business_signals": 0,
            "financial_metrics": 0,
            "qdrant_chunks": 0,
        },
    }


def _normalize_company_id(company_id: str) -> str:
    normalized = str(company_id or "").strip()
    if normalized not in _PROFILE_COMPANY_IDS:
        supported = ", ".join(sorted(_PROFILE_COMPANY_IDS))
        raise ValueError(f"unsupported company_id '{company_id}'. Supported: {supported}")
    if normalized not in COMPANY_IDS:
        raise ValueError(f"company_id '{normalized}' is not registered in src.config.companies")
    return normalized


def _role_for_company(company_id: str) -> Literal["peer", "self"]:
    return "self" if company_id in SELF_COMPANY_IDS else "peer"


def _metadata_for_prompt(metadata: dict[str, Any]) -> dict[str, Any]:
    keep_keys = (
        "page_kind",
        "source_family",
        "document_type",
        "report_type",
        "matched_companies",
        "matched_sectors",
        "summary",
    )
    return {key: metadata[key] for key in keep_keys if key in metadata}


def _direct_company_match(row: Any, company_config: dict[str, Any]) -> dict[str, Any]:
    company_id = str(company_config.get("company_id") or "")
    aliases = [str(alias) for alias in company_config.get("aliases", []) if alias]
    corp_code = str(company_config.get("dart_corp_code") or "")
    row_company = row.get("company") or []
    if isinstance(row_company, str):
        try:
            row_company = json.loads(row_company)
        except json.JSONDecodeError:
            row_company = [row_company]

    label_match = company_id in row_company if isinstance(row_company, list) else False
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    haystack = " ".join(
        [
            str(row.get("title") or ""),
            str(row.get("content") or ""),
            json.dumps(metadata, ensure_ascii=False, default=str),
            str(row.get("url") or ""),
            str(row.get("source_name") or ""),
            str(row.get("publisher") or ""),
        ]
    ).lower()
    alias_match = any(alias.lower() in haystack for alias in aliases)
    corp_code_match: bool | None = None
    if corp_code:
        corp_code_match = corp_code in haystack

    source_type = str(row.get("source_type") or "")
    if label_match and (alias_match or corp_code_match):
        confidence = "high"
        reason = "raw_articles.company 라벨과 companies.py alias/corp code 근거가 함께 확인됨"
    elif label_match and source_type in {"official", "news"}:
        confidence = "medium"
        reason = "raw_articles.company 라벨은 있으나 alias/corp code 직접 확인은 제한적임"
    elif label_match:
        confidence = "low"
        reason = "raw_articles.company 라벨만 확인되어 직접 귀속성 추가 검증 필요"
    elif alias_match or corp_code_match:
        confidence = "medium"
        reason = "company 라벨은 없지만 companies.py alias/corp code가 문서 내용에서 확인됨"
    else:
        confidence = "low"
        reason = "company 라벨, alias, corp code 직접 근거가 약함"

    return {
        "label_match": bool(label_match),
        "alias_match": bool(alias_match),
        "corp_code_match": corp_code_match,
        "match_confidence": confidence,
        "match_reason": reason,
    }


def _compact_text(value: str, *, limit: int) -> str:
    compacted = re.sub(r"\s+", " ", value).strip()
    if len(compacted) <= limit:
        return compacted
    return compacted[:limit].rstrip() + "..."


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _confidence_band(value: Any) -> str:
    numeric = _safe_float_or_none(value)
    if numeric is None:
        return "medium"
    if numeric >= 0.75:
        return "high"
    if numeric >= 0.45:
        return "medium"
    return "low"


def _metric_period(row: Any) -> str:
    year = row.get("period_year")
    quarter = row.get("period_quarter")
    if year and quarter:
        return f"{year}Q{quarter}"
    if year:
        return str(year)
    return ""


def _metric_evidence_text(metric: dict[str, Any]) -> str:
    value = metric.get("value_krwbn")
    unit = "십억원"
    if value is None:
        value = metric.get("value_numeric")
        unit = str(metric.get("unit") or "")
    label = metric.get("metric_label") or metric.get("metric_name") or "metric"
    period = metric.get("period") or ""
    value_text = "" if value is None else f"{value:g}{unit}"
    return " ".join(str(item) for item in [period, label, value_text] if str(item).strip())


def _iso_or_empty(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value or "")


def _document_datetime(row: Any) -> datetime | None:
    value = row.get("published_at") or row.get("collected_at")
    if isinstance(value, datetime):
        return _ensure_aware(value)
    if isinstance(value, str):
        return _parse_iso_datetime(value)
    return None


def _parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.fromisoformat(f"{value}T00:00:00+00:00")
        except ValueError:
            return None
    return _ensure_aware(parsed)


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def run_peer_profile_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for peer profile generation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    load_runtime_env()
    args = _parse_peer_cli_args(argv)
    if not args.all and not args.company:
        raise SystemExit("--company 또는 --all 중 하나를 지정하세요.")
    if args.all and args.company:
        raise SystemExit("--company와 --all은 동시에 사용할 수 없습니다.")

    company_ids = list(_PEER_COMPANY_IDS) if args.all else [args.company]
    sector_config = build_selected_sector_config()
    print_selected_sector_config(sector_config)

    results: list[tuple[str, bool, str]] = []
    agent = ProfileAgent()
    for company_id in company_ids:
        try:
            profile = agent.build_profile(company_id=company_id, lookback_days=args.lookback_days)
            emit_profile_result(company_id=company_id, profile=profile, dry_run=args.dry_run)
            results.append((company_id, True, "ok"))
        except Exception as exc:
            log.exception("프로필 생성 실패 | company=%s", company_id)
            results.append((company_id, False, f"{type(exc).__name__}: {exc}"))

    _print_summary(results)
    return 0 if all(ok for _, ok, _ in results) else 1


def run_skax_profile_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for SK AX profile generation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    load_runtime_env()
    args = _parse_skax_cli_args(argv)
    sector_config = build_selected_sector_config()
    print_selected_sector_config(sector_config)

    company_id = args.company
    try:
        profile = SKAXProfileAgent().build_profile(
            company_id=company_id,
            lookback_days=args.lookback_days,
        )
        emit_profile_result(company_id=company_id, profile=profile, dry_run=args.dry_run)
    except Exception as exc:
        log.exception("SK AX 프로필 생성 실패 | company=%s", company_id)
        _print_summary([(company_id, False, f"{type(exc).__name__}: {exc}")])
        return 1

    _print_summary([(company_id, True, "ok")])
    return 0


def _parse_peer_cli_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build peer company profile JSON.")
    parser.add_argument(
        "--company",
        choices=_PEER_COMPANY_IDS,
        help="특정 peer 회사 하나만 실행",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="peer 4사 전체 실행",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="최신성 판단 focus window 수동 override. 생략하면 DART/IR 및 전체 자료 날짜 흐름으로 자동 산정.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 저장 없이 콘솔에 JSON 출력",
    )
    return parser.parse_args(argv)


def _parse_skax_cli_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SK AX company profile JSON.")
    parser.add_argument(
        "--company",
        default="sk_ax",
        choices=tuple(SELF_COMPANY_IDS),
        help="SK AX self 회사 id",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="최신성 판단 focus window 수동 override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 저장 없이 콘솔에 JSON 출력",
    )
    return parser.parse_args(argv)


def load_runtime_env() -> str:
    """Load project dotenv values while preserving an explicit DATABASE_URL."""
    explicit_database_url = os.environ.get("DATABASE_URL")
    profile = load_profile()
    if explicit_database_url:
        os.environ["DATABASE_URL"] = explicit_database_url
    return profile


def print_selected_sector_config(sector_config: list[dict[str, Any]]) -> None:
    print(
        f"selected_sector_config={len(sector_config)} "
        f"ids={[item.get('sector_id') for item in sector_config]}"
    )


def emit_profile_result(*, company_id: str, profile: dict[str, Any], dry_run: bool) -> None:
    company_config = load_company_config(company_id)
    evidence_map = profile.get("evidence_map", {}) if isinstance(profile, dict) else {}
    source_counts = evidence_map.get("source_counts") or profile.get("source_counts") or {}
    quality = profile.get("quality", {}) if isinstance(profile, dict) else {}
    basis_period = profile.get("basis_period", {}) if isinstance(profile, dict) else {}
    recent_period = (
        basis_period.get("recent_signal_period", {}) if isinstance(basis_period, dict) else {}
    )
    limitations = quality.get("data_limitations", []) if isinstance(quality, dict) else []
    print(
        f"[{company_id}] company_name={company_config.get('company_name')} "
        f"aliases={len(company_config.get('aliases', []))}"
    )
    print(
        f"[{company_id}] raw_matched="
        f"{json.dumps(source_counts.get('raw_matched', {}), ensure_ascii=False)}"
    )
    print(
        f"[{company_id}] analysis_units="
        f"{json.dumps(source_counts.get('analysis_units', {}), ensure_ascii=False)}"
    )
    print(
        f"[{company_id}] document_or_cluster_intelligence_created="
        f"{json.dumps(source_counts.get('document_or_cluster_intelligence_created', {}), ensure_ascii=False)}"
    )
    print(
        f"[{company_id}] used_for_final_profile="
        f"{json.dumps(source_counts.get('used_for_final_profile', {}), ensure_ascii=False)}"
    )
    print(
        f"[{company_id}] news_pipeline="
        f"{json.dumps(source_counts.get('news_pipeline', {}), ensure_ascii=False)}"
    )
    print(
        f"[{company_id}] by_processing_status="
        f"{json.dumps(source_counts.get('by_processing_status', {}), ensure_ascii=False)}"
    )
    print(f"[{company_id}] profile_confidence={quality.get('confidence')}")
    print(
        f"[{company_id}] recency_focus="
        f"{json.dumps({key: recent_period.get(key) for key in ('recency_focus_days', 'recency_focus_start_date', 'recency_focus_policy', 'recency_focus_reason')}, ensure_ascii=False)}"
    )
    print(f"[{company_id}] data_limitations={json.dumps(limitations, ensure_ascii=False)}")
    source_intelligence = profile.pop("_source_intelligence", None)
    if dry_run:
        print(json.dumps(profile, ensure_ascii=False, indent=2))
        return

    if isinstance(source_intelligence, dict):
        _save_source_intelligence(company_id=company_id, source_intelligence=source_intelligence)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _OUTPUT_DIR / f"{company_id}_profile.json"
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[{company_id}] saved={output_path}")


def _save_source_intelligence(*, company_id: str, source_intelligence: dict[str, Any]) -> None:
    units = source_intelligence.get("source_intelligence_units", [])
    if not isinstance(units, list):
        return
    target_dir = _INTELLIGENCE_OUTPUT_DIR / company_id
    target_dir.mkdir(parents=True, exist_ok=True)
    grouped = {
        "news_clusters.json": [
            unit
            for unit in units
            if isinstance(unit, dict) and unit.get("unit_type") == "news_cluster"
        ],
        "dart_ir_documents.json": [
            unit
            for unit in units
            if isinstance(unit, dict) and unit.get("source_type") in {"dart", "ir"}
        ],
        "official_documents.json": [
            unit
            for unit in units
            if isinstance(unit, dict) and unit.get("source_type") == "official"
        ],
        "securities_reports.json": [
            unit
            for unit in units
            if isinstance(unit, dict) and unit.get("source_type") == "securities_report"
        ],
    }
    for filename, items in grouped.items():
        (target_dir / filename).write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"[{company_id}] intelligence_saved={target_dir}")


def _print_summary(results: list[tuple[str, bool, str]]) -> None:
    print("\n=== company profile build summary ===")
    for company_id, ok, reason in results:
        print(f"- {company_id}: {'SUCCESS' if ok else 'FAIL'} ({reason})")


if __name__ == "__main__":
    raise SystemExit(run_peer_profile_cli())


__all__ = [
    "ProfileAgent",
    "PeerProfileAgent",
    "SKAXProfileAgent",
    "SOURCE_INTELLIGENCE_PROMPT",
    "COMPANY_PROFILE_PROMPT",
    "build_selected_sector_config",
    "emit_profile_result",
    "load_company_profile_documents",
    "load_runtime_env",
    "print_selected_sector_config",
    "run_peer_profile_cli",
    "run_skax_profile_cli",
]
