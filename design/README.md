# AXIS AI Agent Architecture

> 기준: Peer사 동향 모니터링 시스템의 2단계 Agent 아키텍처

AXIS AI는 뉴스, DART/IR/공시, 시장·증권 리포트, Peer사 동향, 산업 동향 브리핑을 수집하고,
이를 기반으로 Peer사 동향을 통합·분석한 뒤 SK AX 관점의 시사점을 도출한다.
최종적으로 카드뉴스를 생성하고, 이후 저장된 데이터를 활용해 리포트, 인사이트,
키워드 그래프, 챗봇을 제공한다.

## 전체 단계

```text
1단계
데이터 수집·정제·통합·분석·시사점 도출·카드뉴스 생성

2단계
저장된 데이터 활용·리포트·인사이트·키워드 그래프·챗봇 생성
```

## 1단계 파이프라인

```text
수집 데이터 소스
  ├─ 뉴스
  ├─ DART / IR / 공시
  ├─ 시장·증권 리포트
  ├─ Peer사 동향
  └─ 산업 동향 브리핑
        │
        ▼
데이터 수집 크롤링
        │
        ▼
원문 저장
        │
        ▼
전처리·적합성 판단
        │
        ▼
기업·섹터·이벤트 매칭
        │
        ▼
Raw / 정제 데이터 저장소
        │
        ▼
AnalysisGraphRunner (= 1단계 분석 Pipeline 의 thin wrapper)
        ├─ IssueIntegrationAgent
        ├─ ProfileAgent
        ├─ AnalysisAgent
        └─ ImplicationAgent
        │
        ▼
AnalysisPackage
        │
        ▼
CardNewsAgent
        │
        ▼
카드뉴스 / 시사점 저장소
```

### 데이터 수집 크롤링

뉴스, DART/IR/공시, 시장·증권 리포트, Peer사 동향, 산업 동향 브리핑 등 외부 데이터를 수집한다.
이 단계는 분석 단계가 아니며 원문, URL, 제목, 출처, 발행일, 수집일, `source_type` 같은
원천 데이터를 확보한다.

### 원문 저장

수집된 원문과 기본 메타데이터를 저장한다.

저장 대상 예:
- `title`
- `content`
- `url`
- `source_type`
- `content_type`
- `publisher`
- `published_at`
- `collected_at`
- 회사 후보
- `crawl_status`

이 단계에서는 의미 분석을 수행하지 않는다.

### 전처리·적합성 판단

수집된 데이터가 분석 가능한 상태인지 정리하고 판단한다.

주요 기능:
- 본문 추출
- 관련성 판단
- 중복 제거
- 뉴스 클러스터링
- 노이즈 제거
- 문서 품질 확인
- `processing_status` 관리

관련 위치:
- `src/preprocessing/relevance.py`
- `src/preprocessing/dedup.py`
- `src/preprocessing/preprocessing.py`

### 기업·섹터·이벤트 매칭

수집된 기사/문서가 어떤 기업, 어떤 섹터, 어떤 이벤트 유형에 해당하는지 라벨링한다.

주의:
- 이 단계는 ProfileAgent가 아니다.
- 기업·섹터·이벤트 매칭은 raw data에 분석 가능한 라벨을 붙이는 전처리 기능이다.
- ProfileAgent는 시사점 도출용 context provider이다.

관련 위치:
- `src/preprocessing/classification.py`

### Raw / 정제 데이터 저장소

현재 DB 구조 기준:
- 원문/정제 기본 저장소는 `raw_articles`이다.
- `raw_articles`에는 `source_type`, `content_type`, `company`, `matched_companies`,
  `matched_sectors`, `relevance_score`, `processing_status`, `cluster_id`,
  `is_representative` 등이 존재한다.
- 뉴스 클러스터는 별도 `article_clusters` 테이블이 아니라
  `raw_articles.cluster_id + is_representative`로 표현한다.
- DART/IR/문서형 파싱 결과는 `raw_article_parse_results`에 저장된다.
- DART/IR 등의 분석 fact는 `raw_article_financial_metrics`,
  `raw_article_business_signals`에 저장된다.

중요:
- 새 저장 구조나 새 테이블을 만들지 않는다.
- 분석 실행 시 기존 DB에서 `cluster_id`, `raw_article_id`, `document_group_id` 기준으로
  데이터를 조회해 `AnalysisInputBundle`이라는 런타임 내부 DTO로 묶는다.

## 1단계 Agent

### AnalysisGraphRunner

> 명칭 (외부 리뷰 2026-05-21 R-rename) — multi-agent supervisor pattern 이 아닌
> **LangGraph 기반 고정 순서 DAG pipeline** 의 thin wrapper. 기존 이름
> `DataAnalysisSupervisorAgent` 는 backward-compat 으로 유지.

주요 책임:
- Raw / 정제 데이터 저장소에서 분석 대상 데이터를 조회한다.
- 뉴스의 경우 `raw_articles.cluster_id` 기준으로 클러스터 전체 기사를 조회한다.
- DART/IR/리포트의 경우 `raw_articles.id` 또는 `document_group_id` 기준으로
  `raw_article_parse_results`, `raw_article_financial_metrics`,
  `raw_article_business_signals` 등을 함께 조회한다.
- 조회한 데이터를 `AnalysisInputBundle`로 구성한다.
- 다음 노드 순서로 child agent 를 호출한다 — `IssueIntegrationAgent` → `ProfileAgent` →
  (`AnalysisContextBuilder`) → `AnalysisAgent` → `ImplicationAgent`.
- 최종 결과를 `AnalysisPackage`로 묶어 `CardNewsAgent`에 전달한다 (in-graph `card_writer`
  노드).

흐름 (외부 리뷰 R-1 반영 — issue_integrate 가 먼저):

```text
AnalysisInputBundle
→ ① IssueIntegrationAgent      → IntegratedIssue (main_company 확정)
→ ② ProfileContext Loader      → ProfileContext (Tier A snapshot + Tier B recent)
→ ③ AnalysisContextBuilder     → AnalysisContext (timeline / sector pulse / financial 등)
→ ④ AnalysisAgent              → AnalysisResult (peer 관점)
→ ⑤ ImplicationAgent           → ImplicationResult (SK AX 관점)
→ ⑥ validate                   → ValidationReport (hard / soft + W5-1 metric)
   ├ pass → ⑦ assemble → ⑧ card_writer → card_news INSERT (v2 schema)
   └ fail → human_review (flag only)
```

### IssueIntegrationAgent

원문/클러스터/문서/파싱 결과를 하나의 통합 이슈로 정리한다.

기존 요약 중심 역할을 이 이름으로 재정의한다.
단순히 짧게 요약하는 Agent가 아니라, 여러 원문/문서/파싱 결과를 보고 중복 내용을 합치고
빠지면 안 되는 내용을 보존하며 핵심 사실, 주요 수치, 사업 신호, 근거 출처를 구조화한다.

역할:
- 카드뉴스용 3줄 요약 생성 X
- 단순 요약 X
- 원문 전체 기반 이슈 통합 O
- 중복 내용 제거 O
- 핵심 사실/수치/사업 신호 구조화 O
- 분석 가능한 하나의 통합 이슈 글 생성 O

뉴스 기준:
- `raw_articles.cluster_id = X`인 기사 전체를 조회한다.
- 대표기사만 보지 않고 클러스터 전체 원문을 본다.
- 반복되는 내용은 하나로 합친다.
- 기사별 추가 정보는 누락되지 않게 반영한다.
- 여러 기사에서 공통 확인되는 내용은 `consolidated_facts`로 정리한다.
- 일부 기사에만 있는 정보는 보조 fact 또는 uncertain point로 표시한다.

출력 예:

```json
{
  "main_issue": "삼성SDS의 Agentic AI 기반 업무 자동화 플랫폼 고도화",
  "integrated_text": "삼성SDS는 생성형 AI 기반 업무 자동화 플랫폼을 고도화하면서 기업용 AI Agent 서비스 확대와 클라우드 기반 업무 자동화 적용을 함께 추진하고 있다.",
  "consolidated_facts": [],
  "key_numbers": [],
  "business_signals": [],
  "representative_sources": [],
  "missing_or_uncertain_points": []
}
```

현재 코드 기준:
- `src/agents/issue_integration_agent.py`
- `src/analysis/summarizer.py`

### AnalysisAgent

`IntegratedIssue`만을 기반으로 전략적 의미를 분석한다.

입력:
- `IntegratedIssue`
- company / sector / event_type metadata
- sources

주요 책임:
- 기업의 전략적 움직임 분석
- 섹터 변화 분석
- Peer사 경쟁 구도 해석
- 시장 신호 분석
- 리스크 요인 분석
- 해당 이슈가 단순 정보인지 전략적 변화 신호인지 판단

현재 코드 기준:
- `src/agents/analysis_agent.py`
- `src/analysis/analyzer.py`

주의:
- AnalysisAgent는 원문/클러스터/문서 전체를 다시 읽지 않는다.
- 원문 기반 fact 통합은 IssueIntegrationAgent 책임이다.
- AnalysisAgent는 IntegratedIssue 안의 `integrated_text`, `consolidated_facts`,
  `key_numbers`, `business_signals`, `fact_basis`를 근거로 해석한다.

### ProfileAgent

**핵심 책임 (외부 리뷰 2026-05-21 명시): RDB 백필 데이터 기반 회사 전략 프로필 생성.**

ProfileAgent 는 단순 context provider 가 아니라 **DART / IR / 공식 newsroom / 누적
뉴스 / business_signals / financial_metrics 를 LLM 으로 합성하여 회사별 전략 프로필을
만드는 합성 책임자** 다.

두 단계로 운영 (W2-2 2-tier):
* **Tier A (snapshot 생성)** — 분기 1회 CronJob `axis-cron-profile-refresh` 가 회사별
  방향성 / 주요 사업 / 전략 변화 / 역량 평가 narrative 를 합성하여 `peer_companies.
  profile_snapshot` JSONB (별도 컬럼) 에 저장.
* **Tier B (runtime loader)** — Analysis Pipeline ② 노드 (`profile_context`) 가
  Tier A snapshot 을 load + 최근 30일 business_signals top-3 + 최근 분기
  financial_metrics 보강 (DB query only, LLM X).

출력은 **두 관점으로 분리**:
* `ProfileContext.peer_profiles[peer_id]` — AnalysisAgent 입력 (peer 의 전략·역량 해석).
* `ProfileContext.skax_profile` — ImplicationAgent 입력 (SK AX 의 기회/위협/대응 도출).

주의:
- 전처리의 기업·섹터·이벤트 **매칭** (raw data 라벨링) 과 다르다.
- ProfileAgent 는 그 위에 **회사 전략 합성 narrative** 를 만든다.

현재 코드 기준:
- `src/agents/profile_agent.py` (Tier A 합성)
- `src/services/profile_context_loader.py` (`ProfileContextLoader`, Tier B runtime loader)
- `src/services/skax_profile_context_loader.py`
- `scripts/refresh_peer_profile_snapshots.py` (Tier A CronJob entry)

### ImplicationAgent

분석 결과와 프로필 context를 결합해 SK AX 관점의 시사점을 도출한다.

입력:
- `AnalysisResult`
- `ProfileContext`
- `IntegratedIssue`
- 필요 시 `AnalysisInputBundle` / sources

주요 책임:
- SK AX 관점 시사점 생성
- 기회 요인 도출
- 위협 요인 도출
- 대응 방향 제안
- 후속 모니터링 질문 생성

현재 코드 기준:
- `src/agents/implication_agent.py`
- `src/analysis/implication.py`

### CardNewsAgent

`AnalysisPackage`를 사용자에게 보여주기 좋은 카드뉴스/API 응답 형태로 재가공한다.

주요 책임:
- 카드 제목 생성
- 카드뉴스용 3줄 요약 생성
- 핵심 포인트 생성
- 시사점 문장 재가공
- sources / validation 구성
- 기존 card_news 저장 구조에 맞춰 저장 요청

주의:
- 원문 통합을 수행하지 않는다.
- 전략 분석을 수행하지 않는다.
- 카드뉴스용 3줄 요약은 여기서 생성한다.
- 최종 저장은 기존 `card_news`, `card_news_articles`, `evidence_chain` 구조에 맞춘다.

현재 코드 기준:
- `src/agents/card_news_agent.py`

## 2단계 활용 Agent

2단계는 새 데이터를 수집하는 단계가 아니다.
1단계에서 저장된 Raw / 정제 데이터와 카드뉴스 / 시사점 데이터를 활용한다.

```text
Raw / 정제 데이터 저장소
+
카드뉴스 / 시사점 저장소
+
ProfileContext
        │
        ▼
DataUsageOrchestrator
        ├─ MixerAgent
        ├─ ITTrendAgent
        ├─ ReportAgent
        ├─ InsightAgent
        ├─ ChatbotAgent
        └─ KeywordGraphBuilder
```

### DataUsageOrchestrator

2단계 활용 흐름을 조율하는 dispatcher (DAG 가 아닌 dynamic request-based 라우터).

주요 책임:
- 사용자 요청 또는 스케줄에 따라 필요한 활용 Agent를 호출한다.
- 리포트 요청이면 `ReportAgent` 호출
- 여러 기업 비교 요청이면 `MixerAgent` + `InsightAgent` 호출
- 키워드 변화 요청이면 `KeywordGraphBuilder` 호출
- 질의응답 요청이면 `ChatbotAgent` 호출
- IT 트렌드 연결 요청이면 `ITTrendAgent` 호출

현재 코드 기준:
- `src/pipeline/data_usage_orchestrator.py`

### MixerAgent

여러 카드뉴스, 시사점, 분석 결과를 종합해 기업별/섹터별/기간별 흐름을 비교한다.

현재 코드 기준:
- `src/agents/mixer_analysis_agent.py`

### ITTrendAgent

글로벌 IT 트렌드, 산업 동향, 기술 키워드를 Peer사 동향과 연결해 해석한다.

현재 코드 기준:
- `src/agents/it_trend_agent.py`

### ReportAgent

저장된 카드뉴스, 시사점, 정제 데이터, MixerAgent 결과를 바탕으로 보고서를 생성한다.

현재 코드 기준:
- `src/agents/report_agent.py`

### InsightAgent

여러 카드뉴스와 분석 결과를 기반으로 더 큰 흐름과 전략적 의미를 도출한다.

현재 코드 기준:
- `src/agents/insight_cascade_agent.py`

### ChatbotAgent

저장소 기반 RAG 질의응답 Agent이다.

현재 코드 기준:
- `src/agents/chatbot_agent.py`

### KeywordGraphBuilder

기업별·섹터별 키워드의 등장 빈도, 연결 관계, 변화 흐름을 그래프로 생성한다.

현재 코드 기준:
- `src/agents/keyword_graph_agent.py`

## 내부 DTO / 모델 주의사항

`src/analysis/models.py`의 모델은 DB schema가 아니다.
Agent 간 데이터 전달을 위한 내부 DTO/dataclass이다.

예:
- `AnalysisInputBundle`
- `IntegratedIssue`
- `AnalysisResult`
- `ProfileContext`
- `ImplicationResult`
- `AnalysisPackage`
- `CardNews`

중요:
- `src/schemas.py`는 건드리지 않는다.
- DB schema는 건드리지 않는다.
- repository/save 로직은 건드리지 않는다.
- 기존 `save_card_news` 같은 저장 경로를 그대로 사용한다.
- `CardNews` dataclass가 있더라도 DB `card_news` 테이블 schema가 아니다.
- 최종 저장 시에는 기존 `card_news`, `card_news_articles`, `evidence_chain` 구조에 맞춰 저장한다.
- 근거 패키지처럼 보이는 별도 저장 구조명은 사용하지 않는다.
- 기존 코드 호환이 필요하면 내부 DTO alias 정도만 허용한다.
