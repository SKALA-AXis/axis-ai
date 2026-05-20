# Supervisor Topology

> 기준: 1단계 데이터 분석 Supervisor / 2단계 데이터 활용 Orchestrator

## 1단계: 데이터 수집·정제·통합·분석·시사점·카드뉴스

```text
수집 데이터
→ 원문 저장
→ 전처리·적합성 판단
→ 기업·섹터·이벤트 매칭
→ raw_articles / 관련 테이블 저장
→ DataAnalysisSupervisorAgent
   → AnalysisInputBundle 구성
   → IssueIntegrationAgent
   → AnalysisAgent
   → ProfileAgent
   → ImplicationAgent
→ AnalysisPackage
→ CardNewsAgent
→ 기존 save_card_news 경로
→ card_news / card_news_articles / evidence_chain 저장
```

## 2단계: 저장 데이터 활용

```text
Raw / 정제 데이터 저장소
+
카드뉴스 / 시사점 저장소
+
ProfileContext
→ DataUsageOrchestrator
   → MixerAgent
   → ITTrendAgent
   → ReportAgent
   → InsightAgent
   → ChatbotAgent
   → KeywordGraphAgent
```

## DataAnalysisSupervisorAgent

`DataAnalysisSupervisorAgent`는 1단계 분석 흐름을 조율하는 Supervisor Agent이다.
직접 통합, 분석, 시사점을 모두 수행하는 Agent가 아니라 하위 Agent들의 실행 순서와
데이터 전달을 관리한다.

주요 책임:
- Raw / 정제 데이터 저장소에서 분석 대상 데이터를 조회한다.
- 뉴스의 경우 `raw_articles.cluster_id` 기준으로 클러스터 전체 기사를 조회한다.
- DART/IR/리포트의 경우 `raw_articles.id` 또는 `document_group_id` 기준으로
  `raw_article_parse_results`, `raw_article_financial_metrics`,
  `raw_article_business_signals` 등을 함께 조회한다.
- 조회한 데이터를 `AnalysisInputBundle`로 구성한다.
- 하위 Agent를 조율한다.
- 최종 결과를 `AnalysisPackage`로 묶어 `CardNewsAgent`에 전달한다.

내부 흐름:

```text
AnalysisInputBundle
→ IssueIntegrationAgent
→ IntegratedIssue
→ AnalysisAgent
→ AnalysisResult
→ ProfileAgent
→ ProfileContext
→ ImplicationAgent
→ ImplicationResult
→ AnalysisPackage
```

## IssueIntegrationAgent

기존 요약 Agent의 역할을 대체하는 이슈 통합 Agent이다.

역할:
- 원문/클러스터/문서/파싱 결과를 하나의 통합 이슈로 정리한다.
- 여러 기사 또는 문서에서 반복되는 사실을 합친다.
- 일부 원문에만 있는 중요한 정보는 보조 fact로 보존한다.
- 수치, 사업 신호, 리스크, 불확실성을 구조화한다.
- 분석 가능한 하나의 이슈 글을 만든다.

출력:

```json
{
  "main_issue": "",
  "integrated_text": "",
  "consolidated_facts": [],
  "key_numbers": [],
  "business_signals": [],
  "representative_sources": [],
  "missing_or_uncertain_points": []
}
```

주의:
- 카드뉴스용 3줄 요약을 만들지 않는다.
- 시사점이나 대응 방향을 만들지 않는다.

## AnalysisAgent

`IntegratedIssue`를 기반으로 전략적 의미를 분석한다.

입력:
- `IntegratedIssue`
- company / sector / event_type metadata

출력:

```json
{
  "strategic_moves": [],
  "market_signals": [],
  "competitive_meaning": "",
  "risk_factors": []
}
```

주의:
- 원문/클러스터/문서 전체를 다시 읽지 않는다.
- 원문 기반 fact 통합은 IssueIntegrationAgent 책임이다.
- AnalysisAgent는 IntegratedIssue 안의 `integrated_text`, `consolidated_facts`,
  `key_numbers`, `business_signals`, `fact_basis`를 근거로 해석한다.

## ProfileAgent

시사점 도출에 필요한 SK AX / Peer사 / 섹터 context를 제공한다.

역할:
- SK AX의 사업군, 역량, 전략 방향 context 제공
- Peer사의 사업군, 주요 역량, 최근 집중 섹터 context 제공
- 해당 이슈가 SK AX의 어떤 사업 방향과 연결되는지 판단할 context 제공

주의:
- 전처리 단계의 기업·섹터·이벤트 매칭과 다르다.
- 기업·섹터 매칭은 raw data에 라벨을 붙이는 기능이다.
- ProfileAgent는 시사점 도출용 context provider이다.

## ImplicationAgent

분석 결과와 프로필 context를 결합해 SK AX 관점의 시사점을 도출한다.

입력:
- `AnalysisResult`
- `ProfileContext`
- `IntegratedIssue`
- 필요 시 `AnalysisInputBundle` / sources

출력:

```json
{
  "implication": "",
  "opportunities": [],
  "threats": [],
  "recommended_actions": [],
  "follow_up_questions": []
}
```

## CardNewsAgent

`AnalysisPackage`를 사용자에게 보여주기 좋은 카드뉴스/API 응답 형태로 재가공한다.

역할:
- 카드 제목 생성
- 카드뉴스용 3줄 요약 생성
- 핵심 포인트 생성
- 시사점 문장 재가공
- sources / validation 구성
- 기존 저장 구조에 맞춘 저장 요청

주의:
- 원문 통합을 수행하지 않는다.
- 전략 분석을 수행하지 않는다.
- 카드뉴스용 3줄 요약은 여기서 생성한다.

## DataUsageOrchestrator

2단계 활용 흐름을 조율하는 Supervisor이다.

DataAnalysisSupervisorAgent와의 차이:
- `DataAnalysisSupervisorAgent`는 하나의 이슈/클러스터/문서를 분석해 카드뉴스를 만드는 1단계 Supervisor이다.
- `DataUsageOrchestrator`는 이미 저장된 결과를 활용해 리포트, 인사이트, 챗봇, 키워드 그래프를 만드는 2단계 Supervisor이다.

지원 Agent:
- `MixerAgent`
- `ITTrendAgent`
- `ReportAgent`
- `InsightAgent`
- `ChatbotAgent`
- `KeywordGraphAgent`

## 내부 DTO 원칙

`src/analysis/models.py`에 있는 모델은 DB schema가 아니다.

허용 내부 DTO:
- `AnalysisInputBundle`
- `IntegratedIssue`
- `AnalysisResult`
- `ProfileContext`
- `ImplicationResult`
- `AnalysisPackage`
- `CardNews`

금지:
- DB schema 임의 변경
- `src/schemas.py` 임의 변경
- repository/save 로직 임의 변경
- 별도 근거 패키지 저장 테이블 추가
