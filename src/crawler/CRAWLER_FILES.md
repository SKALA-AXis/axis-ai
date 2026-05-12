# src/crawler 현재 구조

작성 기준: 현재 워크스페이스의 `src/crawler`와 루트 실행 파일 기준.

## 핵심 요약

- 운영/파이프라인용 크롤러 구현은 `src/crawler/sources/`를 기준으로 둔다.
- `src/crawler/local/`은 `run_local_crawler_once.py`가 직접 JSON을 만들 때 쓰는 로컬 실행용 사본이다.
- 해외 peer 공식 발표는 `sources/global_newsroom.py`에서 수집하며 `source_type="official"`이다.
- DART는 OpenDART 목록 + 원문 document XML/HTML/TXT payload를 수집한다.
- IR, 네이버 리서치, SPRi/BCG 등 PDF 계열은 `parsers/pdf_payload.py` 기반으로 텍스트, 페이지 블록, 표 후보, 이미지 후보를 extra에 보존한다.

## 폴더 역할

| 경로 | 역할 |
|---|---|
| `base.py` | `RawArticle`, `DailyLimitGuard`, source/content/status 타입 정의 |
| `base_crawler.py` | async crawler 공통 부모 클래스 |
| `batch_processor.py` | DB 저장용 Track A/B 크롤러 오케스트레이션 |
| `scheduler.py` | APScheduler 등록. Track A, global newsroom, 공식 뉴스룸, DART/IR 등 주기 실행 |
| `result_writer.py` | 로컬 JSON 저장 공통 유틸 |
| `article_filter.py` | HTML 태그/entity 정리 유틸 |
| `playwright_client.py` | Playwright 브라우저 공통 클라이언트 |
| `sources/` | 파이프라인용 소스별 크롤러 |
| `local/` | `run_local_crawler_once.py` 전용 로컬 실행 크롤러 |
| `parsers/` | HTML/PDF/link/dedup 파서와 검사 유틸 |
| `monitors/` | keepalive 등 운영 보조 |
| `crawler_results/` | 로컬 크롤링 JSON과 전처리 결과 저장 위치 |

## 공통 결과 모델

기사/문서형 크롤러는 대부분 `RawArticle`을 반환한다. `RawArticle.to_common_dict()` 결과는 대략 다음 형태다.

```json
{
  "id": "uuid",
  "source_type": "news|official|dart|ir|securities_report|trend_report|job|search_trend|social",
  "source_name": "source identifier",
  "publisher": "publisher name or null",
  "title": "title",
  "content": "body/snippet/text",
  "url": "source url",
  "url_hash": "md5(url)",
  "published_at": "ISO datetime or null",
  "collected_at": "ISO datetime",
  "company": ["company id"],
  "company_tier": {"company id": "self|domestic|overseas"},
  "language": "ko|en|...",
  "content_type": "html|pdf|api|text|unknown",
  "crawl_status": "success|failed|skipped",
  "error_message": null,
  "extra": {}
}
```

`stock.py`/`local/stock_crawler.py`는 `RawArticle`이 아닌 시장 데이터 객체를 만들지만 `to_common_dict()`를 제공하므로 같은 JSON 저장 경로에서 처리된다.

## Source Type 라우팅

`run_preprocess_once.py`는 파일명이 아니라 JSON row 내부의 `source_type`을 기준으로 흐름을 나눈다.

| source_type | 예시 source_name | 전처리 흐름 |
|---|---|---|
| `news` | `naver_news` | credibility → relevance → dedup/cluster → classification |
| `official` | `company_news`, `nvidia_official` | credibility → relevance → dedup/cluster → classification |
| `dart` | `dart` | credibility → parser_router → parser_quality → auto relevant 보존 |
| `ir` | `ir_pdf` | credibility → parser_router → parser_quality → auto relevant 보존 |
| `securities_report` | `naver_research` | credibility → parser_router → parser_quality → auto relevant 보존 |
| `trend_report` | `SPRi`, `BCG` | credibility → parser_router → 산업 동향 문서로 보존 |
| `job` | `work24_job` | 구조화 신호로 보존 |
| `search_trend` | `naver_datalab` | 구조화 신호로 보존 |
| `market_data` | `stock_market` | 구조화 신호로 보존 |

`signal_agent.py`는 현재 전처리 흐름에 포함하지 않는다. 추후 급변/급증 탐지 단계에서 사용할 예정이다.

## sources/ 파일

| 파일 | source_type | 설명 |
|---|---|---|
| `sources/naver.py` | `news` | 네이버 뉴스 API + 본문 enrichment. 피어사가 필수로 들어간 기사만 수집/필터링한다. |
| `sources/company_news.py` | `official` | 국내 peer 공식 뉴스룸/보도자료 수집. |
| `sources/global_newsroom.py` | `official` | NVIDIA, Microsoft, Google 등 해외 peer 공식 뉴스룸 수집. 이전 `global_newsroom_crawler.py` 구현이 이 파일로 통합되었다. |
| `sources/dart.py` | `dart` | OpenDART 목록과 원문 document payload 수집. PDF 저장 방식이 아니다. |
| `sources/ir.py` | `ir` | 국내 peer IR PDF 수집 및 PDF payload 생성. |
| `sources/naver_research.py` | `securities_report` | 네이버 증권사 리포트 PDF 링크/본문 payload 수집. |
| `sources/jobs.py` | `job` | 채용공고 구조화 데이터 수집. |
| `sources/keyword.py` | `search_trend` | 네이버 데이터랩 검색 트렌드 수집. |
| `sources/stock.py` | `market_data` | 주가/시장 데이터 수집. |
| `sources/spri.py` | `trend_report` | SPRi 산업 동향 PDF/시각자료 수집. |
| `sources/bcg.py` | `trend_report` | BCG 산업 리포트 수집. |
| `sources/official.py` | `official` | 레거시/보조 공식 뉴스룸 구현. 새 흐름에서는 주로 `company_news.py`와 `global_newsroom.py`를 사용한다. |

## local/ 파일

`local/` 아래 파일은 루트 `run_local_crawler_once.py`가 JSON 결과를 만들 때 직접 import한다.

| 파일 | 대응 source |
|---|---|
| `local/naver_crawler.py` | `naver_news` |
| `local/company_news_crawler.py` | `company_news` |
| `local/dart_crawler.py` | `dart` |
| `local/ir_crawler.py` | `ir` |
| `local/research_crawler.py` | `naver_research` |
| `local/job_crawler.py` | `jobs` |
| `local/keyword_crawler.py` | `naver_datalab` |
| `local/stock_crawler.py` | `stock` |
| `local/bcg_crawler.py` | `bcg` |
| `local/spri_crawler.py` | `spri` |

새로운 운영용 크롤러를 추가할 때는 먼저 `sources/`에 구현하고, 로컬 단독 실행 전용 코드가 꼭 필요할 때만 `local/`에 둔다.

## parsers/ 파일

| 파일 | 역할 |
|---|---|
| `parsers/article_content.py` | HTML 본문 텍스트와 이미지 URL 추출 |
| `parsers/content.py` | readability 기반 본문 추출 |
| `parsers/dedup.py` | URL 기준 중복 제거 |
| `parsers/link_check.py` | 저장 전 URL 접근성 검사. DART/IR/search_trend 등은 검사 skip 대상 |
| `parsers/pdf_payload.py` | PDF 텍스트, 페이지별 블록, 표 후보, 이미지 후보, bbox metadata 추출 |

`bbox`는 PDF 페이지 안에서 텍스트/표/이미지 후보가 위치한 사각형 좌표다. 추후 표/그래프 crop, OCR, layout 기반 분석에 사용할 수 있다.

## 실행 파일과 연결

### run_all_once.py

DB에 수집 결과를 저장하고 전처리까지만 한 번에 실행하는 진입점이다.

```bash
uv run python run_all_once.py
uv run python run_all_once.py --track all --env local
uv run python run_all_once.py --company samsung_sds --company nvidia
```

동작:

1. `run_crawler_once.py`를 subprocess로 실행해 DB `raw_articles`에 RAW 데이터를 저장한다.
2. 크롤러가 성공한 경우에만 `run_pipeline_once.py --preprocess-only`를 이어 실행한다.
3. DB 전처리는 RAW row를 읽어 credibility, source_type 라우팅, relevance/parser quality, dedup/clustering, classification까지만 처리한다.

`--skip-preprocess`를 주면 크롤링까지만 실행한다. 이슈카드/evidence/financial refs는 만들지 않는다.

### run_pipeline_once.py --preprocess-only

DB의 `raw_articles.processing_status='RAW'` row를 전처리까지만 처리한다.

```bash
uv run python run_pipeline_once.py --preprocess-only
uv run python run_pipeline_once.py --preprocess-only --env local
uv run python run_pipeline_once.py --preprocess-only --company samsung_sds
```

실행 범위는 RAW ID 로드, credibility score/grade 보정, source_type별 relevance/parser quality 라우팅, dedup/clustering, classification까지다. issue card/evidence/vector index는 생성하지 않는다.

### run_local_crawler_once.py

로컬 JSON 확인용 진입점이다.

```bash
uv run python run_local_crawler_once.py
uv run python run_local_crawler_once.py --source naver_news --company samsung_sds
uv run python run_local_crawler_once.py --source global_newsroom --company nvidia
uv run python run_local_crawler_once.py --source official --company-tier overseas
uv run python run_local_crawler_once.py --source naver_news,global_newsroom,naver_research
```

특징:

- `--source`는 반복 지정, 쉼표 구분, 공백 포함 쉼표 구분을 모두 지원한다.
- `--company`는 국내 peer와 해외 peer를 모두 받는다.
- `--company-tier`/`--company_tier`는 `domestic`, `overseas`, `self`를 받는다.
- 국내 회사별 source는 국내 peer에만 실행한다.
- `global_newsroom`은 해외 peer에 대해 실행한다. company를 생략하면 설정된 글로벌 회사 전체를 돈다.
- `--source official --company-tier overseas`는 `global_newsroom`으로 해석한다.
- `--source rss`는 하위 호환 alias로 `global_newsroom`을 실행한다.
- 결과는 `src/crawler/crawler_results/{source}_{YYYYMMDD_HHMMSS}.json` 형태로 저장한다.

### run_crawler_once.py

DB 저장용 크롤러 실행 진입점이다.

```bash
uv run python run_crawler_once.py --track a
uv run python run_crawler_once.py --track b
uv run python run_crawler_once.py --track c
uv run python run_crawler_once.py --track all
uv run python run_crawler_once.py --track b --company nvidia
```

Track 구분:

- Track A: 네이버 뉴스와 주가/시장 데이터 중심의 고빈도/실시간성 수집.
- Track B: 증권사 리포트, 공식 뉴스룸, 글로벌 뉴스룸, 채용, 검색 트렌드 등 일중/일간 신호 수집.
- Track C: DART, IR, SPRi, BCG, SK AX 사이트 등 저빈도/무거운 문서형 수집.

Track B/C 기본 company 목록에는 국내 peer와 글로벌 peer가 모두 포함된다. 글로벌 peer는 `global_newsroom` 중심으로 수집된다.

### run_preprocess_once.py

이미 저장된 JSON만 읽어 전처리한다. 크롤링은 실행하지 않는다.

```bash
uv run python run_preprocess_once.py
uv run python run_preprocess_once.py --source-type news
uv run python run_preprocess_once.py --source-type news --source-type official
uv run python run_preprocess_once.py --source-type securities_report
```

결과는 `src/crawler/crawler_results/preprocessed/`에 저장된다.

### run_pipeline_once.py

DB의 `raw_articles`를 읽어 ingestion pipeline을 1회 실행한다. 기본 company 목록은 국내 peer와 글로벌 peer를 포함한다.

```bash
uv run python run_pipeline_once.py
uv run python run_pipeline_once.py --company samsung_sds
uv run python run_pipeline_once.py --company nvidia
```

## 현재 제거/대체된 항목

| 이전 항목 | 현재 상태 |
|---|---|
| `src/crawler/rss_crawler.py` | 제거. RSS 수집 미사용 |
| `src/crawler/sources/rss.py` | 제거. `global_newsroom`으로 대체 |
| `src/crawler/global_newsroom_crawler.py` | 제거. 구현은 `sources/global_newsroom.py`에 직접 통합 |
| 루트 `*_crawler.py` 구현 파일들 | 로컬 실행용은 `src/crawler/local/`로 이동 |

## 점검 명령

```bash
uv run python -m py_compile run_local_crawler_once.py run_crawler_once.py run_pipeline_once.py run_preprocess_once.py src/crawler/sources/*.py src/crawler/local/*.py
uv run pytest tests/test_crawler.py tests/test_parsers.py
```
