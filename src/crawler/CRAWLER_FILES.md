# src/crawler 파일별 동작 문서

작성 기준: 현재 워크스페이스의 `src/crawler/**/*.py` 실제 코드 기준.  
목적: 크롤러 폴더 안의 각 Python 파일이 어떤 입력을 받고, 어떤 단계로 실행되며, 어떤 결과물을 만드는지 운영/개발자가 바로 확인할 수 있게 정리한다.

## 목차

1. [전체 구조와 공통 결과 모델](#전체-구조와-공통-결과-모델)
2. [루트 파일](#루트-파일)
   1. [`__init__.py`](#__init__py)
   2. [`article_filter.py`](#article_filterpy)
   3. [`base.py`](#basepy)
   4. [`base_crawler.py`](#base_crawlerpy)
   5. [`batch_processor.py`](#batch_processorpy)
   6. [`dart_crawler.py`](#dart_crawlerpy)
   7. [`ir_crawler.py`](#ir_crawlerpy)
   8. [`job_crawler.py`](#job_crawlerpy)
   9. [`keyword_crawler.py`](#keyword_crawlerpy)
   10. [`naver_crawler.py`](#naver_crawlerpy)
   11. [`playwright_client.py`](#playwright_clientpy)
   12. [`research_crawler.py`](#research_crawlerpy)
   13. [`result_writer.py`](#result_writerpy)
   14. [`rss_crawler.py`](#rss_crawlerpy)
   15. [`scheduler.py`](#schedulerpy)
   16. [`stock_crawler.py`](#stock_crawlerpy)
3. [`parsers/` 파일](#parsers-파일)
   1. [`parsers/__init__.py`](#parsers__init__py)
   2. [`parsers/content.py`](#parserscontentpy)
   3. [`parsers/dedup.py`](#parsersdeduppy)
   4. [`parsers/link_check.py`](#parserslink_checkpy)
4. [`monitors/` 파일](#monitors-파일)
   1. [`monitors/__init__.py`](#monitors__init__py)
   2. [`monitors/keepalive.py`](#monitorskeepalivepy)
5. [`sources/` 파일](#sources-파일)
   1. [`sources/__init__.py`](#sources__init__py)
   2. [`sources/consensus.py`](#sourcesconsensuspy)
   3. [`sources/dart.py`](#sourcesdartpy)
   4. [`sources/jobs.py`](#sourcesjobspy)
   5. [`sources/kipris.py`](#sourceskiprispy)
   6. [`sources/naver.py`](#sourcesnaverpy)
   7. [`sources/naver_research.py`](#sourcesnaver_researchpy)
   8. [`sources/official.py`](#sourcesofficialpy)
   9. [`sources/rss.py`](#sourcesrsspy)
6. [현재 코드 기준 import/실행 주의점](#현재-코드-기준-import실행-주의점)

## 전체 구조와 공통 결과 모델

`src/crawler`는 크게 네 계층으로 나뉜다.

1. 공통 기반 계층
   - `base.py`: `RawArticle`, 상태 타입, 수집 한도 가드, 재시도 정책을 제공한다.
   - `base_crawler.py`: peer/company/limit_guard를 표준화하는 추상 부모 클래스를 제공한다.
   - `result_writer.py`: 크롤러 결과 객체를 JSONL로 저장한다.
   - `parsers/link_check.py`, `parsers/dedup.py`, `parsers/content.py`: 접근성 검사, URL 중복 제거, HTML 본문 추출을 맡는다.

2. 실제 수집 계층
   - 루트의 `dart_crawler.py`, `ir_crawler.py`, `job_crawler.py`, `keyword_crawler.py`, `research_crawler.py`, `stock_crawler.py`는 로컬 실행 스크립트나 개별 크롤러 실행에 직접 쓰기 쉬운 구현이다.
   - `sources/` 아래에는 `BatchProcessor`가 Track A/B에서 쓰려는 소스별 크롤러가 있다. 단, 현재 import 계약이 일부 깨져 있어 주의가 필요하다.

3. 오케스트레이션 계층
   - `batch_processor.py`: Track A/Track B 수집, 링크 검사, 중복 제거, DB 저장을 연결한다.
   - `scheduler.py`: APScheduler 잡으로 Track A/B와 keepalive를 등록한다.

4. 운영 보조 계층
   - `playwright_client.py`: Playwright 브라우저 공유 클라이언트.
   - `monitors/keepalive.py`: Supabase/Qdrant Cloud 비활성화를 방지하는 ping.

공통 기사형 결과는 대부분 `RawArticle`이다. `RawArticle.to_common_dict()` 또는 `result_writer._to_dict()`를 거치면 다음 형태로 저장된다.

```json
{
  "id": "uuid",
  "source_type": "news|ir|securities_report|dart|job|trend_report|search_trend|social|official",
  "source_name": "source identifier",
  "publisher": "publisher name or null",
  "title": "title",
  "content": "body/snippet/text",
  "url": "source url",
  "url_hash": "md5(url)",
  "published_at": "ISO datetime or null",
  "collected_at": "ISO datetime",
  "company": ["company id or company text"],
  "language": "ko",
  "content_type": "html|pdf|api|rss|text|unknown",
  "crawl_status": "success|failed|skipped",
  "error_message": null,
  "extra": {}
}
```

`stock_crawler.py`만 `RawArticle`이 아니라 `StockCrawlResult`를 반환한다. 이 객체도 `to_common_dict()`를 제공하므로 `result_writer.py`와 같은 저장 경로에서 처리된다. `data` 필드에는 OHLCV 배열이 들어간다.

## 루트 파일

### `__init__.py`

역할:
- `src/crawler`를 Python 패키지로 인식시키는 빈 초기화 파일이다.
- 현재 import 부작용, re-export, 패키지 레벨 상수는 없다.

실행 단계:
1. Python이 `src.crawler` 패키지를 import할 때 파일을 읽는다.
2. 코드가 없으므로 아무 작업도 하지 않는다.

결과물:
- 직접 결과 객체를 만들지 않는다.

예외/주의:
- 내용이 비어 있으므로 패키지 차원의 `from src.crawler import RawArticle` 같은 단축 import는 지원하지 않는다.

자체 검증:
- 파일이 0라인임을 확인했다. 설명 대상은 패키지 초기화 역할뿐이며 생략된 함수/상수는 없다.

### `article_filter.py`

역할:
- 구형 `_crawler` 계열 코드와 신규 코드 양쪽에서 재사용하는 아주 작은 텍스트 정리 유틸이다.
- HTML 태그와 HTML entity를 제거해 제목/본문 스니펫을 일반 문자열로 만든다.

주요 함수:
- `strip_html(text: str | None) -> str`
  - 입력이 `None` 또는 빈 값이면 `""`를 반환한다.
  - `html.unescape()`로 `&quot;`, `&amp;` 같은 entity를 실제 문자로 바꾼다.
  - 정규식 `<[^>]+>`로 HTML 태그를 제거한다.
  - 앞뒤 공백을 제거해 반환한다.

처리 단계:
1. 입력값 존재 여부 확인.
2. HTML entity unescape.
3. 태그 제거.
4. strip 후 반환.

결과물:
- 정제된 문자열.
- `RawArticle`을 직접 만들지는 않는다.

예외/주의:
- 태그 제거는 정규식 기반이라 복잡한 HTML DOM 보존에는 적합하지 않다.
- 본문 추출용이 아니라 제목/요약 문자열 정리용이다.

자체 검증:
- 파일 내 함수는 `strip_html` 하나뿐이다. 입력, 처리, 반환, 한계까지 포함해 설명했다.

### `base.py`

역할:
- 크롤러 공통 데이터 모델과 상태 타입, 재시도 정책, 수집량 제한 가드를 정의한다.

주요 상수/타입:
- `RETRY_POLICY`
  - `timeout`: 기본 HTTP timeout 10초.
  - `source_timeout`: source timeout 재시도 3회, backoff `[10, 60, 300]`.
  - `db_failure`: DB 실패 재시도 3회, backoff `[5, 30, 120]`.
  - `playwright_timeout`: Playwright timeout 재시도 2회, backoff `[15, 60]`.
  - `playwright_blocked`: Playwright 차단 시 재시도 1회, backoff `[300]`.
- `SourceType`
  - 허용 문자열: `news`, `ir`, `securities_report`, `dart`, `job`, `trend_report`, `search_trend`, `social`, `official`.
- `ContentType`
  - 허용 문자열: `html`, `pdf`, `api`, `rss`, `text`, `unknown`.
- `CrawlStatus`
  - 허용 문자열: `success`, `failed`, `skipped`.

`RawArticle` 필드:
- 필수: `url`, `title`, `content`, `source_name`.
- 시간: `published_at`, `collected_at`.
- 식별: `id`, `url_hash`.
- 분류: `source_type`, `content_type`, `publisher`, `company`, `language`.
- 상태: `crawl_status`, `error_message`.
- 확장: `extra`.
- 호환: `peer_id`.

`RawArticle.__post_init__()` 처리 단계:
1. `url_hash`가 비어 있으면 `url`의 MD5 해시를 만든다.
2. `peer_id`가 있고 `company`가 비어 있으면 `company = [peer_id]`로 채운다.
3. `company` 배열의 공백 값과 중복 값을 제거한다.

`RawArticle.to_common_dict()` 결과:
- 공통 JSON envelope로 변환한다.
- `published_at`, `collected_at`은 ISO 문자열로 변환한다.
- `extra`는 그대로 포함한다.

`DailyLimitGuard` 처리:
- `GLOBAL_LIMIT = 5000`.
- `SOURCE_TYPE_LIMITS`는 source type별 기본 제한을 가진다.
- `_maybe_reset()`은 날짜가 바뀌면 카운터를 초기화한다.
- `check(new_count)`는 전역 카운터를 증가시키기 전에 제한 초과 여부를 확인한다.
- `allow(source)`는 source별 제한과 전역 제한을 함께 확인한다.

현재 코드 주의:
- `allow()` 내부에서 `self.SOURCE_LIMITS.get(...)`을 호출하지만 실제 클래스 속성명은 `SOURCE_TYPE_LIMITS`이다. 현재 상태로 `allow()`가 실행되면 `AttributeError`가 발생할 수 있다.
- `batch_processor.py`와 일부 `sources/*`는 `CrawlWindow`를 `base.py`에서 import하려 하지만 현재 `base.py`에는 `CrawlWindow` 클래스가 없다.
- 일부 `sources/*`는 `BaseCrawler`도 `base.py`에서 import하려 하지만 실제 `BaseCrawler`는 `base_crawler.py`에 있다.

자체 검증:
- `RawArticle`, `DailyLimitGuard`, `_dedupe_keep_order`, 타입 Literal, `RETRY_POLICY`를 모두 확인했다.
- 코드상 불일치(`SOURCE_LIMITS`, `CrawlWindow`, `BaseCrawler` 위치)도 생략하지 않고 명시했다.

### `base_crawler.py`

역할:
- 모든 크롤러가 같은 생성자 패턴과 `crawl()` 인터페이스를 갖도록 하는 추상 부모 클래스다.

`BaseCrawler.__init__()` 입력:
- `company`: `None`, 문자열, 문자열 리스트를 허용한다.
- `peer_id`: 구형 peer 기반 코드 호환용.
- `limit_guard`: 외부에서 공유할 `DailyLimitGuard`. 없으면 새로 만든다.

처리 단계:
1. `company is None`이면 `peer_id`가 있을 때 `[peer_id]`로 company 후보를 만든다.
2. `company`가 문자열이면 단일 원소 리스트로 바꾼다.
3. `company`가 리스트면 그대로 사용한다.
4. `_dedupe_keep_order()`로 빈 값과 중복을 제거한다.
5. `self.peer_id`는 명시 `peer_id`를 우선 사용하고, 없으면 `company[0]`을 사용한다.
6. `self.limit_guard`는 전달값 또는 신규 `DailyLimitGuard`.

주요 함수:
- `crawl()`: 추상 async 메서드. 하위 클래스는 `list[RawArticle]` 반환을 구현해야 한다.
- `_is_blocked(status_code)`: HTTP 403, 429를 차단성 응답으로 판단한다.
- `_dedupe_keep_order(values)`: 순서를 유지하며 문자열 리스트의 공백/중복을 제거한다.

결과물:
- 직접 수집 결과를 만들지는 않고, 하위 클래스가 `crawl()`에서 `RawArticle` 리스트를 반환하도록 계약을 제공한다.

현재 코드 주의:
- 일부 파일에서 `super().__init__(peer_id)`처럼 첫 번째 위치 인자로 peer id를 넘긴다. 이 경우 `company=peer_id`, `peer_id=None`으로 해석되지만 내부 로직상 `self.peer_id`가 `company[0]`으로 설정되어 결과적으로 동작한다.
- 명확성을 위해 신규 코드는 `super().__init__(peer_id=peer_id)`가 더 안전하다.

자체 검증:
- 클래스 메서드 3개와 helper 1개를 모두 설명했다.
- 인자 위치 혼동 가능성까지 검토했다.

### `batch_processor.py`

역할:
- Track A와 Track B 크롤러를 묶어 실행하고, 링크 접근성 검사, 중복 제거, DB 저장까지 연결하는 오케스트레이터다.

주요 구성:
- `_Crawlable`: `async crawl() -> list[RawArticle]` 프로토콜.
- `BatchProcessor.__init__()`
  - `DailyLimitGuard` 생성.
  - `DedupStore` 생성.
  - `LinkChecker` 생성.

`run_track_a(keywords, persist=True)` 단계:
1. 함수 내부에서 `sources.naver.NaverNewsCrawler`, `sources.rss.RssCrawler`, `sources.rss.GoogleNewsRssCrawler`를 import한다.
2. `keywords` dict의 각 `peer_id`, keyword list를 순회한다.
3. peer별로 Naver News, RSS, Google News RSS 크롤러를 생성한다.
4. 각 크롤러의 `crawl()`을 await하고 결과를 `articles`에 누적한다.
5. 개별 크롤러 예외는 로그만 남기고 다음 크롤러로 진행한다.
6. `LinkChecker.filter_accessible()`로 접근 가능한 URL만 남긴다.
7. `DedupStore.filter_new()`로 실행 중 이미 본 URL을 제거한다.
8. `persist=True`면 `save_articles(new_articles)`로 DB에 저장한다.
9. raw/accesssible/rejected/new/db_inserted 수를 로그로 남기고 `new_articles`를 반환한다.

`run_track_b(keywords, persist=True, crawl_window=None)` 단계:
1. 함수 내부에서 Track B용 `sources.*` 크롤러들을 import한다.
2. peer별로 DART, 공식 뉴스룸, Jobs 크롤러를 생성해 실행한다.
3. `KiprisCrawler`, `HankyungConsensusCrawler`, `NaverResearchCrawler`는 내부에서 여러 peer를 처리하므로 shared crawler로 1회만 실행한다.
4. Track A와 동일하게 링크 검사, 중복 제거, DB 저장을 수행한다.
5. `new_articles`를 반환한다.

결과물:
- 반환: 신규 `RawArticle` 리스트.
- 저장: `persist=True`일 때 `src.db.article_store.save_articles()`를 통해 DB 저장.

현재 코드 주의:
- 현재 `from src.crawler.base import CrawlWindow`가 실패한다. `base.py`에 `CrawlWindow`가 없기 때문이다.
- Track A/B에서 참조하는 `sources.*` 일부도 현재 import 계약이 깨져 있어 `BatchProcessor`가 정상 import되지 않는다.
- `DailyLimitGuard.allow()`가 `SOURCE_LIMITS`를 찾는 버그가 있어, import를 고쳐도 실제 limit guard 호출 단계에서 추가 오류가 날 수 있다.

자체 검증:
- Track A와 Track B의 크롤러 생성 순서, 후처리 순서, 반환/저장 결과를 모두 포함했다.
- import smoke test에서 실제 실패한 지점을 현재 코드 주의에 반영했다.

### `dart_crawler.py`

역할:
- DART OpenAPI에서 공시 목록을 수집한다.
- 선택적으로 `document.xml` 원문 ZIP을 내려받아 XML/HTML/TXT 본문과 표 구조를 텍스트화한다.
- 루트 파일의 `DartCrawler`는 `run_local_crawler_once.py` 스타일의 개별 peer 실행에 적합한 구현이다.

환경변수:
- `DART_API_KEY`: 필수. 없으면 빈 리스트 반환.
- `DART_CORP_CODE_{PEER_ID.upper()}`: 생성자 `corp_code`가 없을 때 fallback.
- `DART_LOOKBACK_DAYS`: 기본 365.
- `DART_PAGE_COUNT`: 기본 100.
- `DART_FETCH_DOCUMENT`: 기본 false. true면 원문 document.xml 조회.
- `DART_MAX_TEXT_CHARS` 또는 `DART_MAX_DOCUMENT_LENGTH`: 본문 최대 길이. 기본 200000.
- `DART_DISCLOSURE_TYPES`: 기본 `A,B,F`.

주요 상수:
- `CORP_CODE_URL`: 법인코드 ZIP 조회.
- `DART_LIST_URL`: 공시 목록 조회.
- `DART_DOCUMENT_URL`: 공시 원문 ZIP 조회.
- `_DART_STATUS_OK = "000"`, `_DART_STATUS_NO_DATA = "013"`, `_DART_STATUS_BLOCKED = {"010","011","012","020"}`.
- `_DISCLOSURE_TYPE_LABELS`: `A=regular`, `B=material_event`, `F=external_audit`.

`DartCrawler.__init__()` 단계:
1. `BaseCrawler` 초기화.
2. corp code를 생성자, env 순서로 결정한다.
3. corp name 후보 리스트를 저장한다.
4. API key와 lookback/page_count/fetch_document/max length/disclosure types를 환경변수에서 읽는다.

`crawl()` 단계:
1. API key가 없으면 warning 후 `[]`.
2. `corp_code`가 있으면 단일 코드 사용.
3. `corp_code`가 없으면 `_resolve_corp_codes()`로 DART 전체 corp code ZIP에서 이름 매칭.
4. 코드가 하나도 없으면 warning 후 `[]`.
5. corp code별로 `_fetch_disclosures()` 실행.
6. 모든 공시 `RawArticle`을 합쳐 반환.

`_resolve_corp_codes()` 단계:
1. `corp_names`가 없으면 `[]`.
2. `corpCode.xml` ZIP을 `crtfc_key`로 요청한다.
3. ZIP 내부 `CORPCODE.xml`을 읽는다.
4. `corp_name`을 `_normalize_name()`으로 정규화해 생성자 corp name 후보와 비교한다.
5. 매칭된 `corp_code`를 반환한다.
6. 조회/파싱 실패 시 로그 후 `[]`.

`_fetch_disclosures(corp_code)` 단계:
1. 현재 UTC 시각 기준 `lookback_days`만큼 begin/end 범위를 만든다.
2. disclosure type 목록을 순회한다.
3. `_fetch_disclosures_by_type()` 결과를 받는다.
4. `receipt_no` 중복을 제거한다.
5. 최종 공시 article 리스트를 반환한다.

`_fetch_disclosures_by_type()` 단계:
1. DART `/api/list.json` 파라미터를 구성한다.
2. httpx로 요청하고 HTTP 오류를 처리한다.
3. DART status를 판정한다.
   - `013`: 결과 없음으로 정상 처리.
   - blocked set: API 키/요청 제한 오류로 보고 빈 결과.
   - `000`이 아닌 기타 status: warning 후 빈 결과.
4. `list` 항목별로 `rcept_no`, `report_nm`, `rcept_dt`를 읽는다.
5. 날짜 파싱 실패 항목은 skip.
6. 기본 content는 `corp_name`, `flr_nm`, `rm` 조합으로 만든다.
7. `fetch_document=True`이면 `_fetch_document_payload()`로 원문 텍스트를 시도한다.
8. max length 초과 시 truncate한다.
9. `RawArticle`을 생성한다.

`RawArticle` 결과:
- `url`: `https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt_no}`
- `title`: 보고서명.
- `content`: 원문 텍스트 또는 fallback content.
- `source_name`: `dart`.
- `source_type`: `dart`.
- `content_type`: `api`.
- `company`: `[peer_id]`.
- `publisher`: DART corp name.
- `extra`: corp code, receipt no, stock code, report name, receipt date, disclosure type, filer, document fetch 여부, content 길이, 표/이미지 개수, truncation 여부, 수집 시각 등.

원문 파싱 함수:
- `_fetch_document_payload(receipt_no)`: `/api/document.xml`을 호출하고 ZIP 여부를 확인한다.
- `_extract_payload_from_dart_document(content, receipt_no)`: ZIP 내부 XML/HTML/TXT 파일을 순회하며 텍스트를 합친다.
- `_extract_text_payload_from_markup(markup, filename, receipt_no)`: script/style 제거, parser 선택, table/image 카운트, 표 텍스트 보존을 수행한다.
- `_replace_tables_with_text()`: `<table>`을 `[표 n]...[/표 n]` 텍스트 블록으로 치환한다.
- `_table_to_text()`: tr/th/td를 `" | "` 구분 텍스트 행으로 바꾼다.
- `_select_markup_parser()`: 파일 확장자와 head 내용으로 `html.parser` 또는 `xml` 선택.
- `_normalize_document_text()`: CSS성 라인과 불필요한 스타일 텍스트 제거.
- `_decode_bytes()`: `utf-8`, `euc-kr`, `cp949` 순으로 디코딩.

보조 함수:
- `_parse_dart_date()`: `YYYYMMDD`를 UTC datetime으로 변환.
- `_normalize_name()`: 회사명 매칭용으로 공백, `㈜`, `(주)`, `주식회사` 제거 후 lower.
- `_disclosure_content()`: fallback content 구성.
- `_env_bool()`, `_env_list()`: 환경변수 파싱.

예외/주의:
- API key가 없거나 corp code가 없으면 실패가 아니라 skip으로 처리한다.
- DART 원문 fetch는 기본 off다.
- 원문이 ZIP이 아니면 `non_zip_response`로 표시하고 본문 없이 목록 정보만 남길 수 있다.
- 표는 구조화 테이블이 아니라 텍스트 보존 방식이다.

자체 검증:
- 클래스 메서드 5개와 문서 추출 helper 전체를 확인했다.
- 목록 수집, corp code 역조회, 원문 ZIP/BadZip/XML/HTML/TXT fallback, 결과 extra 필드를 모두 반영했다.

### `ir_crawler.py`

역할:
- 각 회사 IR 페이지에서 실적 발표 PDF를 찾고, PDF 본문/페이지 블록/이미지 메타를 추출해 `RawArticle`로 만든다.
- 삼성SDS, LG CNS, 현대오토에버, 포스코DX, SK AX처럼 서로 다른 IR 페이지 구조를 다중 전략으로 처리한다.

환경변수/설정:
- `IR_CONFIG`: `src.config.companies`에서 peer별 IR 페이지, fetch strategy, click fallback을 가져온다.
- `IR_PAGES_{PEER_ID.upper()}`: peer별 페이지 URL override.
- `IR_LOOKBACK_DAYS`: 기본 365.
- `IR_PDF_MAX_TEXT_CHARS`: 기본 200000.

`IRCrawler.__init__()` 단계:
1. `BaseCrawler` 초기화.
2. `IR_CONFIG[peer_id]`를 읽는다.
3. env override 페이지 목록을 읽는다.
4. `ir_pages`, `fetch_strategy`, `click_fallback`, `lookback_days`를 결정한다.
5. `PlaywrightClient`를 생성한다.
6. 설정 로그를 남긴다.

`crawl()` 단계:
1. IR 페이지가 없으면 warning 후 `[]`.
2. httpx client를 열고 페이지별 수집을 수행한다.
3. `fetch_strategy == "playwright_click_download"`이면 `_crawl_by_clicking_downloads()`를 바로 사용한다.
4. 그 외에는 `_fetch_page_html()`로 HTML을 가져오고 `_parse_ir_page()`로 정적 링크를 찾는다.
5. 정적 결과가 없고 `click_fallback=True`이면 클릭 다운로드 fallback을 수행한다.
6. 페이지별 예외는 warning으로 남기고 계속 진행한다.
7. `peer_id == "sk_ax"`이면 `_dedupe_articles_by_period()`로 같은 연/분기 중복을 제거한다.
8. `RawArticle` 리스트 반환.

`_fetch_page_html()` 전략:
- `httpx_first`: httpx로 먼저 가져오고 실패하면 Playwright.
- 그 외 기본: Playwright로 먼저 렌더링하고 실패하면 httpx.

`_parse_ir_page()` 단계:
1. BeautifulSoup으로 HTML 파싱.
2. `_find_static_pdf_links()`로 직접 PDF/download 링크 후보 탐색.
3. SK AX가 아니면 `_find_detail_links()`로 상세 페이지 후보도 탐색.
4. 상세 페이지마다 `_fetch_detail_pdf_links()`로 PDF 링크를 추가 탐색.
5. 후보 triple `(pdf_url, label, detail_url)` 중복 제거.
6. 후보별로 label normalize.
7. `_looks_like_ir_performance_label()`이 false면 skip.
8. script/replay PDF와 non-IR PDF를 skip.
9. `_fetch_pdf_bytes()`로 PDF 다운로드.
10. `_build_ir_article_from_pdf()`로 PDF payload와 날짜를 검증해 article 생성.

클릭 다운로드 fallback:
- `_crawl_by_clicking_downloads(peer_id, page_url, lookback_days)`
  1. 독립 Playwright browser 실행.
  2. 페이지 이동 후 스크롤하여 동적 항목 노출.
  3. `_mark_ir_download_candidates()` JS를 실행해 다운로드 버튼 후보에 `data-axis-ir-download=true` 표시.
  4. 후보를 순회하며 label normalize, 성과자료 여부, 중복 key를 검사.
  5. `page.expect_download()`로 파일 다운로드를 받는다.
  6. PDF signature를 확인한다.
  7. `_build_ir_article_from_pdf()`로 article 생성.

링크 탐색 helper:
- `_find_static_pdf_links()`: LG CNS 전용 탐색과 일반 PDF 탐색을 분기.
- `_find_lg_cns_pdf_links()`: anchor href/text에서 PDF/download/다운로드를 찾고 카드 label을 보강한다.
- `_find_lg_cns_card_label()`: 부모 노드 또는 이전 텍스트에서 실적 발표 label을 찾는다.
- `_find_pdf_links()`: href, onclick, data-url, data-file 등 다양한 속성에서 PDF/download URL을 찾는다.
- `_nearest_static_label()`: 후보 주변의 연도/분기/경영실적 텍스트를 label로 추정한다.
- `_find_detail_links()`: PDF가 아닌 view/detail/read/board 계열 상세 링크를 찾는다.
- `_fetch_detail_pdf_links()`: 상세 HTML을 가져와 `_find_pdf_links()`를 재사용한다.
- `_extract_urls_from_text()`: 문자열 속성 안의 PDF/download URL을 정규식으로 추출한다.
- `_clean_candidate_url()`: javascript, mailto, tel, 제어문자, 과도하게 긴 URL을 제거한다.

label normalize/필터:
- `_normalize_ir_label()`: peer별 normalize 함수로 분기.
- `_normalize_hyundai_label()`: 현대오토에버 `20xx년 n분기 경영실적` 패턴 추출.
- `_normalize_lg_cns_label()`: LG CNS 실적 발표 패턴 추출.
- `_normalize_sk_ax_label()`: SK Inc. Presentation, Earnings Briefing, nQ 패턴 추출.
- `_normalize_generic_ir_label()`: 일반 한국어/영문 실적 자료 패턴 추출.
- `_looks_like_ir_performance_label()`: 이사회/위원회/감사/정관/지배구조 등 governance 문서는 제외하고, 실적/분기/연간/earnings/presentation/results/IR 자료는 통과시킨다.
- `_is_script_or_replay_pdf()`: script/transcript/replay/다시듣기 URL skip.
- `_is_non_ir_pdf()`: ESG, sustainability, audit, annual report, business report 등 비실적 PDF skip.
- `_is_sk_ax_presentation_url()`: SK AX는 `/pres/` 또는 `sk_inc_presentation` PDF만 허용한다.

PDF 처리:
- `_fetch_pdf_bytes()`: PDF content-type 또는 `%PDF` signature 확인.
- `_fetch_pdf_text()`: bytes를 받아 `_extract_pdf_payload()`의 text만 반환한다.
- `_extract_pdf_payload()`: PyMuPDF로 페이지 텍스트, 페이지별 blocks, image count, page count, parsed page count를 만든다.
- `_clean_pdf_text()`: 줄바꿈/공백/한글-영문-숫자 붙어 있는 텍스트를 정리한다.
- `_extract_pdf_page_blocks()`: PyMuPDF block 좌표와 텍스트를 `bbox`, `text`로 저장한다.

날짜/분기 추정:
- `_resolve_ir_date()`: peer, label, detail_url, pdf_url, pdf_text를 사용해 발표 기준일을 찾는다.
- 현대오토에버/SK AX는 PDF 본문에서 먼저 연/분기를 찾는다.
- `_resolve_ir_period_from_listing()`은 label/pdf_url에서 연/분기 또는 연간을 파싱한다.
- `_resolve_ir_period_from_pdf_text()`는 PDF 첫 페이지 또는 앞부분에서 연/분기를 찾고 분기 말일을 기준일로 사용한다.
- `_parse_year_quarter_from_text()`, `_parse_year_annual_from_text()`, `_guess_date()`, `_guess_year_quarter()`, `_month_to_quarter()`, `_quarter_end_date()`, `_normalize_year()`가 세부 파싱을 담당한다.

`_build_ir_article_from_pdf()` 결과:
- 날짜 추정 실패, lookback 초과, PDF 본문 추출 실패는 모두 skip.
- 반환 `RawArticle`
  - `url`: PDF URL.
  - `title`: `_format_ir_title()` 결과.
  - `content`: PDF 텍스트.
  - `source_name`: `ir_pdf`.
  - `source_type`: `ir`.
  - `content_type`: `pdf`.
  - `company`: `[peer_id]`.
  - `extra`: source page, detail URL, PDF URL, 텍스트 길이, lookback, date info, raw label, PDF parse strategy, page/image/block payload, chart/table parse strategy, collected_at 등.

중복 제거:
- `_dedupe_pairs()`: `(url,label)` 중 URL 기준 중복 제거.
- `_dedupe_candidate_triples()`: `(pdf_url,label,detail_url)` 중 PDF URL 기준 중복 제거.
- `_dedupe_articles_by_period()`: SK AX용. 연/분기 date_info가 있으면 `(year, quarter)` 기준, 없으면 URL 기준으로 dedupe 후 최신순 정렬.

예외/주의:
- 이 파일은 많은 사이트별 heuristic을 포함한다. 새 IR 페이지 구조가 바뀌면 label 탐색/날짜 파싱부터 깨질 수 있다.
- 클릭 다운로드 fallback은 Playwright chromium 설치가 필요하다.
- PDF 표/차트는 구조화 추출이 아니라 text blocks와 image count를 보존하는 방식이다.

자체 검증:
- 클래스 흐름, 정적 링크 탐색, 상세 페이지 탐색, 클릭 다운로드 fallback, PDF 다운로드/파싱, 날짜/분기 추정, 필터/중복 제거 helper까지 누락 없이 확인했다.
- 매우 큰 파일이므로 함수군별로 묶어 설명하되, 각 함수의 책임은 모두 반영했다.

### `job_crawler.py`

역할:
- 고용24 OpenAPI의 채용공고/공채속보 XML을 수집한다.
- peer 회사명과 매칭되는 공고만 남기고 직무/전형 상세를 본문으로 구성한다.

환경변수:
- `WORK24_API_KEY`: 필수. 없으면 `JobCrawler.crawl()`은 빈 리스트 반환, standalone 함수 `validate_env()`는 예외 발생.
- `WORK24_RETURN_TYPE`: 기본 `XML`.

주요 URL:
- `JOB_POSTING_URL`: 일반 채용공고 목록 API.
- `RECRUIT_NEWS_URL`: 공채속보 목록/상세 API.

`JobCrawler.crawl()` 단계:
1. `WORK24_API_KEY` 없으면 warning 후 `[]`.
2. `crawl_peer_recruit_news(max_pages=5, display=100)` 호출.
3. 고용24 API 오류는 warning 후 `[]`.
4. 반환된 job dict를 순회한다.
5. `peer_id`가 설정되어 있으면 `_matches_peer_id()`로 peer 회사와 맞는 항목만 통과시킨다.
6. roles와 selection_steps를 읽는다.
7. `_build_role_summaries()`로 직무 요약을 만든다.
8. `_build_job_content()`로 content 텍스트를 만든다.
9. `_has_required_job_fields()`로 필수 필드가 없으면 skip한다.
10. `RawArticle` 생성.

`RawArticle` 결과:
- `url`: 채용 상세 URL.
- `title`: 공고명.
- `content`: 회사명, 공고명, 일정, 고용형태, 지역, 직무 정보, 전형 정보.
- `source_name`: `work24_job`.
- `source_type`: `job`.
- `content_type`: `api`.
- `publisher`: `고용24`.
- `company`: API의 회사명 또는 peer id.
- `extra`: emp seqno, peer company, 회사명, 공고명, 시작/마감일, 고용형태, 지역, URL, roles, role_summaries, selection_steps, raw.

API 요청/파싱 함수:
- `validate_env()`: API key 없으면 `ValueError`.
- `request_job_page()`: 일반 채용공고 목록 요청.
- `request_recruit_news_page()`: 공채속보 목록 요청.
- `request_recruit_news_detail(emp_seqno)`: 공채속보 상세 요청.
- `request_work24_page()`: 공통 requests GET. `authKey`, `callTp`, `returnType`, `startPage`, `display`와 추가 파라미터를 전송한다.
- `parse_xml()`: XML root에서 `error`를 확인하고 `wanted`, `job`, `item` 태그를 dict로 추출한다.
- `parse_fallback()`: 표준 태그가 없으면 root 직계 자식들을 row로 변환한다.
- `parse_recruit_news_detail()`: 상세 XML에서 홈페이지, 상세 URL, 전형 단계, 직무 정보를 추출한다.

정규화/필터 함수:
- `clean_text()`: HTML entity unescape, CR to LF, strip.
- `normalize_job()`: 일반 채용공고 row를 공통 key로 변환.
- `normalize_recruit_news()`: 공채속보 row를 공통 key로 변환하고 URL이 없으면 emp_seqno 기반 상세 API URL 생성.
- `pick_value()`: 후보 key 중 첫 유효값 선택.
- `is_peer_company()`: 회사명/제목/URL에 peer alias가 있는지 검사.
- `_matches_peer_id()`: peer id별 alias로 peer company 문자열을 검사.
- `_build_role_summaries()`: roles에서 name/description/headcount/career/education/qualification/region만 정리.
- `_build_job_content()`: 본문 문자열 생성.
- `_has_required_job_fields()`: 회사, URL, roles, role name, role description, headcount가 있어야 true.
- `_parse_job_date()`: `YYYYMMDD`, `YYYY-MM-DD`, `YYYY.MM.DD`, `YY.MM.DD` 지원.

standalone 수집 함수:
- `crawl_peer_jobs()`: 일반 채용공고 목록을 페이지 단위로 수집하고 peer 매칭 항목만 반환한다.
- `crawl_peer_recruit_news()`: 공채속보 목록을 수집하고, 매칭 항목마다 상세 조회를 붙인다.
- `fetch_recruit_news_detail()`: emp_seqno가 있으면 상세 조회, 실패하면 빈 roles/selection_steps.
- `save_to_csv()`: peer jobs를 CSV로 저장한다.
- `main()`: 공채속보 수집 후 `peer_jobs.csv` 저장.

예외/주의:
- http 요청은 동기 `requests`를 사용한다. `JobCrawler.crawl()`은 async지만 내부 수집은 blocking이다.
- 필수 직무 필드가 엄격해 roles가 부족한 공고는 저장되지 않는다.
- `company` 필드에는 내부 peer id가 아니라 API 회사명이 들어갈 수 있다.

자체 검증:
- 클래스, API 요청, XML 파싱, 상세 조회, 정규화, 필수값 검사, CSV standalone까지 전 함수군을 확인했다.

### `keyword_crawler.py`

역할:
- Naver DataLab Search API에서 섹터별 상대 검색지수를 수집한다.
- 검색지수 급등 후보를 표시하고 `RawArticle` 또는 JSONL/차트로 출력한다.

환경변수:
- `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`: 필수. `load_naver_credentials()`가 없으면 `ValueError`.

주요 상수:
- `API_URL`: `https://openapi.naver.com/v1/datalab/search`.
- `SOURCE_NAME`: `naver_datalab`.
- `CRAWL_TYPE`: `trend_signal`.
- 기본 lookback 30일, time unit `date`.
- DataLab 제약: 요청당 group 최대 5개, group당 keyword 최대 20개.
- peak 후보 기준: ratio 70 이상, 직전 대비 delta 25 이상, gap 3일 이하.
- 기본 섹터 그룹: 보안, 인프라, AX(제조), 수주.

인증/키워드 구성:
- `load_naver_credentials()`: `.env`를 load하고 client id/secret을 strip한다.
- `dedupe_texts()`: 빈 문자열/중복 제거.
- `build_sector_keyword_groups()`: 섹터 keyword group을 DataLab 요청 가능 구조와 metadata로 변환한다.
- `build_default_keyword_groups()`: 기본 섹터 그룹 사용.
- `chunk_keyword_groups()`: 5개 단위로 요청 chunk 분리.
- `get_default_date_range()`: KST 기준 현재 날짜와 lookback 시작일 계산.
- `build_payload()`: start/end/timeUnit/keywordGroups/device/gender/ages를 DataLab payload로 만든다.

API 요청:
- `request_datalab_api(payload, retry_count, timeout)`
  1. 인증 정보를 읽는다.
  2. POST 요청을 보낸다.
  3. 200이면 JSON 반환.
  4. 400/401/403은 상세 로그 후 raise.
  5. 429/5xx는 warning 후 retry 대상이 된다.
  6. timeout/request/json parse 실패를 처리한다.
  7. 재시도 간 `DEFAULT_RETRY_DELAY * attempt`만큼 sleep.
  8. 최종 실패 시 `RuntimeError`.

응답 정규화:
- `normalize_datalab_response(response, collected_at, request_chunk_index)`
  - API response의 `startDate`, `endDate`, `timeUnit`, `results`를 row 배열로 바꾼다.
  - 각 row에는 source/type/collected_at/group_name/keywords/period/ratio/is_peak_candidate/peak/metadata가 들어간다.
  - metadata에는 상대지수이며 실제 검색량이 아니라는 해석 주의가 포함된다.

수집 메인:
- `collect_naver_datalab_trends(...)`
  1. keyword group이 없으면 기본 그룹 생성.
  2. 날짜가 없으면 lookback 기준 날짜 생성.
  3. group chunk별 payload 생성.
  4. API 요청.
  5. normalize 결과를 누적.
  6. group/period 기준 정렬 후 반환.

peak 후보:
- `detect_relative_peak_candidates()`: group별 시계열을 정렬하고 직전 row 대비 ratio/delta/gap 조건을 만족하면 후보 row를 만든다.
- `mark_relative_peak_candidates()`: 원본 row에 `is_peak_candidate`, `peak` 정보를 덧입힌다.

파싱/수치 helper:
- `parse_float()`: float 변환 실패 시 None.
- `date_gap_days()`: `YYYY-MM-DD` period 차이를 일수로 계산.
- `parse_period()`: `YYYY-MM-DD`, `YYYY-MM`, `YYYY`를 KST datetime으로 변환.
- `parse_datetime()`: datetime 또는 ISO 문자열 파싱.
- `parse_chart_date()`: chart용 일자 파싱.
- `moving_average()`: 단순 이동평균.

Article 변환:
- `datalab_rows_to_articles(rows, peer_id="all")`
  - 각 row를 `RawArticle`로 변환한다.
  - `url`: DataLab API URL.
  - `title`: `Naver DataLab 상대 검색지수 - {group_name} ({period})`.
  - `content`: group, period, ratio, keywords 설명.
  - `source_name`: `naver_datalab`.
  - `source_type`: `search_trend`.
  - `content_type`: `api`.
  - `publisher`: `Naver DataLab`.
  - `company`: `[group_name]`.
  - `extra`: 원본 row 전체와 `crawl_type`, `is_actual_search_count=False`.

`KeywordCrawler` 클래스:
- 생성자 입력: peer_id, keywords(list/dict), keyword_groups, lookback_days, time_unit, start_date, end_date.
- `crawl()`:
  1. keyword group 구성.
  2. DataLab row 수집.
  3. peak 후보 marking.
  4. `RawArticle` 리스트로 변환.
- `_build_keyword_groups()`:
  - list keywords면 peer/custom 단일 그룹.
  - dict면 group별 custom 그룹.
  - 없으면 기본 섹터 그룹.

저장/시각화:
- `save_jsonl()`: row dict를 JSONL로 저장.
- `build_output_path(prefix)`: `crawler_results/{prefix}_{YYYYMMDD}.jsonl`.
- `configure_korean_font()`: macOS/Windows/Linux 후보 폰트 설정.
- `save_trend_chart()`: group별 ratio scatter와 moving average line chart를 PNG로 저장.
- `run_keyword_crawler(save_chart=False)`: 수집, peak marking, JSONL 저장, 선택 차트 저장.
- `main()`: logging 설정 후 `run_keyword_crawler(save_chart=True)`.

예외/주의:
- DataLab ratio는 실제 검색량이 아니다. 요청 chunk 내부 최대값 100 기준 상대지수다.
- API 호출은 동기 `requests`를 사용한다.
- peak 후보는 원인 탐색 후보일 뿐 급증 확정 신호가 아니다.

자체 검증:
- 인증, payload, API retry, normalize, peak detection, article 변환, JSONL/차트 저장, 클래스/CLI 실행 흐름까지 모두 포함했다.

### `naver_crawler.py`

역할:
- 구형 네이버 뉴스 API 크롤러다.
- `src/crawler/sources/naver.py`보다 단순하며, 본문 enrichment 없이 API description만 content로 사용한다.

환경변수:
- `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`.

`NaverNewsCrawler.__init__()`:
- `peer_id`, aliases, optional search_queries, display를 받는다.
- client id/secret을 환경변수에서 읽는다.

`crawl()` 단계:
1. client id가 없으면 warning 후 `[]`.
2. `search_queries`가 있으면 그대로 사용하고, 없으면 aliases를 query spec으로 바꾼다.
3. query별 `_fetch()` 실행.
4. query별 예외는 로그 후 계속.
5. article 리스트 반환.

`_fetch(query, sector)` 단계:
1. Naver News Search API GET.
2. query/display/sort=date 전송.
3. 403/429면 warning 후 `[]`.
4. HTTP 오류 raise.
5. `items`를 `RawArticle` 리스트로 변환한다.

의도된 결과:
- `url`: item link.
- `title`: HTML 제거 title.
- `content`: HTML 제거 description.
- `published_at`: RFC 2822 pubDate.
- `source_name`: `naver_news`.
- `peer_id`: 생성자 peer id.

현재 코드 주의:
- `RawArticle(...)` 생성 시 `sector=...`, `search_query=...`를 직접 넘긴다. 현재 `RawArticle` dataclass에는 해당 필드가 없으므로 실행 시 `TypeError: unexpected keyword argument`가 발생할 수 있다.
- 이 파일은 구형 호환용으로 보이며, 운영 Track A 구현은 `sources/naver.py` 쪽 의도가 더 강하다. 단, 현재 `sources/naver.py`도 import 계약 문제가 있다.

자체 검증:
- 생성자, crawl, _fetch의 모든 단계와 현재 실행 리스크를 확인했다.

### `playwright_client.py`

역할:
- Playwright Chromium 브라우저를 공유해 동적 페이지 HTML/텍스트를 수집한다.
- 공식 뉴스룸, 컨센서스, 리서치 같은 SPA/동적 페이지용 공통 클라이언트다.

주요 상수:
- `_MEDIA_SELECTORS`: domain별 본문 selector. hankyung, zdnet, etnews, mk, inews24, bloter 등을 포함한다.
- `_USER_AGENT`: Chrome 계열 UA.
- `_STEALTH_SCRIPT`: `navigator.webdriver`, `navigator.plugins`, `window.chrome` 값을 조작해 자동화 탐지를 일부 완화한다.

`PlaywrightClient.get_browser()`:
1. class 변수 `_browser`가 없거나 연결이 끊겼는지 확인한다.
2. `async_playwright().start()` 실행.
3. Chromium을 headless, no-sandbox, automation controlled 비활성화 옵션으로 launch한다.
4. browser 인스턴스를 공유한다.

`new_page(stealth=True)`:
1. 공유 browser를 가져온다.
2. user agent, viewport, locale을 지정한 context를 만든다.
3. stealth가 true면 init script를 주입한다.
4. page를 yield한다.
5. finally에서 context를 close한다.

`fetch_html(url, wait_selector=None)`:
1. 새 page 생성.
2. `page.goto(..., wait_until="networkidle", timeout=15000)`.
3. wait selector가 있으면 `state="attached"`로 최대 10초 대기.
4. `page.content()` 반환.
5. 실패 시 warning 후 `""`.

`fetch_markdown(url, wait_selector=None)`:
1. page goto와 wait selector는 `fetch_html`과 동일.
2. HTML을 가져온 뒤 `_html_to_text()`로 본문 텍스트를 추출한다.
3. 실패 시 warning 후 `""`.

`_html_to_text(html, url)`:
1. URL domain을 구한다.
2. domain별 selector가 있으면 해당 node 텍스트를 우선 반환한다.
3. 실패하면 `readability.Document`로 본문 추출.
4. readability 실패 시 body 전체 텍스트를 최대 3000자 반환.

결과물:
- HTML 문자열 또는 본문 텍스트 문자열.
- 직접 `RawArticle`을 만들지는 않는다.

예외/주의:
- browser는 공유하지만 context/page는 요청마다 새로 만든다.
- Playwright browser shutdown 함수는 이 파일에 없다. 프로세스 종료 시 정리되는 구조다.
- `fetch_markdown` 이름은 markdown을 말하지만 실제 반환은 plain text에 가깝다.

자체 검증:
- browser lifecycle, context/page lifecycle, HTML/text 추출과 fallback을 모두 확인했다.

### `research_crawler.py`

역할:
- 네이버 증권 기업 리서치 페이지에서 특정 peer의 analyst report PDF를 수집한다.
- 루트 파일의 `NaverResearchCrawler`는 peer 단위 실행용 구현이다.

환경변수:
- `NAVER_RESEARCH_LOOKBACK_DAYS`: 기본 3.
- `RESEARCH_PDF_MAX_TEXT_CHARS`: 기본 200000.

설정:
- `NAVER_RESEARCH_URL`: `company_list.naver?searchType=itemCode&itemCode={item_code}`.
- item code는 `src.config.companies.NAVER_ITEM_CODES`에서 가져온다.

`NaverResearchCrawler.__init__()`:
1. `BaseCrawler` 초기화.
2. 생성자 item_code가 있으면 사용, 없으면 peer id로 item code 조회.
3. lookback days 결정.

`crawl()` 단계:
1. item code 없으면 warning 후 `[]`.
2. company research URL 구성.
3. `_fetch(url, report_type="company_report")` 실행.

`_fetch()` 단계:
1. httpx client로 URL GET.
2. `table.type_1 tr` rows 선택.
3. 각 row를 `_parse_row()`에 넘긴다.
4. article이 있으면 누적 후 반환.

`_parse_row()` 단계:
1. `td`가 3개 미만이면 skip.
2. title anchor와 PDF anchor 찾기.
3. 증권사명, report title, report date 파싱.
4. lookback보다 오래된 report는 skip.
5. PDF URL이 있으면 `_fetch_pdf_text()`로 본문 추출.
6. PDF 본문 실패는 warning만 남기고 article은 생성한다.
7. `RawArticle` 생성.

`RawArticle` 결과:
- `url`: PDF URL 또는 base URL.
- `title`: `[{firm}] {report_title}`.
- `content`: PDF 텍스트.
- `source_name`: `naver_research`.
- `source_type`: `securities_report`.
- `content_type`: PDF 있으면 `pdf`, 없으면 `html`.
- `publisher`: 증권사명.
- `company`: `[peer_id]`.
- `extra`: report_type, firm, pdf_url, pdf_text_chars, lookback_days, item_code.

PDF helper:
- `_fetch_pdf_text(client, pdf_url)`: PDF GET 후 `_extract_pdf_text()`.
- `_extract_pdf_text(pdf_bytes)`: PyMuPDF로 페이지 텍스트를 누적하고 max chars에서 중단.
- `_parse_report_date(date_text)`: `%y.%m.%d`, `%Y.%m.%d`, `%Y-%m-%d`.

자체 검증:
- peer item code, HTML row parse, PDF fetch/extract, lookback filter, 결과 extra를 모두 확인했다.

### `result_writer.py`

역할:
- 크롤러 결과를 로컬 JSONL 파일로 저장한다.
- `RawArticle`, `StockCrawlResult`, legacy 객체를 모두 어느 정도 수용한다.

`save_crawler_results(articles, source_name, output_dir=DEFAULT_RESULTS_DIR)` 단계:
1. output directory를 생성한다.
2. 파일명을 `{source_name}_{YYYYMMDD_HHMMSS}.jsonl`로 만든다.
3. articles를 순회한다.
4. `_to_dict(article)` 결과를 `ensure_ascii=False` JSON 한 줄로 쓴다.
5. 저장 경로를 반환한다.

`_to_dict(article)` 단계:
1. article에 `to_common_dict`가 있으면 그 결과를 그대로 반환한다.
2. 없으면 `extra`, `metadata`를 dict로 읽는다.
3. metadata가 있고 extra가 비어 있으면 metadata를 extra로 사용한다.
4. published_at/collected_at은 ISO 문자열로 변환한다.
5. id, source_type, source_name, publisher, title, content, url, url_hash, company, language, content_type, crawl_status, error_message, source_tier, credibility_score, extra를 dict로 구성한다.
6. peer_id가 있고 company가 비어 있으면 company를 `[peer_id]`로 채운다.

결과물:
- JSONL 파일. 한 줄이 하나의 article/document다.

현재 코드 주의:
- `run_local_crawler_once.py`는 `save_crawler_results(..., peer_aliases=PEER_ALIASES)`를 호출하지만 현재 함수 시그니처에는 `peer_aliases` 인자가 없다. 해당 runner 저장 단계에서 `TypeError`가 발생할 수 있다.

자체 검증:
- 저장 파일명, directory 생성, to_common_dict 우선 경로, legacy fallback 경로, runner 시그니처 mismatch를 확인했다.

### `rss_crawler.py`

역할:
- 구형 RSS 피드 크롤러다.
- 기본 RSS source와 선택적 Google News RSS query를 통해 feed entries를 `RawArticle`로 만든다.

상수:
- `RSS_SOURCES`: etnews, zdnet 기본 RSS.
- `GOOGLE_NEWS_RSS_URL`: Google News RSS query URL.

`RssCrawler.__init__()`:
- `peer_id`, aliases, optional search_queries, max_entries_per_query, include_google_news를 받는다.

`crawl()` 단계:
1. `_source_urls()`로 RSS URL dict를 만든다.
2. 각 feed를 `feedparser.parse()`로 파싱한다.
3. feed entries를 `max_entries_per_query`만큼 순회한다.
4. entry link/title/summary/published_at/source_name 등을 `RawArticle`로 만든다.
5. source별 예외는 error 로그 후 계속한다.

`_source_urls()` 단계:
1. 기본 RSS source를 먼저 넣는다.
2. search_queries가 없으면 aliases를 `"alias"` query spec으로 만든다.
3. `include_google_news=False`이면 기본 RSS만 반환한다.
4. true면 query별 Google News RSS URL을 추가한다.

`_parse_published_at(entry)`:
- `published_parsed` 또는 `updated_parsed`를 UTC datetime으로 변환한다.

현재 코드 주의:
- `RawArticle(...)`에 `sector`, `search_query`를 직접 넘긴다. 현재 `RawArticle`에는 해당 필드가 없어 실행 시 TypeError가 날 수 있다.
- 기본 RSS entries는 peer keyword 필터링 없이 source별 상위 N개를 가져온다.
- 운영 Track A 의도는 `sources/rss.py` 쪽이 더 최신 구현이다. 단, 현재 `sources/rss.py`도 import 계약 문제가 있다.

자체 검증:
- URL 구성, Google News 옵션, feed parse, 결과 생성, 현재 TypeError 리스크를 확인했다.

### `scheduler.py`

역할:
- APScheduler `AsyncIOScheduler`를 구성하고 Track A, Track B, keepalive job을 등록한다.

`build_scheduler()` 단계:
1. `Asia/Seoul` timezone으로 scheduler 생성.
2. `BatchProcessor` 생성.
3. Track A job 등록.
   - trigger: 1시간 interval.
   - kwargs: `{"keywords": PEER_ALIASES}`.
   - id: `track_a_crawl`.
   - max_instances 1, coalesce true, misfire grace 600초.
4. Track B job 등록.
   - trigger: 매일 02:00 cron.
   - kwargs: `{"keywords": PEER_ALIASES}`.
   - id: `track_b_crawl`.
   - max_instances 1, coalesce true, misfire grace 3600초.
5. keepalive job 등록.
   - trigger: 24시간 interval.
   - id: `cloud_keepalive`.
6. job 수 로그 후 scheduler 반환.

결과물:
- 설정된 `AsyncIOScheduler` 객체.
- 실제 start/shutdown은 호출자가 수행한다.

현재 코드 주의:
- `scheduler.py`는 `BatchProcessor` import 과정에서 `CrawlWindow` import 오류가 발생한다. 현재 상태로는 모듈 import가 실패한다.
- Track A job name에는 BigKinds가 언급되지만 현재 `BatchProcessor.run_track_a()`는 Naver/RSS/Google News만 사용한다.

자체 검증:
- 등록되는 job 3개와 각 trigger/kwargs/운영 옵션, import 실패 상태를 확인했다.

### `stock_crawler.py`

역할:
- 한국 상장 주식의 일별 OHLCV와 선택적 실시간성 quote를 수집한다.
- 네이버 금융 차트 XML을 historical source로 사용하고, 네이버 polling JSON을 realtime source로 사용한다.
- `RawArticle`이 아닌 `StockCrawlResult`를 반환하지만 `to_common_dict()`를 제공해 저장 파이프라인과 호환된다.

지원 대상:
- 기본 CLI 대상: `src/config/companies.py`의 `NAVER_ITEM_CODES`에 등록된 모든 company.
- 현재 config 기준: `samsung_sds(018260)`, `lg_cns(064400)`, `hyundai_autoever(307950)`, `posco_dx(022100)`, `sk_ax(034730)`.
- company id, company alias, Naver item code를 모두 입력으로 받을 수 있다.
- 종목 표시명은 company config의 `naver_item_name_ko`를 우선 사용하고, 없으면 `name_ko`를 사용한다.
- config에 없는 6자리 임의 KRX ticker도 `krx_{ticker}` key로 받을 수 있다.

환경변수:
- `STOCK_LOOKBACK_DAYS`: 기본 30.

주요 dataclass:
- `StockTarget`: key, ticker, company_name, exchange, aliases.
- `StockOHLCVRecord`: date, open, high, low, close, volume.
- `StockCrawlResult`: 공통 envelope와 `data: list[StockOHLCVRecord]`를 가진다.

`StockCrawlResult.to_common_dict()` 결과:
- 공통 필드: id, schema_name, source_type, source_name, publisher, title, content, ticker, source, url, url_hash, published_at, collected_at, company, peer_id, language, content_type, crawl_status, error_message.
- `extra`: `_public_extra()`로 내부용 source_type/content_type 등은 제거하고 target/requested_window/coverage/validation/realtime만 노출한다.
- `data`: OHLCV dict 배열.

`StockCrawler.__init__()` 입력:
- `targets`: `list[StockTarget]`, 문자열, None.
- `peer_id`: run_local style peer id.
- `item_code`: 종목코드 override.
- `start_date`, `end_date`, `lookback_days`.
- `include_realtime`: 기본 true.
- `missing_policy`: `drop` 또는 `previous_close`.
- `request_delay`, `timeout`, `max_count`.

초기화 단계:
1. `targets`가 문자열이면 peer_id로 해석한다.
2. `_resolve_init_targets()`로 target list를 결정한다.
3. start_date가 없으면 KST 현재일 기준 lookback days를 사용한다.
4. end_date가 없으면 KST 현재일.
5. start > end면 `ValueError`.
6. missing policy가 허용값이 아니면 `ValueError`.

target resolution:
- `_resolve_init_targets()`는 explicit target 우선.
- `TARGETS`는 하드코딩 목록이 아니라 `COMPANIES`와 `NAVER_ITEM_CODES`에서 생성한다.
- `DEFAULT_TARGET_KEYS`는 `NAVER_ITEM_CODES.keys()`를 사용한다.
- `StockTarget.company_name`은 `naver_item_name_ko -> name_ko -> company_id` 순서로 결정한다.
- peer_id가 있으면 `_target_from_company_id()`로 config target을 만든다.
- `item_code`가 있으면 config의 item code를 override할 수 있다.
- 없으면 `resolve_targets([peer_id])`.
- peer_id가 없고 item_code만 있으면 임시 `krx_{ticker}` target을 만든다.
- peer_id/item_code가 모두 없으면 `resolve_targets(None)`으로 config에 등록된 전체 주식 대상.
- `resolve_targets()`는 alias lookup, 6자리 ticker fallback, 중복 제거를 수행한다.

`crawl()` 단계:
1. httpx AsyncClient 생성.
2. target을 순회한다.
3. 두 번째 target부터 request delay + random jitter를 적용한다.
4. `_crawl_one()` 성공 결과를 누적한다.
5. target별 예외는 전체 실패로 중단하지 않고 `_failed_result()`를 만든다.
6. `StockCrawlResult` 리스트 반환.

`_crawl_one()` 단계:
1. `_chart_url()`로 historical chart XML URL 생성.
2. `_get_with_retry()`로 XML GET.
3. `_parse_naver_chart_xml()`로 `date|open|high|low|close|volume` row 추출.
4. `_validate_and_clean_rows()`로 기간 필터, 타입 변환, sanity check, 결측 처리, 중복 제거.
5. 유효 row가 없으면 `DataIntegrityError`.
6. `include_realtime=True`이면 `_fetch_realtime_quote()` 시도.
7. realtime 실패 시 `_realtime_from_last_ohlcv()`로 마지막 일봉 fallback quote 생성.
8. slim extra 구성.
9. `StockCrawlResult` 반환.

historical parsing:
- `_parse_naver_chart_xml(content)`: EUC-KR decode, XML declaration 제거, `./chartdata/item`의 data 속성을 split한다.
- `_row_to_record()`: date, 가격 4개, volume을 typed record로 변환한다.
- `_assert_sanity()`
  - price는 finite float.
  - volume은 int.
  - `high >= low`, `high >= open/close`, `low <= open/close`, `volume >= 0`.
- `_validate_and_clean_rows()`
  - start/end 범위 밖은 제외.
  - invalid row는 dropped count/sample로 기록.
  - duplicate date는 최신 row로 덮고 count 기록.

결측 정책:
- `drop`: 결측/NaN/null row를 버린다.
- `previous_close`: 이전 종가가 있으면 결측 가격을 이전 종가로 채우고 결측 volume은 0으로 채운다.

realtime parsing:
- `_fetch_realtime_quote()`: polling JSON의 `SERVICE_ITEM`을 읽는다.
- 반환 필드: ticker, company_name, quote_time, market_status, current_price, previous_close, change, change_percent, open, high, low, volume, trading_value, nxt_over_market, source.
- `_normalize_nxt_quote()`: NXT after-market price/open/high/low/change/volume 등을 숫자로 변환한다.

retry:
- `_get_with_retry()`는 0.3, 0.8, 1.6초 backoff를 사용한다.
- 403/429는 blocked/rate-limited 오류로 본다.
- 빈 response도 실패 처리한다.

CLI:
- 필수: `--start-date`, `--end-date`.
- 반복 가능: `--ticker`.
- 옵션: `--missing-policy`, `--no-realtime`, `--delay`, `--timeout`, `--max-count`, `--pretty`, `--output`.
- `serialize_results()`는 결과가 1개면 object, 여러 개면 array로 출력한다.

예외/주의:
- `source_type`은 `market_data`라 `RawArticle.SourceType` Literal에는 없지만 이 객체는 별도 dataclass라 저장은 가능하다.
- run_local의 `LinkChecker`가 API endpoint를 검사하지 않게 내부 extra에는 `source_type=api`를 넣지만, 최종 JSON에서는 `_public_extra()`가 이를 숨긴다.
- 네이버 차트 XML은 EUC-KR 선언이 있어 bytes 상태로 `ElementTree`에 넣으면 깨진다. 현재 코드는 명시 decode 후 파싱한다.

자체 검증:
- target resolution, historical XML, realtime JSON, validation pipeline, CLI, run_local 호환 extra, slim public output까지 모두 확인했다.

## `parsers/` 파일

### `parsers/__init__.py`

역할:
- `src.crawler.parsers`를 패키지로 인식시키는 빈 파일이다.

결과물:
- 없음.

자체 검증:
- 파일이 비어 있어 패키지 역할 외 생략된 내용이 없다.

### `parsers/content.py`

역할:
- HTML 기사 URL에서 본문 텍스트를 추출한다.

상수:
- `_ARTICLE_SELECTORS`: article, article-body, news-content, article_body, article-body id, div.content, div.text, div.article 등.
- `_HEADERS`: `AXIS-Crawler/1.0 (research)`.
- `MIN_PARAGRAPH_LENGTH = 30`.

`fetch_full_content(url)` 단계:
1. httpx AsyncClient를 열고 `RETRY_POLICY["timeout"]`을 적용한다.
2. GET 요청을 보낸다.
3. status가 200이 아니면 `""`.
4. `_extract_text(resp.text)` 반환.
5. 예외는 debug log 후 `""`.

`_extract_text(html)` 단계:
1. BeautifulSoup 파싱.
2. selector 순서대로 첫 matching node를 찾는다.
3. node 내부 `<p>` 텍스트 중 30자 이상 문단만 join한다.
4. join 결과가 200자 이상이면 반환.
5. 실패하면 body 전체 텍스트를 최대 2000자 반환.
6. body도 없으면 `""`.

결과물:
- 본문 문자열.

자체 검증:
- selector 우선 파싱과 body fallback까지 포함했다.

### `parsers/dedup.py`

역할:
- 실행 중 URL 기반 중복 기사를 제거하는 인메모리 LRU 유사 캐시다.

구성:
- `MAX_CACHE_SIZE = 10000`.
- `DedupStore._seen`: OrderedDict.

처리:
- `_hash(url)`: SHA-256(url)의 앞 16자리.
- `is_seen(url)`: hash가 seen에 있는지 확인.
- `mark_seen(url)`: hash를 seen에 넣고 max size 초과 시 가장 오래된 항목 제거.
- `filter_new(articles)`: 아직 보지 않은 article만 반환하고, 반환한 article URL은 즉시 mark한다.

결과물:
- 신규 `RawArticle` 리스트.

주의:
- 프로세스 메모리 기반이므로 재시작하면 초기화된다.
- DB 중복 제거와 별개로 실행 중 중복 방지용이다.

자체 검증:
- hash, seen check, mark, filter, LRU 제거를 모두 확인했다.

### `parsers/link_check.py`

역할:
- 수집된 article URL이 비어 있거나 접근 불가능한지 저장 전에 점검한다.

상수:
- reject status: 401, 403, 404, 408, 410, 429, 451.
- 허용 content type: HTML, XHTML, PDF, octet-stream, MS Office/OpenXML 계열.
- skip source types: `api`, `dart`, `ir`, `search_trend`.
- skip domains: `dart.fss.or.kr`, `opendart.fss.or.kr`.

`LinkChecker.__init__()`:
- timeout 기본 6초.
- concurrency 기본 20.

`filter_accessible(articles)` 단계:
1. articles를 순회하며 `_should_skip_link_check()` 대상은 skipped로 분리한다.
2. skipped article에는 `extra` 또는 `metadata`에 `link_check` status `skipped`, error `api_source`를 기록한다.
3. 나머지는 semaphore concurrency 제한 아래 `_is_accessible()`로 검사한다.
4. ok면 valid, 아니면 rejected.
5. skipped도 valid에 합친다.
6. total/valid/rejected/skipped 로그.
7. `(valid, rejected)` 반환.

`_is_accessible()` 단계:
1. URL이 비어 있으면 failed empty_url.
2. scheme이 http/https가 아니면 failed invalid_scheme.
3. HEAD 요청을 먼저 보낸다.
4. 405/501 또는 400 이상이면 GET으로 재시도한다.
5. HTTPError는 request_error로 failed.
6. status/content-type/final_url을 link_check에 기록한다.
7. reject status면 failed.
8. 2xx/3xx가 아니면 failed.
9. content-type이 허용 prefix가 아니면 failed.
10. 그 외 true.

helper:
- `_should_skip_link_check()`: article.source_type, article.extra/metadata.source_type, domain을 검사한다.
- `_has_http_scheme()`: http/https 여부.
- `_is_allowed_content_type()`: content type prefix 검사.
- `_set_link_check()`: article container에 검사 결과 기록.
- `_article_container()`: metadata가 있으면 metadata, 없으면 extra, 없으면 빈 dict.

주의:
- `RawArticle`에는 `metadata`가 없고 `extra`가 있으므로 대부분 extra에 기록된다.
- `stock_crawler.py`는 최종 JSON에서 내부 `extra.source_type=api`를 숨기지만 LinkChecker 단계에서는 이를 사용한다.

자체 검증:
- skip/HEAD/GET/failure reason/content type 검사와 결과 기록 경로를 모두 확인했다.

## `monitors/` 파일

### `monitors/__init__.py`

역할:
- `src.crawler.monitors` 패키지 초기화 파일이다.
- 현재 내용은 없다.

자체 검증:
- 0라인 파일이다. 생략된 함수/상수 없음.

### `monitors/keepalive.py`

역할:
- Supabase Postgres와 Qdrant Cloud가 무료 티어에서 idle suspend되지 않도록 주기적으로 가벼운 요청을 보낸다.

`keepalive()` 단계:
1. `SessionLocal()`로 DB session을 연다.
2. `SELECT 1` 실행.
3. 성공/실패 로그를 남긴다.
4. `get_qdrant_client()`를 호출한다.
5. `client.get_collections()`로 Qdrant에 ping한다.
6. 성공/실패 로그를 남긴다.

결과물:
- 직접 반환값 없음.
- scheduler가 24시간 간격으로 호출하도록 설계되어 있다.

예외/주의:
- DB ping 실패와 Qdrant ping 실패는 서로 독립적으로 처리된다.
- 실패해도 raise하지 않고 error log만 남긴다.

자체 검증:
- DB/Qdrant 두 경로와 예외 처리 방식을 모두 확인했다.

## `sources/` 파일

### `sources/__init__.py`

역할:
- `src.crawler.sources` 패키지 초기화 파일이다.
- 현재 내용은 없다.

자체 검증:
- 0라인 파일이다. 생략된 함수/상수 없음.

### `sources/consensus.py`

역할:
- 한경 컨센서스 analyst report 목록을 Playwright로 수집한다.
- React SPA로 간주하고 브라우저 렌더링 후 검색/리스트 DOM을 탐색한다.

상수:
- `BASE_URL`: `https://consensus.hankyung.com/analysis/list`.
- `_PEERS`: 삼성SDS, LG CNS, 현대오토에버, 포스코DX.
- `_PEER_KEY_HINTS`: 화면명에서 내부 peer id를 추정하는 hint.

`HankyungConsensusCrawler.__init__()`:
- `PlaywrightClient` 생성.
- `DailyLimitGuard` 생성/주입.

`crawl()` 단계:
1. `limit_guard.allow("consensus")` 확인.
2. `_PEERS`를 순회한다.
3. peer별 `_fetch_peer(peer)` 실행.
4. peer별 예외는 error log 후 계속.
5. article 리스트 반환.

`_fetch_peer(peer_name)` 단계:
1. Playwright page를 열고 BASE_URL로 이동.
2. 검색 input selector 후보를 순회한다.
3. 검색 input을 찾으면 peer name을 입력하고 Enter.
4. 결과 list selector 후보를 순회한다.
5. 결과를 찾지 못하면 warning 후 `[]`.
6. 최대 20개 item을 순회한다.
7. title/firm/date/opinion/target-price/link를 query selector로 읽는다.
8. 상대 URL이면 consensus domain을 붙인다.
9. `RawArticle`을 생성한다.

결과:
- `url`: report/detail href.
- `title`: `[{firm}] {peer_name} {title}`.
- `content`: `투자의견: ... / 목표주가: ...`.
- `source_name`: `hankyung_consensus`.
- `peer_id`: `_resolve_peer(peer_name)`.
- `extra` 의도: type, securities_firm, opinion, target_price, report_date, importance_hint.

현재 코드 주의:
- `RawArticle(...)`에 `metadata={...}`를 넘긴다. 현재 `RawArticle`에는 `metadata` 필드가 없어 실행 시 TypeError가 날 수 있다.
- `limit_guard.allow("consensus")`는 `DailyLimitGuard.allow()`의 `SOURCE_LIMITS` 버그 영향을 받는다.

자체 검증:
- 검색 input, list selector fallback, item field extraction, peer resolution, RawArticle 생성 필드를 확인했다.
- metadata 필드 불일치도 누락하지 않았다.

### `sources/dart.py`

역할:
- Track B용 간단 DART OpenAPI 공시 목록 크롤러다.
- 루트 `dart_crawler.py`보다 기능이 적고, `CrawlWindow` 기반 기간 필터를 전제로 한다.

현재 import 상태:
- `from src.crawler.base import RETRY_POLICY, BaseCrawler, CrawlWindow, DailyLimitGuard, RawArticle`를 사용한다.
- 현재 `base.py`에는 `BaseCrawler`, `CrawlWindow`가 없으므로 모듈 import가 실패한다.

의도된 단계:
1. 생성자에서 peer_id, corp_code, limit_guard, crawl_window를 받는다.
2. `DART_API_KEY`를 읽는다.
3. `crawl()`에서 API key/corp code/limit을 확인한다.
4. `_fetch_disclosures()`에서 window를 만들고 `/api/list.json`을 호출한다.
5. DART status가 `000`이 아니면 공시 없음으로 처리한다.
6. list 항목별로 receipt date를 파싱하고 window 포함 여부를 확인한다.
7. `RawArticle` 생성.

의도된 결과:
- `url`: DART viewer URL.
- `title`: report name.
- `content`: report name과 filer name.
- `source_name`: `dart`.
- `peer_id`: peer id.
- `metadata` 의도: source_type, content_type, crawl window, rcept_no, flr_nm, pblntf_ty.

현재 코드 주의:
- import 실패 외에도 `RawArticle(metadata=...)`는 현재 모델과 맞지 않는다. `extra=...`, `source_type="dart"`, `content_type="api"`로 바꿔야 한다.

자체 검증:
- import 실패 때문에 실제 실행 단계가 아니라 의도된 코드 흐름과 불일치 지점을 모두 분리해 설명했다.

### `sources/jobs.py`

역할:
- Track B용 Saramin API 채용공고 크롤러다.

현재 import 상태:
- `BaseCrawler`, `CrawlWindow`를 `base.py`에서 import하므로 현재 import 실패.

의도된 단계:
1. 생성자에서 peer_id, keywords, limit_guard, crawl_window를 받는다.
2. `SARAMIN_API_KEY`를 읽는다.
3. `crawl()`에서 key/limit/company name을 확인한다.
4. `_fetch()`에서 Saramin `/job-search` API를 호출한다.
5. 응답 `jobs.job` dict 리스트를 `_to_article()`로 변환한다.
6. `window.contains(article.published_at)`으로 기간 필터 후 반환한다.

의도된 결과:
- `url`: Saramin job URL.
- `title`: `[채용] {company} — {title}`.
- `content`: 직종과 고용형태.
- `source_name`: `saramin`.
- `peer_id`: peer id.
- `metadata` 의도: job source/content type, company, job_type, industry, expiration_at, crawl window.

현재 코드 주의:
- import 실패.
- `RawArticle(metadata=...)` 불일치.
- `BaseCrawler.__init__` 시그니처와 생성자 `super().__init__(peer_id, limit_guard, crawl_window)` 의도도 현재 `base_crawler.py`와 맞지 않는다.

자체 검증:
- API 파라미터, 변환 결과, window filter, import/model mismatch를 모두 확인했다.

### `sources/kipris.py`

역할:
- KIPRIS 특허/실용신안 출원 정보를 applicant name 기준으로 수집한다.
- AI 관련 IPC 코드를 별도 flag로 표시한다.

환경변수:
- `KIPRIS_API_KEY`.

상수:
- `KIPRIS_API_BASE`: `http://plus.kipris.or.kr/openapi/rest`.
- `AI_IPC_CODES`: `G06N`, `G06F`, `G06V`, `G06T`, `H04L`.
- applicant mapping은 `KIPRIS_APPLICANTS`에서 가져온다.

`KiprisCrawler.__init__()`:
- API key와 limit_guard를 저장한다.

`crawl()` 단계:
1. API key 없으면 warning 후 `[]`.
2. `limit_guard.allow("kipris")` 확인.
3. applicant별 `_fetch(peer_id, applicant)`를 `asyncio.gather()`로 병렬 실행한다.
4. list 결과는 articles에 합치고 exception은 error log.
5. articles 반환.

`_fetch()` 단계:
1. timeout은 `RETRY_POLICY["source_timeout"]["backoff"][0]`, 즉 10초.
2. applicantNameSearchInfo endpoint 호출.
3. applicant, sortSpec AD, descSort true, numOfRows 50, accessKey 전송.
4. 403/429는 warning 후 `[]`.
5. HTTP 오류 raise.
6. `_parse_xml(resp.text, peer_id)`.

`_parse_xml()` 단계:
1. XML root 파싱.
2. `.//PatentUtilityInfo` item 순회.
3. IPC, invention name, application number/date, applicant, abstract 추출.
4. IPC prefix가 AI 관련 코드면 `is_ai_related=True`.
5. `RawArticle` 생성.

결과:
- `url`: KIPRIS 검색 URL + application no query.
- `title`: `[특허] {InventionName}`.
- `content`: abstract 또는 출원번호/출원인/출원일 fallback.
- `source_name`: `kipris`.
- `peer_id`: peer id.
- `published_at`: application date.
- `metadata` 의도: type, ipc_code, application_no, is_ai_related, importance_hint.

현재 코드 주의:
- `RawArticle(metadata=...)`는 현재 모델에 없는 필드다. 실행 시 TypeError 가능.
- `limit_guard.allow("kipris")`는 현재 `DailyLimitGuard.allow()` 버그 영향을 받는다.

자체 검증:
- 병렬 applicant 수집, XML tag, AI IPC flag, 결과 필드, 현재 모델 mismatch를 확인했다.

### `sources/naver.py`

역할:
- Track A용 네이버 뉴스 API 크롤러다.
- API 결과를 먼저 `RawArticle`로 만들고, 기사 링크 본문을 병렬 enrichment한다.

현재 import 상태:
- `BaseCrawler`를 `base.py`에서 import하므로 현재 모듈 import가 실패한다.

환경변수:
- `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`.

의도된 `NaverNewsCrawler.crawl()` 단계:
1. client id가 없으면 warning 후 `[]`.
2. keyword별 limit guard 확인.
3. `_fetch(keyword)`로 Naver Search API 요청.
4. 모든 article에 대해 `_enrich_with_bodies()` 실행.
5. 본문 200자 이상으로 enrich된 건수를 로그.
6. article 반환.

의도된 `_fetch(keyword)`:
1. `/v1/search/news.json` GET.
2. query, display 20, sort date.
3. 403/429는 warning 후 `[]`.
4. items를 `RawArticle`로 변환.

본문 enrichment:
- `_enrich_with_bodies()`: concurrency 5 semaphore, timeout 6초.
- `_enrich_one()`: article URL GET 후 `_extract_body()`가 200자 이상이면 content 교체.
- `_extract_body()`: 네이버/언론사 selector 시도, readability fallback, body fallback.
- `_strip_html()`: HTML tag/entity 제거.
- `_parse_naver_date()`: Naver pubDate format 파싱.

현재 코드 주의:
- import 실패.
- import를 고쳐도 `BaseCrawler.__init__` 호출이 `super().__init__(peer_id, limit_guard)`라 현재 `base_crawler.py` 시그니처에서는 두 번째 인자가 `peer_id`로 해석된다. `peer_id=peer_id, limit_guard=limit_guard` 형태가 더 안전하다.
- `limit_guard.allow("naver_news")`는 `DailyLimitGuard.allow()` 버그 영향을 받는다.

자체 검증:
- API 수집과 enrichment pipeline, selector/readability fallback, 현재 import/생성자/limit 문제를 모두 확인했다.

### `sources/naver_research.py`

역할:
- Track B용 네이버 금융 리서치 크롤러다.
- 기업 리포트와 IT서비스 섹터 리포트를 Playwright로 수집하고 PDF 본문을 추출한다.

현재 import 상태:
- `CrawlWindow`를 `base.py`에서 import하므로 현재 import 실패.

의도된 설정:
- company item code는 `NAVER_ITEM_CODES`.
- 기업 리포트 URL: `company_list.naver?searchType=itemCode&itemCode={code}`.
- 산업 리포트 URL: `industry_list.naver?searchType=upjongCode&upjongCode=54`.
- lookback 기본 30일.
- PDF text max 12000자.

의도된 `crawl()` 단계:
1. `limit_guard.allow("naver_research")` 확인.
2. company peer별 `_fetch_company(pid)`와 `_fetch_industry()`를 gather로 병렬 실행.
3. 결과 list는 누적하고 예외는 error log.
4. article 반환.

의도된 `_fetch_company()` / `_fetch_industry()`:
1. Playwright page open.
2. URL 이동.
3. `table.type_1` wait.
4. rows를 `_parse_rows()`로 전달.

의도된 `_parse_rows()`:
1. row별 `td` 3개 이상만 처리.
2. title, firm, date, pdf anchor 추출.
3. report date parse.
4. crawl window 외부면 skip.
5. PDF URL이 있으면 `_fetch_pdf_text()`로 본문 추출.
6. `RawArticle` 생성.

결과 의도:
- `url`: PDF URL 또는 industry URL.
- `title`: `[{firm}] {title}`.
- `content`: PDF text.
- `source_name`: `naver_research`.
- `peer_id`: company report는 peer id, industry report는 None.
- `metadata`: source_type, content_type, type, firm, report_date, pdf_url, pdf_text_chars, crawl window.

현재 코드 주의:
- import 실패.
- `RawArticle(metadata=...)` 불일치.
- `limit_guard.allow()` 버그 영향.

자체 검증:
- company/industry 병렬 수집, Playwright table parse, PDF fetch, window filter, import/model mismatch를 확인했다.

### `sources/official.py`

역할:
- 공식 뉴스룸 4 peer 수집기다.
- 삼성SDS는 Playwright HTML, LG CNS는 내부 fingerprint REST API, 현대오토에버/포스코DX는 generic Playwright selector 방식이다.

현재 import 상태:
- `BaseCrawler`를 `base.py`에서 import하므로 현재 import 실패.

상수:
- `_SDS_URL`, `_SDS_BASE`, `_SDS_ARTICLE_RE`.
- `_GENERIC_SOURCES`: 현대오토에버/포스코DX list URL, base, source_name, list selectors.
- `_SDS_BODY_SELECTORS`: 본문 selector 후보.
- `_LGCNS_FP_URL`, `_LGCNS_FETCH_URL`, `_LGCNS_BASE`, `_LGCNS_HEADERS`.

의도된 `OfficialNewsroomCrawler.crawl()`:
1. `limit_guard.allow("official")` 확인.
2. peer id에 따라 `_fetch_sds()`, `_fetch_lgcns()`, `_fetch_generic()` 분기.
3. 미지원 peer는 `[]`.
4. 예외는 error log 후 `[]`.

삼성SDS `_fetch_sds()`:
1. Playwright로 뉴스룸 HTML fetch, `a[href*='/news/']` wait.
2. BeautifulSoup으로 anchor 탐색.
3. `/kr/news/<slug>-YYMMDD.html` 패턴만 통과.
4. "자세히 보기" 제거, 5자 이상 title만 사용.
5. 같은 href 중 가장 긴 title을 선택.
6. `RawArticle` 생성, published_at은 URL slug에서 파싱.
7. `_enrich_sds_bodies()`로 본문 병렬 수집.
8. 본문 200자 미만 article 제거.

LG CNS `_fetch_lgcns()`:
1. fingerprint endpoint 호출.
2. fingerprint 없으면 warning 후 `[]`.
3. fetch endpoint에 cfDirectoryPath, page, itemsPerPage, locale, fp, sortField, latestCount, category filter를 전송.
4. list item에서 title, jcrName, link, date 추출.
5. link가 상대경로면 base URL을 붙인다.
6. `RawArticle` 생성.

generic `_fetch_generic()`:
1. selector 후보를 순회하며 Playwright fetch_html.
2. HTML이 없으면 warning 후 `[]`.
3. selector별 anchor를 추출한다.
4. href/title 유효성, javascript/hash, 외부 domain, 중복을 제거한다.
5. 최대 30개 후보로 제한한다.
6. `RawArticle` 생성.
7. `_enrich_sds_bodies()` 재사용으로 본문 보강.
8. 본문 200자 이상만 반환.

본문/날짜 helper:
- `_enrich_sds_bodies()`: concurrency 4로 httpx GET, selector 본문 추출, readability fallback.
- `_parse_sds_url_date()`: `-YYMMDD.html`을 20YY-MM-DD로 변환.
- `_parse_iso()`: LG CNS 날짜 ISO 또는 `%Y-%m-%d` 파싱.
- `_attr_str()`: BeautifulSoup attribute union을 문자열로 안전 변환.

현재 코드 주의:
- import 실패.
- `BaseCrawler.__init__` 호출 인자 위치가 현재 `base_crawler.py`와 명확히 맞지 않는다.
- `limit_guard.allow()` 버그 영향.
- generic selector는 best-effort라 실제 사이트 구조 변경에 민감하다.

자체 검증:
- SDS, LG CNS, generic 세 경로와 본문 enrichment, 날짜 helper, 현재 import/limit 문제를 모두 확인했다.

### `sources/rss.py`

역할:
- Track A용 RSS 크롤러다.
- 국내 IT 미디어 RSS와 Google News RSS를 수집한다.

현재 import 상태:
- `BaseCrawler`를 `base.py`에서 import하므로 현재 import 실패.

상수:
- `_RSS_HEADERS`: feedparser 기본 UA 차단을 피하기 위한 browser-like UA.
- `RSS_FEEDS`: etnews IT/산업/경제, zdnet, bloter, yonhap tech.
- `GOOGLE_NEWS_RSS`: query 기반 Google News RSS URL.

`_fetch_feed(url)` 의도:
1. httpx GET으로 RSS XML을 가져온다.
2. status가 200이 아니면 warning 후 None.
3. feedparser.parse(r.text) 반환.
4. 예외는 warning 후 None.

`RssCrawler.crawl()` 의도:
1. source feed별 `limit_guard.allow("rss")`.
2. `_fetch_feed()` 호출.
3. feed entries 전체를 순회한다.
4. title + summary/description haystack에 peer keyword가 하나라도 포함되는지 검사한다.
5. match된 entry만 `RawArticle`로 생성한다.
6. source별 total/matched 로그.

`GoogleNewsRssCrawler.crawl()` 의도:
1. keyword별 `limit_guard.allow("google_news")`.
2. query URL 생성.
3. feed entries를 모두 `RawArticle`로 생성한다.
4. keyword별 entries count 로그.

`_parse_entry_date(entry)`:
- entry published 문자열을 `email.utils.parsedate_to_datetime()`으로 변환한다.
- 실패 시 None.

현재 코드 주의:
- import 실패.
- `limit_guard.allow()` 버그 영향.
- Google News RSS는 keyword match를 별도로 하지 않고 query 결과 전체를 수집한다.

자체 검증:
- feed fetch, keyword filter, Google News query path, date parse, import 문제를 모두 확인했다.

## 현재 코드 기준 import/실행 주의점

아래는 `uv run python`으로 각 모듈 import를 확인한 결과와 코드 분석을 합친 운영 주의사항이다.

정상 import되는 주요 루트 모듈:
- `article_filter.py`
- `base.py`
- `base_crawler.py`
- `dart_crawler.py`
- `ir_crawler.py`
- `job_crawler.py`
- `keyword_crawler.py`
- `naver_crawler.py`
- `playwright_client.py`
- `research_crawler.py`
- `result_writer.py`
- `rss_crawler.py`
- `stock_crawler.py`
- `parsers/*`
- `monitors/keepalive.py`
- `sources/consensus.py`
- `sources/kipris.py`

현재 import 실패 모듈:
- `batch_processor.py`: `CrawlWindow`가 `base.py`에 없어 실패.
- `scheduler.py`: `BatchProcessor` import 경유로 실패.
- `sources/dart.py`: `BaseCrawler`, `CrawlWindow`를 `base.py`에서 import해 실패.
- `sources/jobs.py`: `BaseCrawler`, `CrawlWindow`를 `base.py`에서 import해 실패.
- `sources/naver.py`: `BaseCrawler`를 `base.py`에서 import해 실패.
- `sources/naver_research.py`: `CrawlWindow`를 `base.py`에서 import해 실패.
- `sources/official.py`: `BaseCrawler`를 `base.py`에서 import해 실패.
- `sources/rss.py`: `BaseCrawler`를 `base.py`에서 import해 실패.

반복적으로 확인된 모델 불일치:
- 현재 `RawArticle`은 `extra`를 확장 필드로 사용한다.
- 일부 구형/Track 소스 파일은 `metadata={...}`를 넘긴다.
- 일부 구형 루트 파일은 `sector=...`, `search_query=...`를 직접 넘긴다.
- 이 값들은 현재 `RawArticle` dataclass 필드가 아니므로 실행 시 `TypeError`가 날 수 있다.

반복적으로 확인된 수집 한도 가드 문제:
- `DailyLimitGuard.allow()`가 `self.SOURCE_LIMITS`를 찾지만 현재 클래스 속성은 `SOURCE_TYPE_LIMITS`다.
- `allow()`를 호출하는 모든 크롤러는 import가 해결된 뒤에도 이 문제를 만날 수 있다.

문서 작성 자체 검증:
- `find src/crawler -type f -name '*.py'` 기준 31개 파일을 전부 목차와 본문에 포함했다.
- 빈 `__init__.py` 3개도 별도 섹션으로 기록했다.
- 루트 구현과 `sources/` 구현이 중복 이름을 갖는 경우도 각각 별도로 설명했다.
- 각 섹션마다 결과물, 단계, 예외/주의, 자체 검증을 포함했다.
