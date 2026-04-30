"""채용공고 크롤러"""

import csv
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from html import unescape

import requests
from dotenv import load_dotenv

from src.crawler.base_crawler import BaseCrawler, RawArticle

log = logging.getLogger(__name__)

load_dotenv()


WORK24_API_KEY = os.getenv("WORK24_API_KEY")
RETURN_TYPE = os.getenv("WORK24_RETURN_TYPE", "XML")

WORK24_OPEN_API_HOST = "https://www.work24.go.kr"
JOB_POSTING_URL = (
    f"{WORK24_OPEN_API_HOST}/cm/openApi/call/wk/callOpenApiSvcInfo210L01.do"
)
RECRUIT_NEWS_URL = (
    f"{WORK24_OPEN_API_HOST}/cm/openApi/call/wk/callOpenApiSvcInfo210L21.do"
)

PEER_COMPANIES = [
    "삼성SDS",
    "삼성에스디에스",
    "LG CNS",
    "엘지씨엔에스",
    "현대오토에버",
    "HYUNDAI AUTOEVER",
    "포스코DX",
    "포스코 디엑스",
    "POSCO DX",
    "포스코ICT",
    "POSCO ICT",
    "SK AX",
    "SK C&C",
    "SK주식회사 C&C",
    "에스케이씨앤씨",
]

PEER_ALIASES = {
    "samsung_sds": ["삼성SDS", "삼성에스디에스"],
    "lg_cns": ["LG CNS", "엘지씨엔에스"],
    "hyundai_autoever": ["현대오토에버", "HYUNDAI AUTOEVER"],
    "posco_dx": ["포스코DX", "포스코 디엑스", "POSCO DX", "포스코ICT", "POSCO ICT"],
    "sk_ax": ["SK AX", "SK C&C", "SK주식회사 C&C", "에스케이씨앤씨"],
}


class Work24APIError(RuntimeError):
    """Raised when Work24 returns an application-level API error."""


class JobCrawler(BaseCrawler):
    """Compatibility wrapper for the existing Work24 job crawler functions."""

    async def crawl(self) -> list[RawArticle]:
        if not WORK24_API_KEY:
            log.warning("WORK24_API_KEY 미설정. 채용공고 크롤링 스킵.")
            return []

        try:
            jobs = crawl_peer_recruit_news(max_pages=5, display=100)
        except Work24APIError as exc:
            log.warning("고용24 API 응답 오류. 채용공고 크롤링 스킵 | error=%s", exc)
            return []

        articles: list[RawArticle] = []
        for job in jobs:
            peer_company = job.get("peer_company", "")
            if self.peer_id and not _matches_peer_id(self.peer_id, peer_company):
                continue
            articles.append(
                RawArticle(
                    url=job.get("url", ""),
                    title=job.get("title", ""),
                    content=(
                        f"{job.get('company', '')} "
                        f"{job.get('employment_type', '')} "
                        f"{job.get('region', '')}"
                    ),
                    published_at=None,
                    source_name="work24_job",
                    peer_id=self.peer_id,
                    source_type="job",
                    content_type="api",
                    publisher="고용24",
                    company=[job.get("company", "") or self.peer_id],
                    extra={
                        "peer_company": peer_company,
                        "employment_type": job.get("employment_type", ""),
                        "region": job.get("region", ""),
                        "roles": job.get("roles", []),
                    },
                )
            )
        return articles


def validate_env():
    if not WORK24_API_KEY:
        raise ValueError("WORK24_API_KEY가 .env에 설정되어 있지 않습니다.")


def request_job_page(page_no=1, display=100):
    return request_work24_page(JOB_POSTING_URL, page_no=page_no, display=display)


def request_recruit_news_page(page_no=1, display=100):
    return request_work24_page(RECRUIT_NEWS_URL, page_no=page_no, display=display)


def request_recruit_news_detail(emp_seqno: str):
    return request_work24_page(
        RECRUIT_NEWS_URL,
        call_tp="D",
        extra_params={"empSeqno": emp_seqno},
    )


def request_work24_page(
    url: str,
    page_no=1,
    display=100,
    call_tp: str = "L",
    extra_params: dict[str, str] | None = None,
):
    params = {
        "authKey": WORK24_API_KEY,
        "callTp": call_tp,
        "returnType": RETURN_TYPE,
        "startPage": page_no,
        "display": display,
    }
    if extra_params:
        params.update(extra_params)

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.text


def parse_xml(xml_text):
    root = ET.fromstring(xml_text)
    error = clean_text(root.findtext("error"))
    if error:
        raise Work24APIError(error)

    jobs = []

    for item in root.iter():
        if item.tag.lower() in ["wanted", "job", "item"]:
            job = {child.tag: clean_text(child.text) for child in item}
            if job:
                jobs.append(job)

    if not jobs:
        jobs = parse_fallback(root)

    return jobs


def parse_fallback(root):
    rows = []

    for elem in root:
        row = {}
        for child in elem:
            row[child.tag] = clean_text(child.text)

        if row:
            rows.append(row)

    return rows


def clean_text(value):
    if value is None:
        return ""
    return unescape(value).replace("\r", "\n").strip()


def normalize_job(raw_job):
    return {
        "company": pick_value(
            raw_job,
            ["corpNm", "company", "wantedCompany", "businoNm", "empBusiNm"],
        ),
        "title": pick_value(raw_job, ["wantedTitle", "title", "recrutPbancTtl", "jobTitle"]),
        "start_date": pick_value(raw_job, ["wantedStdt", "startDate", "receiptBeginDt", "regDt"]),
        "end_date": pick_value(raw_job, ["wantedEndt", "closeDt", "receiptEndDt", "endDate"]),
        "employment_type": pick_value(raw_job, ["empTpNm", "employmentType", "hireTypeNm"]),
        "region": pick_value(raw_job, ["region", "workRegion", "workPlcNm", "addr"]),
        "url": pick_value(raw_job, ["wantedInfoUrl", "detailUrl", "url"]),
        "raw": raw_job,
    }


def normalize_recruit_news(raw_job):
    url = pick_value(
        raw_job,
        [
            "wantedInfoUrl",
            "detailUrl",
            "url",
            "empWantedHomepgDetail",
            "empWantedMobileUrl",
        ],
    )
    emp_seqno = pick_value(raw_job, ["empSeqno", "empSeqNo", "recrutPbancSeq"])
    if not url and emp_seqno:
        url = (
            f"{WORK24_OPEN_API_HOST}/cm/openApi/call/wk/"
            f"callOpenApiSvcInfo210D21.do?empSeqno={emp_seqno}"
        )

    return {
        "emp_seqno": emp_seqno,
        "company": pick_value(raw_job, ["empBusiNm", "company", "corpNm", "coNm", "businoNm"]),
        "title": pick_value(
            raw_job,
            ["empWantedTitle", "wantedTitle", "title", "recrutPbancTtl", "jobTitle"],
        ),
        "start_date": pick_value(raw_job, ["empWantedStdt", "wantedStdt", "startDate", "regDt"]),
        "end_date": pick_value(raw_job, ["empWantedEndt", "wantedEndt", "endDate", "closeDt"]),
        "employment_type": pick_value(raw_job, ["empWantedTypeNm", "empTpNm", "hireTypeNm"]),
        "region": pick_value(raw_job, ["workRegion", "region", "workPlcNm", "addr"]),
        "url": url,
        "raw": raw_job,
    }


def parse_recruit_news_detail(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)
    error = clean_text(root.findtext("error"))
    if error:
        raise Work24APIError(error)

    return {
        "homepage": clean_text(root.findtext("empWantedHomepg")),
        "detail_url": clean_text(root.findtext("empWantedHomepgDetail")),
        "selection_steps": [
            {
                "name": clean_text(item.findtext("selsNm")),
                "schedule": clean_text(item.findtext("selsSchdCont")),
                "description": clean_text(item.findtext("selsCont")),
                "memo": clean_text(item.findtext("selsMemoCont")),
            }
            for item in root.findall(".//empSelsListInfo")
        ],
        "roles": [
            {
                "name": clean_text(item.findtext("empRecrNm")),
                "description": clean_text(item.findtext("jobCont")),
                "career": clean_text(item.findtext("empWantedCareerNm")),
                "education": clean_text(item.findtext("empWantedEduNm")),
                "qualification": clean_text(item.findtext("sptCertEtc")),
                "headcount": clean_text(item.findtext("recrPsncnt")),
                "memo": clean_text(item.findtext("empRecrMemoCont")),
                "region": clean_text(item.findtext("workRegionNm")),
            }
            for item in root.findall(".//empRecrListInfo")
        ],
    }


def pick_value(data, keys):
    for key in keys:
        if key in data and data[key]:
            return data[key]
    return ""


def is_peer_company(job, peer_companies):
    company = job.get("company", "")
    title = job.get("title", "")
    url = job.get("url", "")

    text = f"{company} {title} {url}".lower()

    for peer in peer_companies:
        if peer.lower().replace(" ", "") in text.replace(" ", ""):
            return peer

    return None


def _matches_peer_id(peer_id: str, peer_company: str) -> bool:
    peer_text = peer_company.lower().replace(" ", "")
    aliases = PEER_ALIASES.get(peer_id, [peer_id])
    return any(alias.lower().replace(" ", "") in peer_text for alias in aliases)


def crawl_peer_jobs(max_pages=5, display=100, sleep_sec=0.3):
    validate_env()

    results = []

    for page_no in range(1, max_pages + 1):
        xml_text = request_job_page(page_no=page_no, display=display)
        raw_jobs = parse_xml(xml_text)

        if not raw_jobs:
            break

        for raw_job in raw_jobs:
            job = normalize_job(raw_job)
            matched_peer = is_peer_company(job, PEER_COMPANIES)

            if matched_peer:
                job["peer_company"] = matched_peer
                job["crawled_at"] = datetime.now().isoformat(timespec="seconds")
                results.append(job)

        time.sleep(sleep_sec)

    return results


def crawl_peer_recruit_news(max_pages=5, display=100, sleep_sec=0.3):
    validate_env()

    results = []

    for page_no in range(1, max_pages + 1):
        xml_text = request_recruit_news_page(page_no=page_no, display=display)
        raw_jobs = parse_xml(xml_text)

        if not raw_jobs:
            break

        for raw_job in raw_jobs:
            job = normalize_recruit_news(raw_job)
            matched_peer = is_peer_company(job, PEER_COMPANIES)

            if matched_peer:
                detail = fetch_recruit_news_detail(job)
                job.update(detail)
                job["peer_company"] = matched_peer
                job["crawled_at"] = datetime.now().isoformat(timespec="seconds")
                results.append(job)

        time.sleep(sleep_sec)

    return results


def fetch_recruit_news_detail(job: dict) -> dict:
    emp_seqno = job.get("emp_seqno", "")
    if not emp_seqno:
        return {"roles": [], "selection_steps": []}

    try:
        return parse_recruit_news_detail(request_recruit_news_detail(emp_seqno))
    except (requests.RequestException, ET.ParseError, Work24APIError) as exc:
        log.warning(
            "고용24 공채속보 상세 조회 실패 | emp_seqno=%s error=%s",
            emp_seqno,
            exc,
        )
        return {"roles": [], "selection_steps": []}


def save_to_csv(jobs, output_path="peer_jobs.csv"):
    fieldnames = [
        "peer_company",
        "company",
        "title",
        "start_date",
        "end_date",
        "employment_type",
        "region",
        "url",
        "crawled_at",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for job in jobs:
            writer.writerow({key: job.get(key, "") for key in fieldnames})


def main():
    jobs = crawl_peer_recruit_news(max_pages=5, display=100)
    save_to_csv(jobs)

    print(f"수집 완료: {len(jobs)}건")
    print("저장 파일: peer_jobs.csv")


if __name__ == "__main__":
    main()
