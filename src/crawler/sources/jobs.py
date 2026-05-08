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

from src.config.companies import COMPANY_ALIASES, all_company_aliases
from src.crawler.base import RawArticle
from src.crawler.base_crawler import BaseCrawler

log = logging.getLogger(__name__)

load_dotenv()

WORK24_API_KEY = os.getenv("WORK24_API_KEY")
RETURN_TYPE = os.getenv("WORK24_RETURN_TYPE", "XML")

WORK24_OPEN_API_HOST = "https://www.work24.go.kr"
JOB_POSTING_URL = f"{WORK24_OPEN_API_HOST}/cm/openApi/call/wk/callOpenApiSvcInfo210L01.do"
RECRUIT_NEWS_URL = f"{WORK24_OPEN_API_HOST}/cm/openApi/call/wk/callOpenApiSvcInfo210L21.do"

PEER_COMPANIES = all_company_aliases()
PEER_ALIASES = dict(COMPANY_ALIASES)


class Work24APIError(RuntimeError):
    """고용24 API 응답 오류"""


class JobCrawler(BaseCrawler):
    """고용24 공채속보 기반 Peer사 채용공고 크롤러"""

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

            roles = job.get("roles", [])
            selection_steps = job.get("selection_steps", [])

            role_summaries = _build_role_summaries(roles)
            content = _build_job_content(job, role_summaries, selection_steps)

            if not _has_required_job_fields(job, role_summaries):
                log.warning(
                    "채용공고 필수 필드 누락으로 스킵 | peer_id=%s company=%s title=%s url=%s",
                    self.peer_id,
                    job.get("company", ""),
                    job.get("title", ""),
                    job.get("url", ""),
                )
                continue

            company_ids = (
                [self.peer_id] if self.peer_id else _company_ids_from_peer_company(peer_company)
            )

            articles.append(
                RawArticle(
                    url=job.get("url", ""),
                    title=job.get("title", ""),
                    content=content,
                    published_at=_parse_job_date(job.get("start_date", "")),
                    source_name="work24_job",
                    peer_id=self.peer_id,
                    source_type="job",
                    content_type="api",
                    publisher="고용24",
                    company=company_ids,
                    extra={
                        "emp_seqno": job.get("emp_seqno", ""),
                        "peer_company": peer_company,
                        "company": job.get("company", ""),
                        "job_title": job.get("title", ""),
                        "start_date": job.get("start_date", ""),
                        "end_date": job.get("end_date", ""),
                        "employment_type": job.get("employment_type", ""),
                        "region": job.get("region", ""),
                        "url": job.get("url", ""),
                        "roles": roles,
                        "role_summaries": role_summaries,
                        "selection_steps": selection_steps,
                        "raw": job.get("raw", {}),
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

    return unescape(str(value)).replace("\r", "\n").strip()


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

    text = f"{company} {title} {url}".lower().replace(" ", "")

    for peer in peer_companies:
        if peer.lower().replace(" ", "") in text:
            return peer

    return None


def _matches_peer_id(peer_id: str, peer_company: str) -> bool:
    peer_text = peer_company.lower().replace(" ", "")
    aliases = PEER_ALIASES.get(peer_id, [peer_id])

    return any(alias.lower().replace(" ", "") in peer_text for alias in aliases)


def _company_ids_from_peer_company(peer_company: str) -> list[str]:
    peer_text = peer_company.lower().replace(" ", "")
    if not peer_text:
        return []

    return [
        peer_id
        for peer_id, aliases in PEER_ALIASES.items()
        if any(alias.lower().replace(" ", "") in peer_text for alias in aliases)
    ]


def _build_role_summaries(roles: list[dict]) -> list[dict]:
    summaries = []

    for role in roles:
        summaries.append(
            {
                "name": clean_text(role.get("name", "")),
                "description": clean_text(role.get("description", "")),
                "headcount": clean_text(role.get("headcount", "")),
                "career": clean_text(role.get("career", "")),
                "education": clean_text(role.get("education", "")),
                "qualification": clean_text(role.get("qualification", "")),
                "region": clean_text(role.get("region", "")),
            }
        )

    return summaries


def _build_job_content(
    job: dict,
    role_summaries: list[dict],
    selection_steps: list[dict],
) -> str:
    role_text = "\n".join(
        [
            " / ".join(
                [
                    f"직무명: {role.get('name', '')}",
                    f"직무 설명: {role.get('description', '')}",
                    f"모집 인원: {role.get('headcount', '')}",
                    f"경력: {role.get('career', '')}",
                    f"학력: {role.get('education', '')}",
                    f"자격요건: {role.get('qualification', '')}",
                    f"근무지역: {role.get('region', '')}",
                ]
            )
            for role in role_summaries
        ]
    )

    selection_text = "\n".join(
        [
            " / ".join(
                [
                    f"전형명: {step.get('name', '')}",
                    f"일정: {step.get('schedule', '')}",
                    f"설명: {step.get('description', '')}",
                    f"메모: {step.get('memo', '')}",
                ]
            )
            for step in selection_steps
        ]
    )

    return "\n".join(
        [
            f"회사명: {job.get('company', '')}",
            f"공고명: {job.get('title', '')}",
            f"채용 시작일: {job.get('start_date', '')}",
            f"채용 마감일: {job.get('end_date', '')}",
            f"고용형태: {job.get('employment_type', '')}",
            f"지역: {job.get('region', '')}",
            f"URL: {job.get('url', '')}",
            "",
            "[직무 정보]",
            role_text,
            "",
            "[전형 정보]",
            selection_text,
        ]
    ).strip()


def _has_required_job_fields(job: dict, role_summaries: list[dict]) -> bool:
    if not job.get("company"):
        return False

    if not job.get("url"):
        return False

    if not role_summaries:
        return False

    has_role_name = any(role.get("name") for role in role_summaries)
    has_role_description = any(role.get("description") for role in role_summaries)
    has_headcount = any(role.get("headcount") for role in role_summaries)

    return has_role_name and has_role_description and has_headcount


def _parse_job_date(date_text: str) -> datetime | None:
    text = clean_text(date_text)

    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y.%m.%d", "%y.%m.%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return None


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
        detail = parse_recruit_news_detail(request_recruit_news_detail(emp_seqno))

        if not job.get("url"):
            detail_url = detail.get("detail_url", "")
            homepage = detail.get("homepage", "")
            job["url"] = detail_url or homepage

        return detail

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
