from __future__ import annotations

import base64

import pytest

from src.api.chat_schemas import ChatPdfRequest, ChatTurnRequest
from src.api.router import _build_pdf_chat_response, chat_pdf


def test_pdf_chat_response_contains_printable_report_draft() -> None:
    request = ChatTurnRequest(
        message="임원 보고서 형태로 요약해줘",
        conversation_id="11111111-1111-1111-1111-111111111111",
    )
    response = _build_pdf_chat_response(
        request,
        file_name="axis-market.pdf",
        content_type="application/pdf",
        file_hash="a" * 64,
        pdf_payload={
            "text": "\n".join(
                [
                    "[PAGE 1]",
                    "AX 시장은 클라우드와 AI 운영 자동화를 중심으로 "
                    "투자 우선순위가 이동하고 있습니다.",
                    "2026년 계약 규모는 전년 대비 18% 증가했고, "
                    "보안과 운영 책임 요구가 함께 확대됐습니다.",
                    "고객사는 PoC보다 운영 전환 속도와 감사 대응 근거를 우선 확인하고 있습니다.",
                ]
            ),
            "page_count": 1,
            "parsed_page_count": 1,
            "pdf_parse_strategy": "text_blocks_table_candidates_with_optional_ocr",
        },
    )

    assert response["intent"] == "pdf_attachment_analysis"
    assert response["scope"] == "uploaded_pdf"
    assert response["sources"][0]["type"] == "pdf_attachment"
    assert response["report_draft"]["title"]
    assert len(response["report_draft"]["sections"]) >= 5
    assert response["report_draft"]["sections"][0]["title"] == "목차 및 구성"
    assert response["answer_blocks"][0]["title"] == "PDF 핵심 요약"
    assert response["provenance"]["attachment"]["file_name"] == "axis-market.pdf"


@pytest.mark.asyncio
async def test_chat_pdf_endpoint_extracts_uploaded_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    monkeypatch.setenv("AXIS_CHAT_PDF_ENABLE_LLM", "0")

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "AX market PDF test. SK AX should review cloud operation risk and contract evidence.",
    )
    pdf_bytes = doc.write()

    response = await chat_pdf(
        ChatPdfRequest(
            request=ChatTurnRequest(
                message="이 PDF를 요약해줘",
                conversation_id="22222222-2222-2222-2222-222222222222",
            ),
            file_name="uploaded.pdf",
            content_type="application/pdf",
            pdf_base64=base64.b64encode(pdf_bytes).decode("ascii"),
        )
    )
    payload = response.model_dump()

    assert payload["intent"] == "pdf_attachment_analysis"
    assert payload["sources"][0]["type"] == "pdf_attachment"
    assert payload["report_draft"]["sections"][0]["title"] == "목차 및 구성"
    assert payload["report_draft"]["sections"][1]["title"] == "Executive Summary"
    assert payload["provenance"]["attachment"]["text_chars"] > 0
