"""Live end-to-end verification suite against real Groq LLM, Qdrant, Redis, SearXNG & OpenAlex/arXiv."""

from __future__ import annotations

import time
from typing import Any

import httpx
import pymupdf
from fastapi.testclient import TestClient

from ai.bootstrap import resolve_settings
from ai.main import create_app


def _build_demo_pdf_bytes() -> bytes:
    """Create a 2-page technical report PDF in memory for live RAG ingestion."""
    doc = pymupdf.open()
    doc.set_metadata({"title": "Flux Lab Technical Report 2026: Adaptive Sparse KV-Cache"})

    page1 = doc.new_page()
    page1.insert_text(
        (54, 72),
        "# Architecture\n"
        "Phuong phap FluxKV nen bo nho KV-Cache giam 63.8% dung luong VRAM "
        "so voi baseline FP16 nho chon loc token quan trong theo tang.",
        fontsize=11,
    )

    page2 = doc.new_page()
    page2.insert_text(
        (54, 72),
        "# Benchmark Results\n"
        "Tren GPU RTX 4090 voi ngu canh 32k tokens, thong luong suy luan dat "
        "2410 tokens/sec va do tre TTFT giam con 41 ms.",
        fontsize=11,
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _print_case_result(label: str, body: dict[str, Any]) -> None:
    print(f"\n==================== {label} ====================")
    print(f"route={body.get('route')} | status={body.get('status')}")
    usage = body.get("usage") or {}
    print(
        f"usage: in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)} "
        f"total={usage.get('total_tokens', 0)} latency_ms={usage.get('latency_ms', 0):.1f}"
    )
    print(f"answer:\n{body.get('content', '')}")
    citations = body.get("citations") or []
    print(f"citations ({len(citations)}):")
    for idx, cite in enumerate(citations, start=1):
        print(
            f"  [{idx}] title={cite.get('title')!r} "
            f"source_id={cite.get('source_id')!r} "
            f"page={cite.get('page')} url={cite.get('url')}"
        )


def run_experiments() -> None:
    settings = resolve_settings()
    internal_key = settings.server.internal_api_key.get_secret_value()
    headers = {"X-Internal-Key": internal_key}
    space_id = "live-exp-space"

    print(
        f"Starting live experiment: app_env={settings.server.app_env.value}, "
        f"model={settings.llm.model}, router={settings.llm.effective_router_model}, "
        f"agent={settings.llm.effective_agent_model}, "
        f"embedding={settings.embedding.provider}, searxng={settings.web_search.searxng_url}"
    )

    app = create_app(settings=settings)
    with TestClient(app) as client:
        try:
            # 0. Health & Agents Discovery
            health = client.get("/health")
            print("\n/health ->", health.status_code, health.json())
            agents = client.get("/agents", headers=headers)
            print("/agents ->", agents.status_code, [a["name"] for a in agents.json()["agents"]])

            # 1. Live PDF Ingest + rag_agent autonomous routing
            pdf_bytes = _build_demo_pdf_bytes()
            ingest_res = client.post(
                "/knowledge/ingest",
                headers=headers,
                data={
                    "space_id": space_id,
                    "source_type": "file",
                    "source_id": "fluxkv-2026",
                },
                files={"file": ("fluxkv_report.pdf", pdf_bytes, "application/pdf")},
            )
            print(f"\n/knowledge/ingest -> {ingest_res.status_code} {ingest_res.json()}")
            ingest_res.raise_for_status()

            r1 = client.post(
                "/chat/sync",
                headers=headers,
                json={
                    "space_id": space_id,
                    "chat_id": "exp-rag-1",
                    "document_ids": ["fluxkv-2026"],
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Theo báo cáo kỹ thuật nội bộ vừa tải lên trong dự án, "
                                "phương pháp FluxKV giảm bao nhiêu phần trăm VRAM và "
                                "đạt thông lượng bao nhiêu tokens/sec trên RTX 4090? "
                                "Trả lời ngắn gọn kèm trích dẫn."
                            ),
                        }
                    ],
                },
            )
            r1.raise_for_status()
            _print_case_result("EXPERIMENT 1: rag_agent (Live Qdrant + PDF)", r1.json())
            time.sleep(15.0)

            # 2. Live Web Search via self-hosted SearXNG + SafeHttpFetch
            r2 = client.post(
                "/chat/sync",
                headers=headers,
                json={
                    "space_id": space_id,
                    "chat_id": "exp-web-1",
                    "context": {"allowed_domains": ["fastapi.tiangolo.com"]},
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Tìm trên trang tài liệu chính thức fastapi.tiangolo.com "
                                "xem FastAPI khuyến nghị dùng cơ chế lifespan context manager nào "
                                "để thay thế cho @app.on_event startup và shutdown cũ? "
                                "Trả lời ngắn gọn."
                            ),
                        }
                    ],
                },
            )
            r2.raise_for_status()
            _print_case_result(
                "EXPERIMENT 2: web_search_agent (Live SearXNG + Allowed Domain)",
                r2.json(),
            )
            time.sleep(15.0)

            # 3. Live Scholarly Search via OpenAlex / arXiv
            r3 = client.post(
                "/chat/sync",
                headers=headers,
                json={
                    "space_id": space_id,
                    "chat_id": "exp-scholar-1",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Tìm các bài báo khoa học trên kho học thuật (arXiv/OpenAlex) "
                                "về thuật toán FlashAttention của Tri Dao và tóm tắt ngắn gọn "
                                "trong tối đa 4 gạch đầu dòng kèm trích dẫn nguồn."
                            ),
                        }
                    ],
                },
            )
            r3.raise_for_status()
            _print_case_result(
                "EXPERIMENT 3: web_search_agent (Live OpenAlex / arXiv Scholar)",
                r3.json(),
            )
            time.sleep(25.0)

            # 4. Multi-Agent Fan-Out (Supervisor -> rag_agent + web_search_agent)
            r4 = client.post(
                "/chat/sync",
                headers=headers,
                json={
                    "space_id": space_id,
                    "chat_id": "exp-multi-1",
                    "document_ids": ["fluxkv-2026"],
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Đối chiếu ngắn gọn (tối đa 4 gạch đầu dòng) số liệu giảm VRAM "
                                "và tốc độ của FluxKV trong tài liệu nội bộ dự án với hướng "
                                "tối ưu bộ nhớ IO-aware trong bài báo FlashAttention từ kho học thuật."
                            ),
                        }
                    ],
                },
            )
            r4.raise_for_status()
            _print_case_result(
                "EXPERIMENT 4: Multi-Agent (rag_agent + web_search_agent)",
                r4.json(),
            )
        finally:
            try:
                cleanup = client.delete(f"/spaces/{space_id}/knowledge", headers=headers)
                print(f"\nCleanup Qdrant space={space_id} -> {cleanup.status_code} {cleanup.json()}")
            except (httpx.HTTPError, RuntimeError, OSError, ValueError) as cleanup_exc:
                print(f"\nCleanup skipped due to error: {type(cleanup_exc).__name__}: {cleanup_exc}")


if __name__ == "__main__":
    run_experiments()
