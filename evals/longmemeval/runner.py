"""Reproducible 5-axis LongMemEval-aligned project memory benchmark runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from ai.shared.config.settings import load_settings

SUITE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = SUITE_DIR / "corpus.json"
CASES_PATH = SUITE_DIR / "cases.json"

_ABSTENTION_PHRASES: tuple[str, ...] = (
    "không có thông tin",
    "không đề cập",
    "không ghi nhận",
    "không được ghi",
    "không được cung cấp",
    "không lưu",
    "không tìm thấy",
    "chưa có căn cứ",
    "chưa lấy được",
    "không truy cập được",
    "không thể trả lời",
    "chưa có thông tin",
    "hệ thống hr",
    "nhân sự",
    "not mentioned",
    "not recorded",
    "no information",
)
_INVENTED_SALARY_RE = re.compile(
    r"(\$|usd|vnd|đồng|triệu)\s*\d|\d[\d.,]*\s*(usd|vnd|triệu|\$)",
    re.IGNORECASE,
)
_UNICODE_HYPHENS = re.compile(r"[\u2010-\u2015\u2212]")
_UNICODE_SPACES = re.compile(r"[\u00a0\u202f]")


def _normalize_text(text: str) -> str:
    cleaned = _UNICODE_HYPHENS.sub("-", text)
    return _UNICODE_SPACES.sub(" ", cleaned)


def _match_token(text_lower: str, token: str) -> bool:
    needle = _normalize_text(token).lower()
    if needle.isdigit():
        pattern = rf"(?<![\da-zA-Z.]){re.escape(needle)}(?![\da-zA-Z.])"
        return bool(re.search(pattern, text_lower))
    return needle in text_lower


def evaluate_case(
    case: dict[str, Any],
    chat_result: dict[str, Any],
    *,
    search_hits_count: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Deterministically score one LongMemEval case against chat + optional search output."""
    http_ok = chat_result.get("http_status") == 200
    body = chat_result.get("body") if isinstance(chat_result.get("body"), dict) else {}
    status_ok = body.get("status") == "completed"
    text = str(body.get("content") or "")
    text_lower = _normalize_text(text).lower()
    citations = body.get("citations") or []

    missing_groups: list[list[str]] = []
    for group in case.get("required_all", []):
        if not any(_match_token(text_lower, alt) for alt in group):
            missing_groups.append(group)

    forbidden_hits = [tok for tok in case.get("forbidden_any", []) if _match_token(text_lower, tok)]
    citation_ok = (len(citations) > 0) if case.get("require_citations", False) else True

    abstention_ok = True
    invented_salary = False
    if case.get("expect_abstention", False):
        abstains = any(phrase in text_lower for phrase in _ABSTENTION_PHRASES)
        invented_salary = bool(_INVENTED_SALARY_RE.search(text_lower))
        no_citations = len(citations) == 0 if case.get("check_search_zero", False) else True
        abstention_ok = abstains and not invented_salary and no_citations

    search_zero_ok = True
    if case.get("check_search_zero", False):
        search_zero_ok = search_hits_count == 0

    passed = (
        http_ok
        and status_ok
        and not missing_groups
        and not forbidden_hits
        and citation_ok
        and abstention_ok
        and search_zero_ok
    )
    details = {
        "id": case["id"],
        "axis": case["axis"],
        "axis_name": case["axis_name"],
        "space_id": case["space_id"],
        "passed": passed,
        "http_status": chat_result.get("http_status"),
        "elapsed_ms": chat_result.get("elapsed_ms"),
        "route": body.get("route"),
        "citations_count": len(citations),
        "citation_titles": [c.get("title") or c.get("source_id") for c in citations],
        "missing_groups": missing_groups,
        "forbidden_hits": forbidden_hits,
        "invented_salary": invented_salary,
        "search_hits_count": search_hits_count,
        "usage": body.get("usage"),
        "response_excerpt": text,
    }
    return passed, details


async def _post_with_429_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    max_attempts: int = 4,
    **kwargs: Any,
) -> httpx.Response:
    """POST and sleep through Flux's per-space HTTP 429 rate-limit windows."""
    resp = await client.post(url, headers=headers, **kwargs)
    for attempt in range(max_attempts - 1):
        if resp.status_code != 429:
            return resp
        wait_seconds = 15.0
        retry_header = resp.headers.get("retry-after")
        if retry_header and retry_header.isdigit():
            wait_seconds = float(retry_header)
        else:
            try:
                detail = resp.json().get("detail")
            except ValueError:
                detail = None
            if isinstance(detail, dict) and isinstance(
                detail.get("reset_after_seconds"), int | float
            ):
                wait_seconds = float(detail["reset_after_seconds"])
        await asyncio.sleep(min(max(wait_seconds, 1.0), 65.0) + 1.0 + attempt * 0.25)
        resp = await client.post(url, headers=headers, **kwargs)
    return resp


async def _ingest_doc(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    headers: dict[str, str],
    space_id: str,
    doc: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    metadata: dict[str, Any] = {
        "month": doc["month"],
        "effective_at": doc["effective_at"],
        "valid_from": doc["effective_at"],
        "author": doc["author"],
    }
    if doc.get("supersedes_source_id"):
        metadata["supersedes_source_id"] = doc["supersedes_source_id"]
    if doc.get("supersedes_source_ids"):
        metadata["supersedes_source_ids"] = doc["supersedes_source_ids"]
    resp = await _post_with_429_retry(
        client,
        f"{base_url}/knowledge/ingest",
        headers=headers,
        data={
            "space_id": space_id,
            "source_type": doc["source_type"],
            "source_id": doc["source_id"],
            "title": doc["title"],
            "metadata_json": json.dumps(metadata),
        },
        files={"file": (doc["filename"], doc["content"].encode("utf-8"), "text/markdown")},
    )
    if resp.status_code != 202:
        return {
            "source_id": doc["source_id"],
            "month": doc["month"],
            "status": "submit_failed",
            "http_status": resp.status_code,
            "error": resp.text,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    job_id = resp.json()["job_id"]
    deadline = time.perf_counter() + 90.0
    while time.perf_counter() < deadline:
        poll = await client.get(f"{base_url}/knowledge/jobs/{job_id}", headers=headers)
        poll_data = poll.json()
        if poll_data["status"] in ("completed", "failed"):
            poll_data["month"] = doc["month"]
            poll_data["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return poll_data
        await asyncio.sleep(0.35)

    return {
        "source_id": doc["source_id"],
        "month": doc["month"],
        "job_id": job_id,
        "status": "timeout",
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


async def _cleanup_spaces(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    headers: dict[str, str],
    spaces: tuple[str, ...],
) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for space in spaces:
        r_kb = await client.delete(f"{base_url}/spaces/{space}/knowledge", headers=headers)
        r_th = await client.delete(f"{base_url}/spaces/{space}/threads", headers=headers)
        cleaned[space] = {
            "knowledge_delete_status": r_kb.status_code,
            "threads_delete_status": r_th.status_code,
        }
    verify = await _post_with_429_retry(
        client,
        f"{base_url}/knowledge/search",
        headers=headers,
        json={"space_id": spaces[0], "query": "Helios-AI", "limit": 5},
    )
    cleaned["post_cleanup_search_hits"] = len(verify.json().get("hits", []))
    return cleaned


def _summarize_axes(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    axes: dict[str, dict[str, Any]] = {}
    for row in results:
        axis = row["axis"]
        bucket = axes.setdefault(
            axis,
            {"axis_name": row["axis_name"], "passed": 0, "total": 0, "latencies_ms": []},
        )
        bucket["total"] += 1
        if row["passed"]:
            bucket["passed"] += 1
        if isinstance(row.get("elapsed_ms"), int | float):
            bucket["latencies_ms"].append(float(row["elapsed_ms"]))

    for bucket in axes.values():
        lats = bucket.pop("latencies_ms")
        total = bucket["total"]
        bucket["pass_rate"] = round(bucket["passed"] / total, 4) if total else 0.0
        bucket["avg_latency_ms"] = round(sum(lats) / len(lats), 1) if lats else 0.0
    return axes


def _generate_noise_docs(count: int) -> list[dict[str, Any]]:
    """Generate deterministic distractor Helios-AI notes across Months 1-12."""
    templates: tuple[tuple[str, str, str], ...] = (
        (
            "Sprint Standup & Triage Notes",
            "report",
            (
                "Reviewed staging smoke tests, dependency patch bumps, and UI spacing polish for "
                "Helios-AI workspace tabs. Verified health probe intervals and log rotation."
            ),
        ),
        (
            "PDF Ingestion Worker Operations Note",
            "project_doc",
            (
                "Audited Grafana memory panels during routine Helios-AI PDF ingestion queue checks "
                "and confirmed alert thresholds for dead-letter retry backoff."
            ),
        ),
        (
            "Frontend Citation Drawer Polish Log",
            "project_doc",
            (
                "Adjusted dark-mode border contrast, tooltip delay, and keyboard navigation inside "
                "the Helios-AI Citation Viewer drawer after pilot feedback."
            ),
        ),
        (
            "Retrieval Quality Calibration Memo",
            "report",
            (
                "Inspected weekly retrieval latency histograms and Qdrant collection disk snapshots "
                "across active Helios-AI test workspaces with no schema changes."
            ),
        ),
        (
            "Platform Deployment Checklist Runbook",
            "policy",
            (
                "Validated container non-root startup order, Redis PING warmup, and internal key "
                "rotation rehearsal on pre-production staging nodes."
            ),
        ),
    )
    docs: list[dict[str, Any]] = []
    for idx in range(count):
        title_prefix, source_type, body = templates[idx % len(templates)]
        month = (idx % 12) + 1
        day = (idx % 28) + 1
        docs.append(
            {
                "source_id": f"helios-noise-{idx + 1:03d}",
                "source_type": source_type,
                "filename": f"noise_{idx + 1:03d}_m{month:02d}.md",
                "title": f"Helios-AI {title_prefix} #{idx + 1} (Month {month})",
                "month": month,
                "effective_at": f"2025-{month:02d}-{day:02d}",
                "author": "Operations Log",
                "content": (
                    f"# Helios-AI {title_prefix} #{idx + 1} (2025-{month:02d}-{day:02d})\n\n"
                    f"Month {month} routine operational log.\n\n{body}\n"
                ),
            }
        )
    return docs


def _summarize_doc_content(content: str, *, max_chars: int = 420) -> str:
    lines: list[str] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "Author:", "Authors:", "Prepared by:")):
            continue
        lines.append(line.lstrip("- "))
    compact = " ".join(lines)
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def _build_corpus_timeline_events(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    superseded_all: set[str] = set()
    for doc in corpus:
        if isinstance(doc.get("supersedes_source_id"), str):
            superseded_all.add(doc["supersedes_source_id"])
        for sid in doc.get("supersedes_source_ids") or ():
            superseded_all.add(str(sid))

    events: list[dict[str, Any]] = []
    for doc in corpus:
        superseded_refs = doc.get("supersedes_source_ids") or ()
        first_superseded = doc.get("supersedes_source_id") or (
            superseded_refs[0] if superseded_refs else None
        )
        events.append(
            {
                "source_id": doc["source_id"],
                "source_type": doc["source_type"],
                "title": doc["title"],
                "effective_at": doc["effective_at"],
                "summary": _summarize_doc_content(doc["content"]),
                "lifecycle_status": (
                    "superseded" if doc["source_id"] in superseded_all else "active"
                ),
                "supersedes_source_id": first_superseded,
            }
        )
    return events


async def run_suite(
    *,
    base_url: str,
    space_primary: str,
    space_other: str,
    out_path: Path | None,
    noise_docs: int = 0,
    with_digest: bool = False,
) -> dict[str, Any]:
    corpus: list[dict[str, Any]] = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    all_docs = [*corpus, *_generate_noise_docs(noise_docs)]
    cases: list[dict[str, Any]] = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    settings = load_settings()
    headers = {"X-Internal-Key": settings.server.internal_api_key.get_secret_value()}

    report: dict[str, Any] = {
        "base_url": base_url,
        "corpus_docs": len(all_docs),
        "golden_docs": len(corpus),
        "noise_docs": noise_docs,
        "cases_total": len(cases),
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        health = await client.get(f"{base_url}/health")
        report["health_before"] = health.json()
        await _cleanup_spaces(
            client,
            base_url=base_url,
            headers=headers,
            spaces=(space_primary, space_other),
        )

        try:
            t_seed = time.perf_counter()
            ingest_jobs: list[dict[str, Any]] = []
            for idx in range(0, len(all_docs), 5):
                batch = all_docs[idx : idx + 5]
                batch_res = await asyncio.gather(
                    *(
                        _ingest_doc(
                            client,
                            base_url=base_url,
                            headers=headers,
                            space_id=space_primary,
                            doc=doc,
                        )
                        for doc in batch
                    )
                )
                ingest_jobs.extend(batch_res)

            failed_ingest = [j for j in ingest_jobs if j.get("status") != "completed"]
            if failed_ingest:
                raise RuntimeError(f"Corpus ingestion failed: {failed_ingest}")

            report["ingest"] = {
                "elapsed_ms": round((time.perf_counter() - t_seed) * 1000, 1),
                "total_chunks": sum(j.get("chunks_indexed", 0) for j in ingest_jobs),
                "docs_completed": len(ingest_jobs),
            }
            if with_digest:
                digest_resp = await _post_with_429_retry(
                    client,
                    f"{base_url}/knowledge/timeline-digest",
                    headers=headers,
                    json={
                        "space_id": space_primary,
                        "digest_source_id": "helios-timeline-digest-y1",
                        "period_label": "Months 3-12 (2025)",
                        "events": _build_corpus_timeline_events(corpus),
                    },
                )
                if digest_resp.status_code != 200:
                    raise RuntimeError(
                        f"Timeline digest failed ({digest_resp.status_code}): {digest_resp.text}"
                    )
                report["timeline_digest"] = digest_resp.json()

            case_results: list[dict[str, Any]] = []
            for case in cases:
                space_id = (
                    space_other if case["space_id"] == "eval-longmem-other" else space_primary
                )
                search_hits_count: int | None = None
                if case.get("check_search_zero", False):
                    search_resp = await _post_with_429_retry(
                        client,
                        f"{base_url}/knowledge/search",
                        headers=headers,
                        json={"space_id": space_id, "query": case["prompt"], "limit": 5},
                    )
                    search_hits_count = len(search_resp.json().get("hits", []))

                t_case = time.perf_counter()
                chat_resp = await _post_with_429_retry(
                    client,
                    f"{base_url}/chat/sync",
                    headers=headers,
                    json={
                        "messages": [{"role": "user", "content": case["prompt"]}],
                        "space_id": space_id,
                        "stateless": True,
                    },
                )
                elapsed_ms = round((time.perf_counter() - t_case) * 1000, 1)
                is_json = chat_resp.headers.get("content-type", "").startswith("application/json")
                chat_result = {
                    "http_status": chat_resp.status_code,
                    "elapsed_ms": elapsed_ms,
                    "body": chat_resp.json() if is_json else chat_resp.text,
                }
                _, scored = evaluate_case(
                    {**case, "space_id": space_id},
                    chat_result,
                    search_hits_count=search_hits_count,
                )
                case_results.append(scored)

            resilience = await client.get(f"{base_url}/health/resilience", headers=headers)
            report["health_resilience"] = resilience.json()
            report["results"] = case_results
            report["axis_summary"] = _summarize_axes(case_results)
            passed = sum(1 for r in case_results if r["passed"])
            latencies = sorted(float(r["elapsed_ms"]) for r in case_results)
            p95_idx = max(0, min(len(latencies) - 1, int(len(latencies) * 0.95)))
            report["summary"] = {
                "passed": passed,
                "total": len(case_results),
                "pass_rate": round(passed / len(case_results), 4) if case_results else 0.0,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
                "p95_latency_ms": latencies[p95_idx] if latencies else 0.0,
            }
        finally:
            report["cleanup"] = await _cleanup_spaces(
                client,
                base_url=base_url,
                headers=headers,
                spaces=(space_primary, space_other),
            )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5-axis LongMemEval benchmark against Flux AI."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18420")
    parser.add_argument("--space-primary", default="eval-longmem-12m")
    parser.add_argument("--space-other", default="eval-longmem-other")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--noise-docs", type=int, default=0)
    parser.add_argument("--with-digest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary_report = asyncio.run(
        run_suite(
            base_url=args.base_url.rstrip("/"),
            space_primary=args.space_primary,
            space_other=args.space_other,
            out_path=args.out,
            noise_docs=max(0, args.noise_docs),
            with_digest=args.with_digest,
        )
    )
    overall = summary_report["summary"]
    print(f"LONGMEMEVAL_DONE {overall['passed']}/{overall['total']} passed")
    for axis, stats in summary_report["axis_summary"].items():
        print(
            f"  {axis} ({stats['axis_name']}): "
            f"{stats['passed']}/{stats['total']} (avg {stats['avg_latency_ms']} ms)"
        )
    if args.out is not None:
        print(f"Report saved to: {args.out}")
