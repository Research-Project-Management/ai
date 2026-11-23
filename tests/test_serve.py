"""Tests for hardware-adaptive uvicorn server sizing (`ai.serve`)."""

from __future__ import annotations

from pathlib import Path

from ai.serve import compute_worker_count, detect_cpu_count, detect_ram_gb


def test_compute_worker_count_scales_and_caps_safely() -> None:
    assert compute_worker_count(cpu_count=1, ram_gb=1.0) == 1
    assert compute_worker_count(cpu_count=2, ram_gb=4.0) == 2
    assert compute_worker_count(cpu_count=4, ram_gb=8.0) == 4
    assert compute_worker_count(cpu_count=16, ram_gb=4.0) == 5
    assert compute_worker_count(cpu_count=16, ram_gb=32.0) == 8
    assert compute_worker_count(cpu_count=2, ram_gb=1.0, env_override=3) == 3
    assert compute_worker_count(cpu_count=4, ram_gb=None) == 4


def test_detect_cpu_and_ram_honor_cgroup_v2_and_meminfo(tmp_path: Path) -> None:
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("200000 100000\n", encoding="utf-8")
    assert detect_cpu_count(cpu_max) >= 1

    cpu_unbounded = tmp_path / "cpu_unbounded.max"
    cpu_unbounded.write_text("max 100000\n", encoding="utf-8")
    assert detect_cpu_count(cpu_unbounded) >= 1

    mem_max = tmp_path / "memory.max"
    mem_max.write_text(str(int(1.5 * 1024**3)), encoding="utf-8")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:        4194304 kB\nMemFree: 1000 kB\n", encoding="utf-8")

    detected_ram = detect_ram_gb(mem_max, meminfo)
    assert detected_ram is not None
    assert abs(detected_ram - 1.5) < 0.01

    mem_unbounded = tmp_path / "memory_unbounded.max"
    mem_unbounded.write_text("max\n", encoding="utf-8")
    host_only = detect_ram_gb(mem_unbounded, meminfo)
    assert host_only is not None
    assert abs(host_only - 4.0) < 0.01

    assert detect_ram_gb(tmp_path / "missing_mem", tmp_path / "missing_proc") is None
