"""Hardware-adaptive uvicorn server entrypoint for container and bare-metal deployments."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import uvicorn

from ai.shared.config.server import ServerConfig

logger = logging.getLogger(__name__)

_CPU_MAX_PATH = Path("/sys/fs/cgroup/cpu.max")
_MEM_MAX_PATH = Path("/sys/fs/cgroup/memory.max")
_PROC_MEMINFO_PATH = Path("/proc/meminfo")
_BYTES_PER_GIB = 1024.0**3
_WORKER_RAM_BUDGET_GB = 0.75
_DEFAULT_MAX_WORKERS = 8


def _read_cgroup_cpu_quota(cpu_max_path: Path) -> int | None:
    if not cpu_max_path.is_file():
        return None
    try:
        parts = cpu_max_path.read_text(encoding="utf-8").strip().split()
    except OSError:
        return None
    if len(parts) < 2 or parts[0] == "max":
        return None
    try:
        quota = int(parts[0])
        period = int(parts[1])
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, math.floor(quota / period))


def detect_cpu_count(cpu_max_path: Path = _CPU_MAX_PATH) -> int:
    """Detect effective CPU count, honoring Linux cgroup v2 CPU quotas when set."""
    host_cpus = max(1, os.cpu_count() or 1)
    cgroup_cpus = _read_cgroup_cpu_quota(cpu_max_path)
    if cgroup_cpus is None:
        return host_cpus
    return max(1, min(host_cpus, cgroup_cpus))


def _read_cgroup_memory_gb(mem_max_path: Path) -> float | None:
    if not mem_max_path.is_file():
        return None
    try:
        raw = mem_max_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        limit_bytes = int(raw)
    except ValueError:
        return None
    if limit_bytes <= 0:
        return None
    return limit_bytes / _BYTES_PER_GIB


def _read_proc_meminfo_gb(proc_meminfo_path: Path) -> float | None:
    if not proc_meminfo_path.is_file():
        return None
    try:
        lines = proc_meminfo_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    kb = int(parts[1])
                except ValueError:
                    return None
                if kb > 0:
                    return (kb * 1024.0) / _BYTES_PER_GIB
    return None


def detect_ram_gb(
    mem_max_path: Path = _MEM_MAX_PATH,
    proc_meminfo_path: Path = _PROC_MEMINFO_PATH,
) -> float | None:
    """Detect effective RAM ceiling in GiB from cgroup v2 limit or `/proc/meminfo`."""
    cgroup_gb = _read_cgroup_memory_gb(mem_max_path)
    host_gb = _read_proc_meminfo_gb(proc_meminfo_path)
    if cgroup_gb is not None and host_gb is not None:
        return min(cgroup_gb, host_gb)
    return cgroup_gb if cgroup_gb is not None else host_gb


def compute_worker_count(
    *,
    cpu_count: int,
    ram_gb: float | None,
    env_override: int | None = None,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> int:
    """Choose safe uvicorn worker count bounded by CPU, RAM (~750 MB/worker), and `max_workers`."""
    if env_override is not None and env_override > 0:
        return env_override

    workers = max(1, cpu_count)
    if ram_gb is not None:
        ram_cap = max(1, int(ram_gb // _WORKER_RAM_BUDGET_GB))
        workers = min(workers, ram_cap)
    return max(1, min(workers, max_workers))


def _parse_optional_positive_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        value = int(stripped)
    except ValueError:
        logger.warning("Ignoring invalid UVICORN_WORKERS=%r", raw)
        return None
    return value if value > 0 else None


def main() -> None:
    """Launch uvicorn with hardware-adaptive concurrency and proxy header support."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    server_cfg = ServerConfig()
    host = server_cfg.host
    port = server_cfg.port
    override = _parse_optional_positive_int(os.getenv("UVICORN_WORKERS"))

    cpu_count = detect_cpu_count()
    ram_gb = detect_ram_gb()
    workers = compute_worker_count(cpu_count=cpu_count, ram_gb=ram_gb, env_override=override)

    ram_label = f"{ram_gb:.2f} GiB" if ram_gb is not None else "unknown"
    logger.info(
        "Starting Flux AI service on %s:%s (cpus=%d, ram=%s, workers=%d, override=%s)",
        host,
        port,
        cpu_count,
        ram_label,
        workers,
        override,
    )
    uvicorn.run(
        "ai.main:app",
        host=host,
        port=port,
        workers=workers,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
