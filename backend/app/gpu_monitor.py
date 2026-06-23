from __future__ import annotations

import subprocess
from typing import Any

from .schemas import GpuInfo


def _query_nvidia_smi() -> dict[int, dict[str, float | int | str]]:
    """Return per-GPU utilization stats from nvidia-smi, keyed by device index."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return {}
        stats: dict[int, dict[str, float | int | str]] = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            idx = int(parts[0])
            stats[idx] = {
                "name": parts[1],
                "utilization_gpu": float(parts[2]) if parts[2] not in ("[N/A]", "N/A", "") else 0.0,
                "memory_used_mb": float(parts[3]) if parts[3] not in ("[N/A]", "N/A", "") else 0.0,
                "memory_total_mb": float(parts[4]) if parts[4] not in ("[N/A]", "N/A", "") else 0.0,
                "temperature_c": float(parts[5]) if parts[5] not in ("[N/A]", "N/A", "") else 0.0,
            }
        return stats
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        return {}


def get_gpu_info() -> list[GpuInfo]:
    """Collect GPU count, memory, and utilization for the dashboard."""
    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    smi = _query_nvidia_smi()
    gpus: list[GpuInfo] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        total_bytes = props.total_memory
        reserved = torch.cuda.memory_reserved(index)
        allocated = torch.cuda.memory_allocated(index)

        smi_row = smi.get(index, {})
        name = str(smi_row.get("name") or props.name)
        total_mb_smi = float(smi_row.get("memory_total_mb") or 0)
        used_mb_smi = float(smi_row.get("memory_used_mb") or 0)

        if total_mb_smi > 0:
            memory_total_mb = total_mb_smi
            memory_used_mb = used_mb_smi
        else:
            memory_total_mb = total_bytes / (1024 * 1024)
            memory_used_mb = max(reserved, allocated) / (1024 * 1024)

        gpus.append(
            GpuInfo(
                index=index,
                name=name,
                memory_total_mb=round(memory_total_mb, 1),
                memory_used_mb=round(memory_used_mb, 1),
                memory_free_mb=round(max(0.0, memory_total_mb - memory_used_mb), 1),
                utilization_gpu=round(float(smi_row.get("utilization_gpu") or 0.0), 1),
                temperature_c=round(float(smi_row.get("temperature_c") or 0.0), 1),
                torch_allocated_mb=round(allocated / (1024 * 1024), 1),
            )
        )
    return gpus


def get_gpu_summary() -> dict[str, Any]:
    gpus = get_gpu_info()
    return {
        "cuda_available": len(gpus) > 0,
        "count": len(gpus),
        "gpus": [g.model_dump() for g in gpus],
    }
