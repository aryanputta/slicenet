"""
Pydantic request/response models for the SliceNet REST API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pydantic is required for the API. Install with: pip install pydantic"
    ) from exc


class SchedulerType(str, Enum):
    fifo = "fifo"
    priority = "priority"
    wfq = "wfq"
    drr = "drr"
    adaptive = "adaptive"


class TransportAlgo(str, Enum):
    reno = "reno"
    cubic = "cubic"
    bbr = "bbr"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SimulationConfig(BaseModel):
    """POST /simulation — start a new simulation run."""

    scheduler: SchedulerType = SchedulerType.adaptive
    transport: TransportAlgo = TransportAlgo.reno
    duration_ms: int = Field(default=2000, ge=100, le=60_000)
    tick_ms: float = Field(default=1.0, ge=0.1, le=100.0)
    load_factor: float = Field(default=0.75, ge=0.0, le=1.0)
    tcp_loss_rate: float = Field(default=0.001, ge=0.0, le=0.5)
    udp_loss_rate: float = Field(default=0.001, ge=0.0, le=0.5)
    link_capacity_mbps: float = Field(default=100.0, ge=1.0, le=400_000.0)
    drain_per_tick: int = Field(default=10, ge=1, le=500)
    verbose: bool = False

    @field_validator("duration_ms")
    @classmethod
    def _max_duration(cls, v: int) -> int:
        if v > 60_000:
            raise ValueError("duration_ms must be <= 60000 (60 seconds)")
        return v


class LoadAdjustRequest(BaseModel):
    """POST /simulation/load — dynamically adjust traffic load."""
    load_factor: float = Field(ge=0.0, le=1.0)


class BurstRequest(BaseModel):
    """POST /simulation/burst — inject a one-shot traffic burst."""
    profile: str = Field(
        default="video",
        description="Traffic profile: voip | video | iot | http | bulk",
    )
    count: int = Field(default=100, ge=1, le=10_000)

    @field_validator("profile")
    @classmethod
    def _valid_profile(cls, v: str) -> str:
        valid = {"voip", "video", "iot", "http", "bulk"}
        if v not in valid:
            raise ValueError(f"profile must be one of {valid}")
        return v


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SimulationStatus(BaseModel):
    running: bool
    elapsed_ms: Optional[float]
    config: Optional[Dict[str, Any]]


class SliceMetrics(BaseModel):
    slice_id: str
    p50_ms: Optional[float]
    p95_ms: Optional[float]
    p99_ms: Optional[float]
    throughput_mbps: float
    loss_rate: float
    sla_violations: int


class MetricsSnapshot(BaseModel):
    """GET /metrics — current metrics snapshot."""
    simulation_running: bool
    elapsed_ms: float
    jains_fairness: float
    scheduler_efficiency: float
    slices: List[SliceMetrics]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
