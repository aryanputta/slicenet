"""
SliceNet REST + WebSocket control plane.

Endpoints:
  GET  /health                    — liveness probe
  POST /simulation                — start a simulation run (async background task)
  GET  /simulation/status         — running / elapsed / config
  DELETE /simulation              — stop the current simulation
  POST /simulation/load           — adjust traffic load factor at runtime
  POST /simulation/burst          — inject a burst of packets mid-simulation
  GET  /metrics                   — current metrics snapshot (JSON)
  GET  /metrics/prometheus        — Prometheus text-format metrics
  WS   /ws/metrics                — WebSocket live metrics stream (1-second ticks)

Usage:
    pip install fastapi uvicorn
    uvicorn slicenet.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
    from fastapi.responses import PlainTextResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "fastapi is required. Install with: pip install fastapi uvicorn"
    ) from exc

from slicenet.api.models import (
    BurstRequest,
    HealthResponse,
    LoadAdjustRequest,
    MetricsSnapshot,
    SimulationConfig,
    SimulationStatus,
    SliceMetrics,
)
from slicenet.engine import SliceNetEngine
from slicenet.metrics.prometheus import render_prometheus

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SliceNet QoS Engine",
    description=(
        "REST + WebSocket control plane for the SliceNet network simulation engine. "
        "Supports real-time load adjustment, burst injection, and live metrics streaming."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Shared engine state
# ---------------------------------------------------------------------------

_engine: Optional[SliceNetEngine] = None
_sim_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
_sim_start_time: Optional[float] = None
_sim_config: Optional[SimulationConfig] = None
_ws_clients: set[WebSocket] = set()


# ---------------------------------------------------------------------------
# Background simulation runner
# ---------------------------------------------------------------------------

async def _run_simulation(cfg: SimulationConfig) -> None:
    """Run the simulation in a thread pool so it doesn't block the event loop."""
    global _engine
    loop = asyncio.get_event_loop()

    def _sync_run() -> None:
        assert _engine is not None
        _engine.run(
            duration_ms=cfg.duration_ms,
            tick_ms=cfg.tick_ms,
            drain_per_tick=cfg.drain_per_tick,
            verbose=cfg.verbose,
        )

    try:
        await loop.run_in_executor(None, _sync_run)
    except asyncio.CancelledError:
        logger.info("Simulation cancelled.")
    except Exception as exc:
        logger.exception("Simulation error: %s", exc)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _build_metrics_snapshot() -> MetricsSnapshot:
    global _engine, _sim_start_time

    if _engine is None:
        return MetricsSnapshot(
            simulation_running=False,
            elapsed_ms=0.0,
            jains_fairness=0.0,
            scheduler_efficiency=0.0,
            slices=[],
        )

    report = _engine.metrics.full_report()
    elapsed = (time.monotonic() - _sim_start_time) * 1000.0 if _sim_start_time else 0.0

    slices = []
    for slice_id, data in report.get("slices", {}).items():
        lat = data.get("latency", {})
        slices.append(
            SliceMetrics(
                slice_id=slice_id,
                p50_ms=lat.get("p50"),
                p95_ms=lat.get("p95"),
                p99_ms=lat.get("p99"),
                throughput_mbps=data.get("throughput_mbps", 0.0),
                loss_rate=data.get("loss_rate", 0.0),
                sla_violations=data.get("sla_violations", 0),
            )
        )

    sys_metrics = report.get("system", {})
    return MetricsSnapshot(
        simulation_running=_sim_task is not None and not _sim_task.done(),
        elapsed_ms=round(elapsed, 2),
        jains_fairness=round(sys_metrics.get("jains_fairness_index", 0.0), 4),
        scheduler_efficiency=round(sys_metrics.get("scheduler_efficiency", 0.0), 4),
        slices=slices,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse()


@app.post("/simulation", status_code=status.HTTP_202_ACCEPTED, tags=["Simulation"])
async def start_simulation(cfg: SimulationConfig) -> dict:
    """
    Start a new simulation run asynchronously.
    Returns 409 if a simulation is already running.
    """
    global _engine, _sim_task, _sim_start_time, _sim_config

    if _sim_task is not None and not _sim_task.done():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A simulation is already running. DELETE /simulation to stop it.",
        )

    _engine = SliceNetEngine(
        scheduler=cfg.scheduler.value,
        load_factor=cfg.load_factor,
        tcp_loss_rate=cfg.tcp_loss_rate,
        udp_loss_rate=cfg.udp_loss_rate,
        link_capacity_mbps=cfg.link_capacity_mbps,
    )
    _sim_config = cfg
    _sim_start_time = time.monotonic()
    _sim_task = asyncio.create_task(_run_simulation(cfg))
    logger.info("Simulation started: %s", cfg.model_dump())
    return {"status": "started", "config": cfg.model_dump()}


@app.get("/simulation/status", response_model=SimulationStatus, tags=["Simulation"])
async def simulation_status() -> SimulationStatus:
    """Return current simulation state."""
    global _sim_task, _sim_start_time, _sim_config
    running = _sim_task is not None and not _sim_task.done()
    elapsed = (
        (time.monotonic() - _sim_start_time) * 1000.0
        if _sim_start_time and running
        else None
    )
    return SimulationStatus(
        running=running,
        elapsed_ms=round(elapsed, 2) if elapsed is not None else None,
        config=_sim_config.model_dump() if _sim_config else None,
    )


@app.delete("/simulation", status_code=status.HTTP_200_OK, tags=["Simulation"])
async def stop_simulation() -> dict:
    """Stop the current simulation if running."""
    global _sim_task
    if _sim_task is None or _sim_task.done():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active simulation.",
        )
    _sim_task.cancel()
    try:
        await _sim_task
    except asyncio.CancelledError:
        pass
    return {"status": "stopped"}


@app.post("/simulation/load", tags=["Simulation"])
async def adjust_load(req: LoadAdjustRequest) -> dict:
    """Dynamically adjust traffic load factor during a running simulation."""
    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No simulation initialized.",
        )
    _engine.set_load(req.load_factor)
    return {"load_factor": req.load_factor}


@app.post("/simulation/burst", tags=["Simulation"])
async def inject_burst(req: BurstRequest) -> dict:
    """Inject a one-shot traffic burst into the running simulation."""
    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No simulation initialized.",
        )
    _engine.inject_burst(profile=req.profile, count=req.count)
    return {"injected": req.count, "profile": req.profile}


@app.get("/metrics", response_model=MetricsSnapshot, tags=["Metrics"])
async def get_metrics() -> MetricsSnapshot:
    """Return current simulation metrics as JSON."""
    return _build_metrics_snapshot()


@app.get("/metrics/prometheus", response_class=PlainTextResponse, tags=["Metrics"])
async def get_prometheus_metrics() -> str:
    """Return metrics in Prometheus text format for scraping."""
    if _engine is None:
        return "# No active simulation\n"
    return render_prometheus(_engine.metrics)


# ---------------------------------------------------------------------------
# WebSocket live streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket) -> None:
    """
    WebSocket endpoint that streams metrics every second.

    Connect with any WS client:
        wscat -c ws://localhost:8000/ws/metrics

    Each message is a JSON-serialised MetricsSnapshot.
    """
    await websocket.accept()
    _ws_clients.add(websocket)
    logger.info("WebSocket client connected. Total clients: %d", len(_ws_clients))
    try:
        while True:
            snapshot = _build_metrics_snapshot()
            payload = snapshot.model_dump_json()
            await websocket.send_text(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.warning("WebSocket error: %s", exc)
    finally:
        _ws_clients.discard(websocket)
