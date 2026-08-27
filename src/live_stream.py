"""Live-streaming demo: pushes incidents one at a time over a WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from data.simulate import DEFAULT_INCIDENT_COUNT, generate_dataset
from .pipeline import run_incident
from .memory import IncidentMemory

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 4.0
LIVE_BATCH_SIZE = 10  # Smaller batch for live demo


async def stream_incidents(
    send_json: Any,
    *,
    incident_count: int = LIVE_BATCH_SIZE,
    interval: float = DEFAULT_INTERVAL_SECONDS,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Generate incidents and stream pipeline results one at a time.

    Args:
        send_json: async callable that accepts a dict to send as JSON.
        incident_count: how many incidents to process.
        interval: seconds between each incident push.
        stop_event: set this to stop the stream early.
    """
    dataset = generate_dataset(incident_count)
    incidents = dataset["incidents"]
    memory = IncidentMemory()

    # Send a header message so the client knows what to expect.
    await send_json({
        "type": "stream_start",
        "total_incidents": len(incidents),
        "interval_seconds": interval,
    })

    for index, incident in enumerate(incidents):
        if stop_event and stop_event.is_set():
            break

        incident_id = incident.get("incident_id", "unknown")

        # Notify the client that processing has started for this incident.
        await send_json({
            "type": "processing",
            "incident_id": incident_id,
            "index": index,
            "total": len(incidents),
        })

        # Run the FULL pipeline synchronously in a thread so we don't block
        # the event loop.
        t0 = time.monotonic()
        record = await asyncio.to_thread(run_incident, incident, memory)
        elapsed_ms = round((time.monotonic() - t0) * 1000)

        # Build a summary matching the /api/incidents list format, plus the
        # full record for detail rendering.
        primary = record["detection"].get("primary_degradation")
        summary = {
            "incident_id": record["incident_id"],
            "time": record["window"].get("current_start"),
            "method": (
                primary["method_display"]
                if primary
                else record["ground_truth"].get("affected_method_display", "Unknown")
            ),
            "cause": record["correlation"]["predicted_cause"],
            "confidence": record["correlation"]["confidence"],
            "status": (
                "escalated"
                if record["recovery"]["primary_action"] == "escalate to human"
                else "resolved"
            ),
            "recovery_mode": record["recovery"].get("recovery_mode", "SIMULATED"),
        }

        await send_json({
            "type": "incident",
            "index": index,
            "total": len(incidents),
            "pipeline_ms": elapsed_ms,
            "summary": summary,
            "record": record,
        })

        logger.info(
            "live stream pushed incident %s (%d/%d) in %dms",
            incident_id,
            index + 1,
            len(incidents),
            elapsed_ms,
            extra={"incident_id": incident_id, "stage": "live_stream"},
        )

        # Wait before sending the next incident (skips wait after the last one).
        if index < len(incidents) - 1:
            if stop_event:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                    break  # stop_event was set during the wait
                except asyncio.TimeoutError:
                    pass  # Normal — interval elapsed, continue
            else:
                await asyncio.sleep(interval)

    await send_json({
        "type": "stream_end",
        "incidents_sent": index + 1 if incidents else 0,
    })
