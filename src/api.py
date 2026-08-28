"""FastAPI surface for the live incident investigation demo."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import threading
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from data.simulate import (
    AMBIGUOUS_FRACTION,
    DEFAULT_INCIDENT_COUNT,
    MAX_AMBIGUOUS_FRACTION,
    MAX_INCIDENT_COUNT,
    MIN_AMBIGUOUS_FRACTION,
    MIN_INCIDENT_COUNT,
    generate_dataset,
)

from .counterfactual import MIN_DELAY_MINUTES, MAX_DELAY_MINUTES, estimate_gmv_saved
from .live_stream import stream_incidents, DEFAULT_INTERVAL_SECONDS, LIVE_BATCH_SIZE
from .evaluate import evaluate
from .postmortem import generate_postmortem
from .io_utils import write_json_atomic
from .logging_config import configure_logging
from .razorpay_integration import get_gateway, integration_status
from .test_link_registry import (
    find_link,
    list_links,
    mark_webhook_processed,
    update_link,
    webhook_was_processed,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "incidents.json"
RESULTS_PATH = ROOT / "results.json"
UI_PATH = ROOT / "ui"
INCIDENT_ID_PATTERN = re.compile(r"^INC-[0-9]{4}$")
WEBHOOK_EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
MAX_WEBHOOK_BODY_BYTES = 1_000_000

configure_logging()
logger = logging.getLogger(__name__)


class SimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_count: int = Field(
        default=DEFAULT_INCIDENT_COUNT,
        ge=MIN_INCIDENT_COUNT,
        le=MAX_INCIDENT_COUNT,
        description="Number of synthetic incidents in the regenerated batch.",
    )
    ambiguous_ratio: float = Field(
        default=AMBIGUOUS_FRACTION,
        ge=MIN_AMBIGUOUS_FRACTION,
        le=MAX_AMBIGUOUS_FRACTION,
        description="Fraction of incidents intentionally left without a clean root-cause signal.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2_147_483_647,
        description="Optional reproducibility seed. Omit it for a fresh batch.",
    )


class ResultStore:
    """Thread-safe snapshot used by API reads during batch regeneration."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._regenerate_lock = threading.Lock()
        self._results: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        if not RESULTS_PATH.exists():
            return self.regenerate(DEFAULT_INCIDENT_COUNT, AMBIGUOUS_FRACTION, None)
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if "aggregate_metrics" not in results or "incidents" not in results:
            raise ValueError("results.json does not contain a complete evaluation snapshot")
        with self._lock:
            self._results = results
        logger.info(
            "result snapshot loaded incidents=%s",
            len(results["incidents"]),
            extra={"incident_id": "batch", "stage": "api_store"},
        )
        return results

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._results is None:
                raise RuntimeError("result snapshot is not ready")
            return self._results

    def regenerate(
        self, incident_count: int, ambiguous_ratio: float, seed: int | None
    ) -> dict[str, Any]:
        with self._regenerate_lock:
            selected_seed = seed if seed is not None else secrets.randbelow(2_147_483_648)
            logger.info(
                "simulation started count=%s ambiguous_ratio=%.3f seed=%s",
                incident_count,
                ambiguous_ratio,
                selected_seed,
                extra={"incident_id": "batch", "stage": "simulate"},
            )
            dataset = generate_dataset(incident_count, ambiguous_ratio, selected_seed)
            write_json_atomic(DATA_PATH, dataset)
            results = evaluate(DATA_PATH, RESULTS_PATH)
            with self._lock:
                self._results = results
            logger.info(
                "simulation completed incidents=%s",
                len(results["incidents"]),
                extra={"incident_id": "batch", "stage": "simulate"},
            )
            return results

    def apply_test_webhook(
        self,
        *,
        incident_id: str,
        link_id: str,
        event_id: str,
        event_type: str,
        amount_inr: int,
        occurred_at: str,
        evidence_source: str = "VERIFIED_WEBHOOK_SIGNATURE",
        source_label: str = "webhook",
    ) -> bool:
        """Apply one confirmed test-mode event and persist the new snapshot.

        `evidence_source` is written verbatim into the audit trail's bounded_by
        field. The signed-webhook path leaves the default; the API status-poll
        fallback passes its own label so the trail never claims a signature
        verification that did not happen.
        """

        with self._lock:
            if self._results is None:
                raise RuntimeError("result snapshot is not ready")
            results = deepcopy(self._results)
            record = next(
                (item for item in results["incidents"] if item["incident_id"] == incident_id),
                None,
            )
            if record is None:
                return False

            recovery = record["recovery"]
            impact = record["impact"]
            current_link_ids = {
                item.get("id") for item in recovery.get("test_payment_links", [])
            }
            if link_id not in current_link_ids:
                logger.warning(
                    "verified webhook link is not owned by the current incident snapshot",
                    extra={"incident_id": incident_id, "stage": "webhook_apply"},
                )
                return False
            events = recovery.setdefault("actual_recovery_events", [])
            events.append(
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "payment_link_id": link_id,
                    "amount_inr": amount_inr,
                    "timestamp": occurred_at,
                    "mode": "LIVE TEST-MODE",
                }
            )
            recovery["recovery_mode"] = "LIVE TEST-MODE"
            recovery["live_api_requested"] = True

            action = f"LIVE TEST-MODE {source_label}: {event_type}"
            reason = f"Confirmed via Razorpay test {source_label} for Payment Link {link_id}."
            if event_type in {"payment.captured", "payment_link.paid"}:
                by_link = recovery.setdefault("actual_recovered_by_link_inr", {})
                by_link[link_id] = max(amount_inr, int(by_link.get(link_id, 0)))
                actual_total = sum(int(value) for value in by_link.values())
                impact.setdefault("modeled_recovered_amount_inr", impact["recovered_amount_inr"])
                impact["actual_recovered_amount_inr"] = actual_total
                impact["recovered_amount_inr"] = actual_total
                impact["recovery_measurement_type"] = "ACTUAL TEST-MODE"
                impact["recovered_amount_basis"] = (
                    f"ACTUAL TEST-MODE: confirmed via Razorpay test {source_label}; "
                    "no production money moved."
                )
                recovery["actual_recovered_amount_inr"] = actual_total
                record["timeline"] = [
                    marker for marker in record["timeline"] if marker.get("kind") != "recovery"
                ]
                record["timeline"].append(
                    {
                        "timestamp": occurred_at,
                        "kind": "recovery",
                        "label": f"Actual TEST-MODE recovery: INR {actual_total:,}",
                    }
                )
                update_link(link_id, status="paid", paid_amount_inr=amount_inr)
            elif event_type == "payment.failed":
                update_link(link_id, status="payment_failed")

            audit_entry = {
                "incident_id": incident_id,
                "action": action,
                "reason": reason,
                "bounded_by": f"{evidence_source}, TEST_MODE_ONLY",
                "timestamp": occurred_at,
                "metadata": {
                    "mode": "LIVE TEST-MODE",
                    "payment_link_id": link_id,
                    "evidence_event_id": event_id,
                    "amount_inr": amount_inr,
                },
            }
            recovery["audit_trail"].append(audit_entry)
            record["audit_trail"] = recovery["audit_trail"]

            aggregate = results["aggregate_metrics"]
            aggregate["total_recovered_amount_inr"] = sum(
                item["impact"]["recovered_amount_inr"] for item in results["incidents"]
            )
            retry_eligible = [
                item for item in results["incidents"]
                if item["recovery"]["primary_action"] == "create Payment Links for high-intent failures"
            ]
            aggregate["total_retry_recovered_amount_inr"] = sum(
                item["impact"].get("retry_recovered_amount_inr", item["impact"].get("recovered_amount_inr", 0))
                for item in retry_eligible
            )
            aggregate["retry_eligible_incident_count"] = len(retry_eligible)
            actual_count = sum(
                item["impact"].get("recovery_measurement_type") == "ACTUAL TEST-MODE"
                for item in results["incidents"]
            )
            aggregate["actual_test_mode_recovery_incident_count"] = actual_count
            if actual_count:
                aggregate["recovered_amount_basis"] = (
                    "MIXED: confirmed actual Razorpay TEST-MODE amounts for labeled incidents; "
                    "all remaining amounts use the labeled modeling assumption. No production money moved."
                )
                aggregate["retry_recovered_amount_basis"] = aggregate["recovered_amount_basis"]
            write_json_atomic(RESULTS_PATH, results)
            self._results = results
            return True


store = ResultStore()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        store.load()
    except Exception:
        logger.exception(
            "failed to initialize result snapshot",
            extra={"incident_id": "system", "stage": "startup"},
        )
        raise
    yield


app = FastAPI(
    title="Payment Incident Investigator API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        results = store.snapshot()
    except RuntimeError:
        return {"status": "starting", "ready": False, "incident_count": 0}
    return {
        "status": "ok",
        "ready": True,
        "incident_count": len(results["incidents"]),
        "integration": integration_status(),
    }


@app.get("/api/incidents")
def list_incidents() -> list[dict[str, Any]]:
    try:
        records = store.snapshot()["incidents"]
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    summaries = []
    for record in records:
        primary = record["detection"].get("primary_degradation")
        summaries.append(
            {
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
        )
    return summaries


@app.get("/api/incidents/{incident_id}")
def incident_detail(incident_id: str) -> dict[str, Any]:
    if not INCIDENT_ID_PATTERN.fullmatch(incident_id):
        raise HTTPException(
            status_code=400,
            detail="incident id must match INC- followed by exactly four digits",
        )
    try:
        records = store.snapshot()["incidents"]
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    record = next((item for item in records if item["incident_id"] == incident_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"incident {incident_id!r} was not found")
    return record


@app.get("/api/incidents/{incident_id}/postmortem.md")
def incident_postmortem(incident_id: str) -> PlainTextResponse:
    if not INCIDENT_ID_PATTERN.fullmatch(incident_id):
        raise HTTPException(
            status_code=400,
            detail="incident id must match INC- followed by exactly four digits",
        )
    try:
        records = store.snapshot()["incidents"]
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    record = next((r for r in records if r["incident_id"] == incident_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"incident {incident_id!r} was not found")
    md = generate_postmortem(record)
    filename = f"{incident_id}-postmortem.md"
    return PlainTextResponse(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/incidents/{incident_id}/counterfactual")
def incident_counterfactual(incident_id: str, delay_minutes: int = 0) -> dict[str, Any]:
    if not INCIDENT_ID_PATTERN.fullmatch(incident_id):
        raise HTTPException(
            status_code=400,
            detail="incident id must match INC- followed by exactly four digits",
        )
    if not MIN_DELAY_MINUTES <= delay_minutes <= MAX_DELAY_MINUTES:
        raise HTTPException(
            status_code=400,
            detail=f"delay_minutes must be between {MIN_DELAY_MINUTES} and {MAX_DELAY_MINUTES}",
        )
    # Verify the incident exists in evaluated results
    try:
        records = store.snapshot()["incidents"]
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not any(r["incident_id"] == incident_id for r in records):
        raise HTTPException(status_code=404, detail=f"incident {incident_id!r} was not found")
    # Load raw incident data (contains payment_events and failure_by_minute)
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="raw incident data is not available")
    raw_dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    raw_incident = next(
        (inc for inc in raw_dataset["incidents"] if inc["incident_id"] == incident_id), None
    )
    if raw_incident is None:
        raise HTTPException(status_code=404, detail=f"raw data for {incident_id!r} was not found")
    return estimate_gmv_saved(raw_incident, delay_minutes)


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    try:
        results = store.snapshot()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        **results["aggregate_metrics"],
        "assumptions": results["assumptions"],
        "metadata": results["metadata"],
        "integration": integration_status(),
    }


@app.post("/api/simulate")
def simulate(payload: SimulateRequest) -> dict[str, Any]:
    try:
        results = store.regenerate(
            payload.incident_count,
            payload.ambiguous_ratio,
            payload.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "simulation request failed",
            extra={"incident_id": "batch", "stage": "simulate"},
        )
        raise HTTPException(status_code=500, detail="simulation failed; see server logs") from exc
    return {
        "status": "regenerated",
        "seed": results["metadata"]["seed"],
        "incident_count": results["aggregate_metrics"]["incident_count"],
        "summary": {
            **results["aggregate_metrics"],
            "assumptions": results["assumptions"],
        },
    }


def _extract_webhook_entities(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    body_payload = payload.get("payload", {})
    payment_link = body_payload.get("payment_link", {}).get("entity", {})
    payment = body_payload.get("payment", {}).get("entity", {})
    return payment_link if isinstance(payment_link, dict) else {}, payment if isinstance(payment, dict) else {}


@app.post("/api/webhooks/razorpay")
async def razorpay_webhook(request: Request) -> dict[str, Any]:
    signature = request.headers.get("x-razorpay-signature", "").strip()
    event_id = request.headers.get("x-razorpay-event-id", "").strip()
    if not signature or len(signature) > 256:
        raise HTTPException(status_code=400, detail="valid X-Razorpay-Signature header required")
    if not WEBHOOK_EVENT_ID_PATTERN.fullmatch(event_id):
        raise HTTPException(status_code=400, detail="valid X-Razorpay-Event-Id header required")
    if webhook_was_processed(event_id):
        return {"status": "duplicate", "event_id": event_id}

    raw_body = await request.body()
    if not raw_body or len(raw_body) > MAX_WEBHOOK_BODY_BYTES:
        raise HTTPException(status_code=400, detail="webhook body is empty or exceeds 1 MB")
    try:
        get_gateway().verify_webhook_signature(raw_body, signature)
    except RuntimeError as exc:
        logger.warning(
            "webhook verification unavailable error_type=%s",
            type(exc).__name__,
            extra={"incident_id": "webhook", "stage": "webhook_verify"},
        )
        raise HTTPException(status_code=503, detail="webhook verification is not configured") from exc
    except Exception as exc:
        logger.warning(
            "webhook signature rejected error_type=%s",
            type(exc).__name__,
            extra={"incident_id": "webhook", "stage": "webhook_verify"},
        )
        raise HTTPException(status_code=400, detail="invalid webhook signature") from exc

    try:
        payload = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="webhook body must be valid JSON") from exc
    event_type = payload.get("event")
    if event_type not in {"payment.captured", "payment.failed", "payment_link.paid"}:
        mark_webhook_processed(event_id)
        return {"status": "ignored", "event_id": event_id, "event_type": event_type}

    payment_link, payment = _extract_webhook_entities(payload)
    candidate_ids = [
        payment_link.get("id"),
        payment.get("invoice_id"),
        payment.get("notes", {}).get("payment_link_id")
        if isinstance(payment.get("notes"), dict)
        else None,
    ]
    link = next((find_link(value) for value in candidate_ids if isinstance(value, str)), None)
    payment_notes = payment.get("notes") if isinstance(payment.get("notes"), dict) else {}
    if link is None and payment_notes.get("incident_id"):
        link = next(
            (
                item
                for item in list_links(active_only=True)
                if item.get("incident_id") == payment_notes["incident_id"]
                and (
                    not payment_notes.get("source_payment_id")
                    or item.get("source_payment_id") == payment_notes["source_payment_id"]
                )
            ),
            None,
        )
    if link is None:
        mark_webhook_processed(event_id)
        return {"status": "unmatched", "event_id": event_id, "event_type": event_type}

    amount_paise = (
        payment_link.get("amount_paid")
        or payment_link.get("amount")
        or payment.get("amount")
        or 0
    )
    if not isinstance(amount_paise, int) or amount_paise < 0:
        raise HTTPException(status_code=400, detail="webhook amount must be a non-negative integer")
    occurred_at = datetime.fromtimestamp(
        int(payment_link.get("updated_at") or payment.get("created_at") or datetime.now(UTC).timestamp()),
        tz=UTC,
    ).isoformat()
    updated = store.apply_test_webhook(
        incident_id=link["incident_id"],
        link_id=link["id"],
        event_id=event_id,
        event_type=event_type,
        amount_inr=round(amount_paise / 100),
        occurred_at=occurred_at,
    )
    mark_webhook_processed(event_id)
    if not updated:
        return {"status": "unmatched_incident", "event_id": event_id}
    logger.info(
        "verified Razorpay test webhook applied event_type=%s link_id=%s",
        event_type,
        link["id"],
        extra={"incident_id": link["incident_id"], "stage": "webhook_apply"},
    )
    return {
        "status": "applied",
        "event_id": event_id,
        "event_type": event_type,
        "incident_id": link["incident_id"],
        "mode": "LIVE TEST-MODE",
    }


@app.websocket("/ws/live")
async def live_stream(ws: WebSocket):
    await ws.accept()
    logger.info(
        "live stream websocket connected",
        extra={"incident_id": "live", "stage": "websocket"},
    )
    stop = asyncio.Event()

    async def send_json(data: dict) -> None:
        await ws.send_json(data)

    try:
        # Parse optional config from query params.
        params = ws.query_params
        count = min(int(params.get("count", LIVE_BATCH_SIZE)), 60)
        count = max(count, 10)  # generate_dataset requires >= 10
        interval = max(float(params.get("interval", DEFAULT_INTERVAL_SECONDS)), 0.05)

        await stream_incidents(
            send_json,
            incident_count=count,
            interval=interval,
            stop_event=stop,
        )
    except WebSocketDisconnect:
        logger.info(
            "live stream websocket disconnected by client",
            extra={"incident_id": "live", "stage": "websocket"},
        )
    except Exception as exc:
        logger.exception(
            "live stream error",
            extra={"incident_id": "live", "stage": "websocket"},
        )
        try:
            await ws.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        stop.set()
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(UI_PATH / "timeline.html")


@app.get("/timeline", include_in_schema=False)
def timeline_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui/timeline.html")


app.mount("/ui", StaticFiles(directory=UI_PATH, html=True), name="ui")
