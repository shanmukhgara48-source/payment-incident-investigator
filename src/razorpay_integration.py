"""Strictly gated Razorpay TEST MODE adapter for recovery side effects."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Callable

from .config import (
    MAX_REAL_LINKS_PER_DEMO_RUN,
    MAX_REAL_LINKS_PER_INCIDENT,
    PAYMENT_LINK_EXPIRY_MINUTES,
    RAZORPAY_TEST_KEY_PREFIX,
    REAL_API_CALL_INTERVAL_SECONDS,
    live_api_mode_enabled,
)
from .test_link_registry import record_link, update_link


logger = logging.getLogger(__name__)
_CLIENT_IMPORT_ERROR = "The razorpay package is not installed."


def load_environment() -> None:
    """Load local environment values without making dotenv an offline dependency."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def credential_readiness() -> dict[str, Any]:
    load_environment()
    key_id = os.getenv("RAZORPAY_KEY_ID", "").strip()
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "").strip()
    webhook_secret = os.getenv("RAZORPAY_WEBHOOK_SECRET", "").strip()
    is_test_key = bool(key_id) and key_id.startswith(RAZORPAY_TEST_KEY_PREFIX)
    return {
        "live_api_requested": live_api_mode_enabled(),
        "key_id_present": bool(key_id),
        "key_secret_present": bool(key_secret),
        "webhook_secret_present": bool(webhook_secret),
        "test_key_format_valid": is_test_key,
        "ready_for_test_api": bool(is_test_key and key_secret),
        "mode_label": "LIVE TEST-MODE" if live_api_mode_enabled() else "SIMULATED",
    }


def integration_status() -> dict[str, Any]:
    readiness = credential_readiness()
    return {
        **readiness,
        "test_mode_only": True,
        "production_money_movement": False,
        "max_real_links_per_incident": MAX_REAL_LINKS_PER_INCIDENT,
        "max_real_links_per_demo_run": MAX_REAL_LINKS_PER_DEMO_RUN,
        "minimum_seconds_between_calls": REAL_API_CALL_INTERVAL_SECONDS,
    }


class RazorpayTestGateway:
    """A capped, throttled SDK wrapper that refuses non-test credentials."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        load_environment()
        self._client: Any | None = client
        self._sleeper = sleeper
        self._clock = clock
        self._throttle_lock = threading.Lock()
        self._last_call_started = 0.0
        self._attempted_create_count = 0

    def _credentials(self) -> tuple[str, str]:
        key_id = os.getenv("RAZORPAY_KEY_ID", "").strip()
        key_secret = os.getenv("RAZORPAY_KEY_SECRET", "").strip()
        if not key_id or not key_secret:
            raise RuntimeError("Razorpay test credentials are missing")
        if not key_id.startswith(RAZORPAY_TEST_KEY_PREFIX):
            raise RuntimeError("Refusing a Razorpay key that is not explicitly test-mode")
        return key_id, key_secret

    def _sdk_client(self) -> Any:
        if self._client is not None:
            return self._client
        key_id, key_secret = self._credentials()
        try:
            import razorpay
        except ImportError as exc:
            raise RuntimeError(_CLIENT_IMPORT_ERROR) from exc
        self._client = razorpay.Client(auth=(key_id, key_secret))
        return self._client

    def _call(self, operation: Callable[[], Any]) -> Any:
        with self._throttle_lock:
            elapsed = self._clock() - self._last_call_started
            delay = max(0.0, REAL_API_CALL_INTERVAL_SECONDS - elapsed)
            if self._last_call_started and delay:
                self._sleeper(delay)
            self._last_call_started = self._clock()
            return operation()

    def create_recovery_links(
        self, incident_id: str, eligible_payments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not live_api_mode_enabled():
            return {
                "mode": "SIMULATED",
                "live_api_requested": False,
                "links": [],
                "fallback_reason": "LIVE_API_MODE is off; no Razorpay API call was made.",
            }

        try:
            client = self._sdk_client()
        except Exception as exc:
            logger.warning(
                "Razorpay test-mode adapter unavailable; using simulated recovery error_type=%s",
                type(exc).__name__,
                extra={"incident_id": incident_id, "stage": "razorpay_test_api"},
            )
            return {
                "mode": "SIMULATED",
                "live_api_requested": True,
                "links": [],
                "fallback_reason": "Test API prerequisites unavailable; simulated recovery used.",
            }

        remaining = max(0, MAX_REAL_LINKS_PER_DEMO_RUN - self._attempted_create_count)
        sample = eligible_payments[: min(MAX_REAL_LINKS_PER_INCIDENT, remaining)]
        if not sample:
            return {
                "mode": "SIMULATED",
                "live_api_requested": True,
                "links": [],
                "fallback_reason": "The configured per-demo real Payment Link cap was reached.",
            }

        created: list[dict[str, Any]] = []
        failed_calls = 0
        for payment in sample:
            reference_id = f"recovery_{incident_id.replace('-', '')}_{uuid.uuid4().hex[:10]}"
            expire_by = int(time.time() + PAYMENT_LINK_EXPIRY_MINUTES * 60)
            payload = {
                "amount": int(payment["amount_inr"]) * 100,
                "currency": "INR",
                "description": f"TEST MODE recovery for {incident_id}",
                "reference_id": reference_id,
                "expire_by": expire_by,
                "notify": {"sms": False, "email": False},
                "reminder_enable": False,
                "notes": {
                    "incident_id": incident_id,
                    "source_payment_id": payment["payment_id"],
                    "mode": "LIVE_TEST_MODE",
                },
            }
            try:
                self._attempted_create_count += 1
                response = self._call(lambda: client.payment_link.create(payload))
                link = {
                    "id": response["id"],
                    "short_url": response["short_url"],
                    "incident_id": incident_id,
                    "source_payment_id": payment["payment_id"],
                    "amount_inr": int(payment["amount_inr"]),
                    "status": response.get("status", "created"),
                    "expire_by": response.get("expire_by", expire_by),
                    "created_at": datetime.now(UTC).isoformat(),
                    "mode": "LIVE TEST-MODE",
                }
                record_link(link)
                created.append(link)
                logger.info(
                    "Razorpay test Payment Link created link_id=%s amount_inr=%s",
                    link["id"],
                    link["amount_inr"],
                    extra={"incident_id": incident_id, "stage": "razorpay_test_api"},
                )
            except Exception as exc:
                failed_calls += 1
                logger.warning(
                    "Razorpay test Payment Link creation failed error_type=%s; continuing",
                    type(exc).__name__,
                    extra={"incident_id": incident_id, "stage": "razorpay_test_api"},
                )

        return {
            "mode": "LIVE TEST-MODE" if created else "SIMULATED",
            "live_api_requested": True,
            "links": created,
            "fallback_reason": (
                None
                if created
                else "All attempted test API calls failed; simulated recovery used."
            ),
            "failed_api_call_count": failed_calls,
        }

    def verify_webhook_signature(self, body: bytes, signature: str) -> None:
        webhook_secret = os.getenv("RAZORPAY_WEBHOOK_SECRET", "").strip()
        if not webhook_secret:
            raise RuntimeError("RAZORPAY_WEBHOOK_SECRET is missing")
        client = self._sdk_client()
        client.utility.verify_webhook_signature(body.decode("utf-8"), signature, webhook_secret)

    def ping(self) -> None:
        client = self._sdk_client()
        self._call(lambda: client.payment_link.all({"count": 1}))

    def cancel_link(self, link_id: str) -> dict[str, Any]:
        client = self._sdk_client()
        response = self._call(lambda: client.payment_link.cancel(link_id))
        update_link(link_id, status=response.get("status", "cancelled"))
        return response


_gateway = RazorpayTestGateway()


def get_gateway() -> RazorpayTestGateway:
    return _gateway
