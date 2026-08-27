"""Durable local registry for Razorpay test Payment Links and webhooks."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .io_utils import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data" / "created_test_links.json"
_LOCK = threading.RLock()


def _empty_registry() -> dict[str, Any]:
    return {"links": [], "processed_webhook_event_ids": []}


def _read() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return _empty_registry()
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_registry()
    if not isinstance(payload.get("links"), list):
        payload["links"] = []
    if not isinstance(payload.get("processed_webhook_event_ids"), list):
        payload["processed_webhook_event_ids"] = []
    return payload


def record_link(link: dict[str, Any]) -> None:
    with _LOCK:
        payload = _read()
        payload["links"] = [item for item in payload["links"] if item.get("id") != link["id"]]
        payload["links"].append(link)
        write_json_atomic(REGISTRY_PATH, payload)


def list_links(*, active_only: bool = False) -> list[dict[str, Any]]:
    with _LOCK:
        links = list(_read()["links"])
    if active_only:
        return [item for item in links if item.get("status") in {"created", "issued"}]
    return links


def find_link(link_id: str) -> dict[str, Any] | None:
    return next((item for item in list_links() if item.get("id") == link_id), None)


def update_link(link_id: str, **changes: Any) -> dict[str, Any] | None:
    with _LOCK:
        payload = _read()
        updated = None
        for item in payload["links"]:
            if item.get("id") == link_id:
                item.update(changes)
                updated = dict(item)
                break
        if updated is not None:
            write_json_atomic(REGISTRY_PATH, payload)
        return updated


def webhook_was_processed(event_id: str) -> bool:
    with _LOCK:
        return event_id in _read()["processed_webhook_event_ids"]


def mark_webhook_processed(event_id: str) -> None:
    with _LOCK:
        payload = _read()
        if event_id not in payload["processed_webhook_event_ids"]:
            payload["processed_webhook_event_ids"].append(event_id)
            payload["processed_webhook_event_ids"] = payload[
                "processed_webhook_event_ids"
            ][-1000:]
            write_json_atomic(REGISTRY_PATH, payload)
