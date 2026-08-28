"""Poll Razorpay TEST MODE Payment Link status and reconcile paid links.

This is the fallback for environments with no public webhook tunnel. It does
NOT verify a webhook signature, because there is no webhook: it reads link
status directly from the authenticated Razorpay test API over TLS. Every record
it writes is labelled AUTHENTICATED_TEST_API_POLL rather than
VERIFIED_WEBHOOK_SIGNATURE so the audit trail never overstates its evidence.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

from .api import ResultStore
from .logging_config import configure_logging
from .razorpay_integration import credential_readiness, get_gateway, load_environment
from .test_link_registry import (
    list_links,
    mark_webhook_processed,
    webhook_was_processed,
)


EVIDENCE_SOURCE = "AUTHENTICATED_TEST_API_POLL"


def _occurred_at(payment: dict | None) -> str:
    epoch = (payment or {}).get("created_at")
    if isinstance(epoch, int):
        return datetime.fromtimestamp(epoch, UTC).isoformat()
    return datetime.now(UTC).isoformat()


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report remote link status without writing the reconciliation.",
    )
    args = parser.parse_args()
    configure_logging()
    logger = logging.getLogger(__name__)

    readiness = credential_readiness()
    if not readiness["ready_for_test_api"]:
        parser.error("valid Razorpay TEST MODE credentials are required to poll link status")

    links = list_links()
    if not links:
        logger.info(
            "no registered test Payment Links to reconcile",
            extra={"incident_id": "batch", "stage": "reconcile"},
        )
        return

    store = ResultStore()
    store.load()

    reconciled = 0
    for link in links:
        link_id = link["id"]
        incident_id = link["incident_id"]
        try:
            remote = get_gateway().fetch_link(link_id)
        except Exception as exc:
            logger.warning(
                "status poll failed link_id=%s error_type=%s",
                link_id,
                type(exc).__name__,
                extra={"incident_id": incident_id, "stage": "reconcile"},
            )
            continue

        status = remote.get("status")
        amount_paid_inr = int(remote.get("amount_paid") or 0) // 100
        logger.info(
            "polled link_id=%s remote_status=%s amount_paid_inr=%s",
            link_id,
            status,
            amount_paid_inr,
            extra={"incident_id": incident_id, "stage": "reconcile"},
        )

        if status != "paid" or amount_paid_inr <= 0:
            continue

        payments = remote.get("payments") or []
        captured = next(
            (item for item in payments if item.get("status") in {"captured", "paid"}),
            None,
        )
        payment_id = (captured or {}).get("payment_id", link_id)
        # Idempotency key mirrors the webhook path's event-ID de-duplication so
        # re-running this command cannot double-count a recovery.
        event_id = f"apipoll_{link_id}_{payment_id}"
        if webhook_was_processed(event_id):
            logger.info(
                "already reconciled event_id=%s; skipping",
                event_id,
                extra={"incident_id": incident_id, "stage": "reconcile"},
            )
            continue

        if args.dry_run:
            logger.info(
                "dry run: would reconcile incident=%s link_id=%s amount_inr=%s",
                incident_id,
                link_id,
                amount_paid_inr,
                extra={"incident_id": incident_id, "stage": "reconcile"},
            )
            continue

        applied = store.apply_test_webhook(
            incident_id=incident_id,
            link_id=link_id,
            event_id=event_id,
            event_type="payment_link.paid",
            amount_inr=amount_paid_inr,
            occurred_at=_occurred_at(captured),
            evidence_source=EVIDENCE_SOURCE,
            source_label="status poll",
        )
        if applied:
            mark_webhook_processed(event_id)
            reconciled += 1
            logger.info(
                "reconciled incident=%s link_id=%s payment_id=%s amount_inr=%s basis=%s",
                incident_id,
                link_id,
                payment_id,
                amount_paid_inr,
                EVIDENCE_SOURCE,
                extra={"incident_id": incident_id, "stage": "reconcile"},
            )
        else:
            logger.warning(
                "snapshot rejected the reconciliation link_id=%s",
                link_id,
                extra={"incident_id": incident_id, "stage": "reconcile"},
            )

    logger.info(
        "reconcile complete links_polled=%s incidents_updated=%s",
        len(links),
        reconciled,
        extra={"incident_id": "batch", "stage": "reconcile"},
    )


if __name__ == "__main__":
    main()
