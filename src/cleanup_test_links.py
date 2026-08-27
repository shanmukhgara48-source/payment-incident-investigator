"""List or cancel Payment Links created by this demo in Razorpay TEST MODE."""

from __future__ import annotations

import argparse
import logging

from .logging_config import configure_logging
from .razorpay_integration import credential_readiness, get_gateway, load_environment
from .test_link_registry import list_links


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Cancel registered active test links. Without this flag, only list them.",
    )
    args = parser.parse_args()
    configure_logging()
    logger = logging.getLogger(__name__)

    active = list_links(active_only=True)
    if not active:
        logger.info(
            "no active registered Razorpay test Payment Links found",
            extra={"incident_id": "batch", "stage": "cleanup"},
        )
        return
    logger.info(
        "registered active test links=%s execute=%s",
        len(active),
        args.execute,
        extra={"incident_id": "batch", "stage": "cleanup"},
    )
    for link in active:
        logger.info(
            "test link link_id=%s incident=%s status=%s",
            link["id"],
            link["incident_id"],
            link["status"],
            extra={"incident_id": link["incident_id"], "stage": "cleanup"},
        )
    if not args.execute:
        logger.info(
            "dry run only; rerun with --execute to cancel these TEST MODE links",
            extra={"incident_id": "batch", "stage": "cleanup"},
        )
        return

    readiness = credential_readiness()
    if not readiness["ready_for_test_api"]:
        parser.error("valid Razorpay TEST MODE credentials are required for cleanup")

    failures = 0
    for link in active:
        try:
            response = get_gateway().cancel_link(link["id"])
            logger.info(
                "test Payment Link cancelled link_id=%s status=%s",
                link["id"],
                response.get("status", "cancelled"),
                extra={"incident_id": link["incident_id"], "stage": "cleanup"},
            )
        except Exception as exc:
            failures += 1
            logger.warning(
                "test Payment Link cancellation failed link_id=%s error_type=%s",
                link["id"],
                type(exc).__name__,
                extra={"incident_id": link["incident_id"], "stage": "cleanup"},
            )
    if failures:
        raise SystemExit(f"cleanup incomplete: {failures} test link(s) could not be cancelled")


if __name__ == "__main__":
    main()
