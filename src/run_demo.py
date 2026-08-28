"""One-command launcher for the complete buildathon demo."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from data.simulate import AMBIGUOUS_FRACTION, DEFAULT_INCIDENT_COUNT, SEED, generate_dataset

from .evaluate import DEFAULT_DATA, DEFAULT_RESULTS, evaluate
from .io_utils import write_json_atomic
from .logging_config import configure_logging
from .razorpay_integration import integration_status, load_environment


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_INCIDENT_COUNT)
    parser.add_argument("--ambiguous-ratio", type=float, default=AMBIGUOUS_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--live-api",
        action="store_true",
        help="Explicitly enable capped Razorpay TEST MODE Payment Link creation.",
    )
    args = parser.parse_args()

    if args.live_api:
        os.environ["LIVE_API_MODE"] = "true"

    configure_logging()
    logger = logging.getLogger(__name__)
    try:
        dataset = generate_dataset(args.count, args.ambiguous_ratio, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    write_json_atomic(Path(DEFAULT_DATA), dataset)
    results = evaluate(Path(DEFAULT_DATA), Path(DEFAULT_RESULTS))

    logger.info(
        "demo data ready incidents=%s detection_accuracy=%.1f%%",
        len(results["incidents"]),
        results["aggregate_metrics"]["detection_accuracy"] * 100,
        extra={"incident_id": "batch", "stage": "startup"},
    )
    logger.info(
        "dashboard URL http://%s:%s/",
        args.host,
        args.port,
        extra={"incident_id": "system", "stage": "startup"},
    )
    status = integration_status()
    logger.info(
        "recovery integration mode=%s requested=%s ready_for_test_api=%s test_mode_only=true",
        status["mode_label"],
        status["live_api_requested"],
        status["ready_for_test_api"],
        extra={"incident_id": "system", "stage": "startup"},
    )

    try:
        import uvicorn
    except ImportError as exc:
        parser.error("uvicorn is not installed; run: python -m pip install -r requirements.txt")
        raise exc

    # Without an installed WebSocket library uvicorn silently downgrades to
    # ws="none": it answers the /ws/live upgrade as a plain HTTP GET, which no
    # route matches, so the live stream 404s at demo time. Fail loudly instead.
    from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol

    if AutoWebSocketsProtocol is None:
        parser.error(
            "no WebSocket library installed, so /ws/live would return 404 under uvicorn; "
            "run: python -m pip install -r requirements.txt"
        )

    from .api import app

    uvicorn.run(app, host=args.host, port=args.port, log_config=None, access_log=True)


if __name__ == "__main__":
    main()
