"""Demo-day GO/NO-GO checks for offline and Razorpay test-mode paths."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from .razorpay_integration import credential_readiness, get_gateway, load_environment


ROOT = Path(__file__).resolve().parents[1]


def _line(ok: bool, label: str, detail: str) -> None:
    print(f"[{'GO' if ok else 'NO-GO'}] {label}: {detail}")


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Return a failure exit code unless the Razorpay test API ping succeeds.",
    )
    args = parser.parse_args()

    dataset_ok = (ROOT / "data" / "incidents.json").is_file()
    results_ok = (ROOT / "results.json").is_file()
    fastapi_ok = importlib.util.find_spec("fastapi") is not None
    razorpay_ok = importlib.util.find_spec("razorpay") is not None
    dotenv_ok = importlib.util.find_spec("dotenv") is not None
    readiness = credential_readiness()

    _line(dataset_ok, "Synthetic dataset", "data/incidents.json is present" if dataset_ok else "missing")
    _line(results_ok, "Evaluation snapshot", "results.json is present" if results_ok else "missing")
    _line(fastapi_ok, "Demo server", "FastAPI is installed" if fastapi_ok else "FastAPI is missing")
    _line(dotenv_ok, "Environment loader", "python-dotenv is installed" if dotenv_ok else "python-dotenv is missing")
    _line(razorpay_ok, "Official SDK", "razorpay is installed" if razorpay_ok else "razorpay is missing")
    _line(
        readiness["key_id_present"] and readiness["key_secret_present"],
        "Test credentials",
        "key ID and secret are set" if readiness["key_id_present"] and readiness["key_secret_present"] else "RAZORPAY_KEY_ID/RAZORPAY_KEY_SECRET missing",
    )
    _line(
        readiness["test_key_format_valid"],
        "Test-key safety gate",
        "key ID uses rzp_test_ prefix" if readiness["test_key_format_valid"] else "no valid rzp_test_ key ID",
    )
    _line(
        readiness["webhook_secret_present"],
        "Webhook verification",
        "RAZORPAY_WEBHOOK_SECRET is set" if readiness["webhook_secret_present"] else "webhook secret missing (Payment Link creation can still work)",
    )

    ping_ok = False
    if razorpay_ok and readiness["ready_for_test_api"]:
        try:
            get_gateway().ping()
            ping_ok = True
            _line(True, "Razorpay test API", "authenticated Payment Link list call succeeded")
        except Exception as exc:
            _line(False, "Razorpay test API", f"ping failed ({type(exc).__name__}); no secret was logged")
    else:
        _line(False, "Razorpay test API", "ping skipped because SDK or test credentials are missing")

    offline_go = dataset_ok and results_ok and fastapi_ok
    live_go = offline_go and dotenv_ok and razorpay_ok and readiness["ready_for_test_api"] and ping_ok
    print()
    print(f"OFFLINE FALLBACK DEMO: {'GO' if offline_go else 'NO-GO'}")
    print(f"RAZORPAY LIVE TEST-MODE DEMO: {'GO' if live_go else 'NO-GO'}")
    print("TEST MODE ONLY: no production money movement is performed by this project.")
    if not offline_go or (args.require_live and not live_go):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
