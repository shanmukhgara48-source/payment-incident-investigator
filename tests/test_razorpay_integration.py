import hashlib
import hmac
from types import SimpleNamespace

import pytest

from src.config import MAX_REAL_LINKS_PER_DEMO_RUN
from src.razorpay_integration import RazorpayTestGateway, integration_status


class FakePaymentLinks:
    def __init__(self):
        self.created = []

    def create(self, payload):
        self.created.append(payload)
        index = len(self.created)
        return {
            "id": f"plink_test_{index}",
            "short_url": f"https://rzp.io/i/test{index}",
            "status": "created",
            "expire_by": payload["expire_by"],
        }


def eligible_payments(count=8):
    return [
        {"payment_id": f"pay_test_{index}", "amount_inr": 1000 + index}
        for index in range(count)
    ]


def test_real_link_creation_is_test_labeled_and_globally_capped(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LIVE_API_MODE", "true")
    monkeypatch.setattr("src.test_link_registry.REGISTRY_PATH", tmp_path / "links.json")
    payment_links = FakePaymentLinks()
    client = SimpleNamespace(payment_link=payment_links)
    gateway = RazorpayTestGateway(client=client, sleeper=lambda _: None)

    first = gateway.create_recovery_links("INC-0001", eligible_payments())
    second = gateway.create_recovery_links("INC-0002", eligible_payments())

    assert first["mode"] == "LIVE TEST-MODE"
    assert len(first["links"]) == MAX_REAL_LINKS_PER_DEMO_RUN
    assert second["links"] == []
    assert len(payment_links.created) == MAX_REAL_LINKS_PER_DEMO_RUN
    assert all(item["description"].startswith("TEST MODE recovery") for item in payment_links.created)
    assert all(item["notify"] == {"sms": False, "email": False} for item in payment_links.created)


def test_missing_credentials_falls_back_without_an_api_call(monkeypatch):
    monkeypatch.setenv("LIVE_API_MODE", "true")
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)
    gateway = RazorpayTestGateway()

    outcome = gateway.create_recovery_links("INC-0001", eligible_payments(1))

    assert outcome["mode"] == "SIMULATED"
    assert outcome["live_api_requested"] is True
    assert outcome["links"] == []
    assert "simulated recovery" in outcome["fallback_reason"]


def test_live_mode_refuses_non_test_key(monkeypatch):
    monkeypatch.setenv("LIVE_API_MODE", "true")
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_live_forbidden")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "not-a-real-secret")
    gateway = RazorpayTestGateway()

    outcome = gateway.create_recovery_links("INC-0001", eligible_payments(1))

    assert outcome["mode"] == "SIMULATED"
    assert outcome["links"] == []


def test_failed_calls_also_consume_the_per_run_request_budget(monkeypatch):
    monkeypatch.setenv("LIVE_API_MODE", "true")

    class FailingPaymentLinks:
        def __init__(self):
            self.attempts = 0

        def create(self, _payload):
            self.attempts += 1
            raise ConnectionError("synthetic test outage")

    payment_links = FailingPaymentLinks()
    client = SimpleNamespace(payment_link=payment_links)
    gateway = RazorpayTestGateway(client=client, sleeper=lambda _: None)

    first = gateway.create_recovery_links("INC-0001", eligible_payments())
    second = gateway.create_recovery_links("INC-0002", eligible_payments())

    assert first["links"] == []
    assert second["links"] == []
    assert payment_links.attempts == MAX_REAL_LINKS_PER_DEMO_RUN


def test_official_sdk_webhook_signature_verification(monkeypatch):
    secret = "local-webhook-test-secret"
    body = b'{"event":"payment_link.paid"}'
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_placeholder")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "placeholder")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", secret)
    gateway = RazorpayTestGateway()

    gateway.verify_webhook_signature(body, signature)
    with pytest.raises(Exception):
        gateway.verify_webhook_signature(body, "invalid-signature")


def test_mode_label_stays_simulated_when_live_api_is_requested_without_credentials(
    monkeypatch,
):
    """--live-api is a request, not a capability: no keys means nothing was live."""
    monkeypatch.setenv("LIVE_API_MODE", "true")
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)

    status = integration_status()

    assert status["live_api_requested"] is True
    assert status["ready_for_test_api"] is False
    assert status["mode_label"] == "SIMULATED"


def test_mode_label_is_live_test_mode_only_with_a_usable_test_key_pair(monkeypatch):
    monkeypatch.setenv("LIVE_API_MODE", "true")
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_placeholder")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "placeholder")

    status = integration_status()

    assert status["ready_for_test_api"] is True
    assert status["mode_label"] == "LIVE TEST-MODE"


def test_mode_label_never_claims_live_for_a_production_key(monkeypatch):
    monkeypatch.setenv("LIVE_API_MODE", "true")
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_live_placeholder")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "placeholder")

    status = integration_status()

    assert status["mode_label"] == "SIMULATED"
