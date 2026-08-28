import json
import shutil

from fastapi.testclient import TestClient

from src import api as api_module
from src.api import app
from src.test_link_registry import record_link


def test_api_health_summary_list_and_detail():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["ready"] is True

        summary = client.get("/api/summary")
        assert summary.status_code == 200
        assert summary.json()["incident_count"] > 0
        assert summary.json()["integration"]["test_mode_only"] is True

        incidents = client.get("/api/incidents")
        assert incidents.status_code == 200
        first = incidents.json()[0]
        assert {"incident_id", "time", "method", "cause", "confidence", "status"} <= set(first)

        detail = client.get(f"/api/incidents/{first['incident_id']}")
        assert detail.status_code == 200
        assert {"rca_text", "correlation", "audit_trail", "impact"} <= set(detail.json())


def test_api_returns_clear_simulation_validation_errors():
    with TestClient(app) as client:
        too_few = client.post(
            "/api/simulate", json={"incident_count": 1, "ambiguous_ratio": 0.15}
        )
        assert too_few.status_code == 422
        assert "greater than or equal to 10" in too_few.text

        bad_ratio = client.post(
            "/api/simulate", json={"incident_count": 20, "ambiguous_ratio": 0.9}
        )
        assert bad_ratio.status_code == 422
        assert "less than or equal to 0.5" in bad_ratio.text


def test_missing_incident_is_a_clear_404():
    with TestClient(app) as client:
        malformed = client.get("/api/incidents/../../etc/passwd")
        assert malformed.status_code in {400, 404}

        malformed_format = client.get("/api/incidents/INC-NOT-A-NUMBER")
        assert malformed_format.status_code == 400

        response = client.get("/api/incidents/INC-9999")
        assert response.status_code == 404
        assert "was not found" in response.json()["detail"]


def test_verified_test_webhook_replaces_modeled_recovery_with_actual(
    monkeypatch, tmp_path
):
    temporary_results = tmp_path / "results.json"
    shutil.copyfile(api_module.ROOT / "results.json", temporary_results)
    test_results = json.loads(temporary_results.read_text(encoding="utf-8"))
    test_results["incidents"][0]["recovery"]["test_payment_links"] = [
        {
            "id": "plink_test_webhook_1",
            "short_url": "https://rzp.io/i/test123",
            "amount_inr": 1234,
            "mode": "LIVE TEST-MODE",
        }
    ]
    # This test's premise is a purely modeled incident receiving its FIRST
    # actual recovery. The checked-in snapshot may already carry a real
    # reconciliation, so reset that state instead of summing on top of it.
    seeded_recovery = test_results["incidents"][0]["recovery"]
    seeded_impact = test_results["incidents"][0]["impact"]
    for key in (
        "actual_recovered_by_link_inr",
        "actual_recovery_events",
        "actual_recovered_amount_inr",
    ):
        seeded_recovery.pop(key, None)
    seeded_impact.pop("actual_recovered_amount_inr", None)
    seeded_impact.pop("recovery_measurement_type", None)
    if "modeled_recovered_amount_inr" in seeded_impact:
        seeded_impact["recovered_amount_inr"] = seeded_impact.pop(
            "modeled_recovered_amount_inr"
        )
    temporary_results.write_text(json.dumps(test_results), encoding="utf-8")
    monkeypatch.setattr(api_module, "RESULTS_PATH", temporary_results)
    monkeypatch.setattr("src.test_link_registry.REGISTRY_PATH", tmp_path / "links.json")

    class Verifier:
        def verify_webhook_signature(self, body, signature):
            assert body
            assert signature == "valid-test-signature"

    monkeypatch.setattr(api_module, "get_gateway", lambda: Verifier())
    with TestClient(app) as client:
        first = client.get("/api/incidents").json()[0]
        incident_id = first["incident_id"]
        record_link(
            {
                "id": "plink_test_webhook_1",
                "short_url": "https://rzp.io/i/test123",
                "incident_id": incident_id,
                "source_payment_id": "pay_test_1",
                "amount_inr": 1234,
                "status": "created",
                "mode": "LIVE TEST-MODE",
            }
        )
        payload = {
            "event": "payment_link.paid",
            "payload": {
                "payment_link": {
                    "entity": {
                        "id": "plink_test_webhook_1",
                        "amount_paid": 123400,
                        "updated_at": 1_788_000_000,
                    }
                }
            },
        }
        response = client.post(
            "/api/webhooks/razorpay",
            content=json.dumps(payload),
            headers={
                "content-type": "application/json",
                "x-razorpay-signature": "valid-test-signature",
                "x-razorpay-event-id": "evt_test_webhook_1",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "applied"

        detail = client.get(f"/api/incidents/{incident_id}").json()
        assert detail["impact"]["recovered_amount_inr"] == 1234
        assert detail["impact"]["recovery_measurement_type"] == "ACTUAL TEST-MODE"
        assert detail["recovery"]["recovery_mode"] == "LIVE TEST-MODE"
        assert any("LIVE TEST-MODE webhook" in item["action"] for item in detail["audit_trail"])

        duplicate = client.post(
            "/api/webhooks/razorpay",
            content=json.dumps(payload),
            headers={
                "content-type": "application/json",
                "x-razorpay-signature": "valid-test-signature",
                "x-razorpay-event-id": "evt_test_webhook_1",
            },
        )
        assert duplicate.json()["status"] == "duplicate"
