"""Shared test fixtures.

`pytest` must never issue a real Razorpay API call. Three things conspire to
make it do so when a developer has a real .env on disk:

1. `load_environment()` calls `load_dotenv()`, which repopulates exactly the
   variables the credential tests delete with `monkeypatch.delenv`.
2. The module-level singleton `_gateway = RazorpayTestGateway()` runs
   `load_environment()` at *import* time, so os.environ is already populated
   before any fixture can patch the loader.
3. The real-uvicorn tests serve from a daemon thread that outlives the test
   body. `monkeypatch` is function-scoped, so its teardown restores
   LIVE_API_MODE=true while that thread is still running the pipeline.

The module-level block below is the load-bearing part: it runs at conftest
import, before the test modules import `src`, and `load_dotenv()` does not
override variables that already exist. That gives every thread in the process a
credential-free, live-API-off floor for the whole session. The autouse fixture
is a convenience on top for tests that assert on delenv/setenv behaviour.
"""

import os

os.environ["LIVE_API_MODE"] = "false"
os.environ["RAZORPAY_KEY_ID"] = ""
os.environ["RAZORPAY_KEY_SECRET"] = ""
os.environ["RAZORPAY_WEBHOOK_SECRET"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["HF_TOKEN"] = ""
os.environ["HUGGING_FACE_HUB_TOKEN"] = ""

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _hermetic_razorpay_env(monkeypatch):
    monkeypatch.setattr("src.razorpay_integration.load_environment", lambda: None)
    # Reset the LLM client cache each test so a stale cached client from a
    # previous test (or import-time init) never leaks a real API call.
    from src.llm import reset_client
    reset_client()
