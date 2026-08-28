"""Shared LLM client for the investigation pipeline.

Supported backends (checked in order):
1. **OpenAI** — when ``OPENAI_API_KEY`` is set.  Default model: ``gpt-4.1-mini``.
2. **Hugging Face Inference API** — when ``HF_TOKEN`` (or
   ``HUGGING_FACE_HUB_TOKEN``) is set.  Uses the OpenAI-compatible
   ``https://router.huggingface.co/v1/`` endpoint.  Default model:
   ``Qwen/Qwen2.5-72B-Instruct``.

When neither key is present, all callers fall back to the existing rule-based
logic.  Outputs are labeled ``LLM_REASONED`` vs ``RULE_BASED_FALLBACK`` so
the audit trail never confuses the two.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# Defaults per backend.
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"
HF_BASE_URL = "https://router.huggingface.co/v1/"

LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 2

# Module-level cached client (lazy init).
_client: Any = None
_client_checked = False
_active_backend: str | None = None  # "openai" or "huggingface"

# Cumulative usage tracking for the process lifetime.
_usage_stats = {
    "total_calls": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_latency_seconds": 0.0,
}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _load_api_key() -> tuple[str | None, str | None]:
    """Return (key, backend) — checks OpenAI first, then HF."""
    _load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY") or None
    if openai_key:
        return openai_key, "openai"
    hf_key = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None
    if hf_key:
        return hf_key, "huggingface"
    return None, None


def get_client() -> Any:
    """Return a cached OpenAI-compatible client, or None if no key is configured."""
    global _client, _client_checked, _active_backend
    if _client_checked:
        return _client
    _client_checked = True
    key, backend = _load_api_key()
    if not key:
        return None
    try:
        from openai import OpenAI

        if backend == "huggingface":
            _client = OpenAI(
                api_key=key,
                base_url=HF_BASE_URL,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            _active_backend = "huggingface"
            logger.info("LLM backend: Hugging Face Inference API")
        else:
            _client = OpenAI(api_key=key, timeout=LLM_TIMEOUT_SECONDS)
            _active_backend = "openai"
            logger.info("LLM backend: OpenAI")
        return _client
    except ImportError:
        logger.warning("openai package not installed; LLM path disabled")
        return None
    except Exception as exc:
        logger.warning("LLM client init failed: %s", exc)
        return None


def get_model() -> str:
    explicit = os.getenv("OPENAI_MODEL")
    if explicit:
        return explicit
    _, backend = _load_api_key()
    if backend == "huggingface":
        return DEFAULT_HF_MODEL
    return DEFAULT_OPENAI_MODEL


def get_backend() -> str | None:
    """Return the active backend name after client init, or None."""
    if not _client_checked:
        get_client()
    return _active_backend


def llm_available() -> bool:
    """Quick check — does not validate the key, just checks one is set."""
    key, _ = _load_api_key()
    return bool(key)


def reset_client() -> None:
    """Reset the cached client (for testing)."""
    global _client, _client_checked, _active_backend
    _client = None
    _client_checked = False
    _active_backend = None


def llm_call(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.2,
) -> dict[str, Any] | None:
    """Structured-JSON LLM call with retry, timeout, and usage logging.

    Returns the parsed JSON dict on success (with ``_llm_meta`` attached),
    or ``None`` on failure after all retries are exhausted.
    """
    client = get_client()
    if client is None:
        return None

    model = get_model()
    last_error: Exception | None = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        t0 = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            elapsed = time.monotonic() - t0

            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            _usage_stats["total_calls"] += 1
            _usage_stats["total_prompt_tokens"] += prompt_tokens
            _usage_stats["total_completion_tokens"] += completion_tokens
            _usage_stats["total_latency_seconds"] += elapsed

            logger.info(
                "LLM call: model=%s latency=%.2fs tokens=%d/%d attempt=%d",
                model,
                elapsed,
                prompt_tokens,
                completion_tokens,
                attempt,
            )

            content = response.choices[0].message.content
            result: dict[str, Any] = json.loads(content)
            result["_llm_meta"] = {
                "model": model,
                "backend": _active_backend or "unknown",
                "latency_seconds": round(elapsed, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "attempt": attempt,
            }
            return result

        except json.JSONDecodeError as exc:
            elapsed = time.monotonic() - t0
            logger.warning(
                "LLM returned non-JSON (attempt %d, %.2fs): %s",
                attempt,
                elapsed,
                exc,
            )
            last_error = exc
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.warning(
                "LLM call failed (attempt %d, %.2fs): %s",
                attempt,
                elapsed,
                exc,
            )
            last_error = exc

    logger.error(
        "LLM call failed after %d attempts: %s", LLM_MAX_RETRIES, last_error
    )
    return None


def get_usage_stats() -> dict[str, Any]:
    """Return cumulative usage stats with estimated cost."""
    model = get_model()
    stats = dict(_usage_stats)
    stats["model"] = model
    stats["backend"] = _active_backend or "none"

    # Per-million-token pricing (approximate, Aug 2026).
    # HF Inference API free-tier models: $0.
    pricing = {
        "4.1-mini": (0.40, 1.60),
        "4.1-nano": (0.10, 0.40),
        "4o-mini": (0.15, 0.60),
    }
    if _active_backend == "huggingface":
        input_rate, output_rate = 0.0, 0.0  # free tier
    else:
        input_rate, output_rate = next(
            (v for k, v in pricing.items() if k in model),
            (2.50, 10.0),  # conservative fallback for unknown models
        )
    input_cost = stats["total_prompt_tokens"] * input_rate / 1_000_000
    output_cost = stats["total_completion_tokens"] * output_rate / 1_000_000
    stats["estimated_cost_usd"] = round(input_cost + output_cost, 6)
    return stats


def reset_usage_stats() -> None:
    _usage_stats["total_calls"] = 0
    _usage_stats["total_prompt_tokens"] = 0
    _usage_stats["total_completion_tokens"] = 0
    _usage_stats["total_latency_seconds"] = 0.0
