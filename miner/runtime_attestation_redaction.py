from __future__ import annotations

from typing import Any


SECRET_MARKERS = (
    "token",
    "api_key",
    "apikey",
    "private_key",
    "secret",
    "mnemonic",
    "seed_phrase",
    "authorization",
    "bearer",
    "password",
)


def redact_runtime_attestation(payload: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        lowered = key.lower()
        if any(marker in lowered for marker in SECRET_MARKERS):
            redacted[key] = "[REDACTED]"
        elif isinstance(value, dict):
            redacted[key] = redact_runtime_attestation(value)
        elif isinstance(value, list):
            redacted[key] = [
                redact_runtime_attestation(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            redacted[key] = value
    return redacted
