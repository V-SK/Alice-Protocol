#!/usr/bin/env python3
"""Flask front-end for the VPS3 full-model download queue."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Tuple

from flask import Flask, jsonify, request

from queue_core import DownloadQueueManager


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("alice-download-queue")


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _client_ip() -> str:
    return (
        str(request.headers.get("X-Real-IP") or request.headers.get("X-Forwarded-For") or request.remote_addr or "")
        .split(",", 1)[0]
        .strip()
    )


def _admin_allowed() -> bool:
    return _client_ip() in {"127.0.0.1", "::1"}


def create_app() -> Flask:
    app = Flask(__name__)
    manager = DownloadQueueManager(
        max_concurrent=_env_int("DL_MAX_CONCURRENT", 8),
        download_timeout=_env_int("DL_TIMEOUT", 1200),
        slot_estimate_sec=_env_int("DL_SLOT_ESTIMATE", 960),
    )
    cleanup_interval = max(5, _env_int("DL_CLEANUP_INTERVAL", 60))

    def _periodic_cleanup() -> None:
        while True:
            time.sleep(cleanup_interval)
            try:
                manager.maintenance_tick()
            except Exception:
                log.exception("Periodic queue cleanup failed")

    threading.Thread(
        target=_periodic_cleanup,
        name="alice-download-queue-cleanup",
        daemon=True,
    ).start()

    @app.get("/health")
    def health() -> Tuple[str, int]:
        return "ok", 200

    @app.post("/models/queue")
    def join_queue():
        data = request.get_json(silent=True) or {}
        payload = manager.join(
            address=str(data.get("address", "")).strip() or _client_ip(),
            instance_id=str(data.get("instance_id", "")).strip(),
            client_ip=_client_ip(),
        )
        return jsonify(payload)

    @app.get("/models/queue")
    def queue_status():
        payload, status = manager.status(request.args.get("queue_id", ""))
        return jsonify(payload), status

    @app.post("/models/queue/heartbeat")
    def heartbeat():
        data = request.get_json(silent=True) or {}
        payload, status = manager.heartbeat(
            download_token=str(data.get("download_token", "")).strip(),
            client_ip=_client_ip(),
        )
        return jsonify(payload), status

    @app.post("/models/queue/complete")
    def complete():
        data = request.get_json(silent=True) or {}
        payload, status = manager.complete(
            queue_id=str(data.get("queue_id", "")).strip(),
            download_token=str(data.get("download_token", "")).strip(),
            client_ip=_client_ip(),
        )
        return jsonify(payload), status

    @app.get("/auth/verify")
    def verify():
        download_token = str(request.headers.get("X-Download-Token") or request.args.get("token") or "").strip()
        original_uri = str(request.headers.get("X-Original-URI") or request.args.get("uri") or "").strip()
        if manager.verify_token(download_token, _client_ip(), original_uri):
            return "", 200
        return "", 403

    @app.get("/models/queue/status")
    def admin_status():
        if not _admin_allowed():
            return jsonify({"error": "forbidden"}), 403
        snapshot: Dict[str, object] = manager.snapshot()
        return jsonify(snapshot), 200

    return app


app = create_app()
