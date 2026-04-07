#!/usr/bin/env python3
"""Core queue state for VPS3 full-model download throttling."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import secrets
import threading
import time
from typing import Callable, Deque, Dict, List, Optional, Tuple


@dataclass
class QueueEntry:
    queue_id: str
    address: str
    instance_id: str
    client_ip: str
    enqueued_at: float


@dataclass
class ActiveSlot:
    queue_id: str
    download_token: str
    address: str
    instance_id: str
    client_ip: str
    started_at: float
    expires_at: float
    enqueued_at: float


class DownloadQueueManager:
    """In-memory FIFO queue with lease-based active download slots."""

    def __init__(
        self,
        max_concurrent: int = 8,
        download_timeout: int = 1200,
        slot_estimate_sec: int = 960,
        audit_limit: int = 200,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.max_concurrent = max(1, int(max_concurrent))
        self.download_timeout = max(60, int(download_timeout))
        self.slot_estimate_sec = max(60, int(slot_estimate_sec))
        self._time_fn = time_fn or time.time
        self._lock = threading.Lock()
        self._queue: List[QueueEntry] = []
        self._queue_index: Dict[str, QueueEntry] = {}
        self._active_by_queue_id: Dict[str, ActiveSlot] = {}
        self._token_index: Dict[str, str] = {}
        self._audit: Deque[Dict[str, object]] = deque(maxlen=max(10, audit_limit))

    def _now(self) -> float:
        return float(self._time_fn())

    def _add_audit(self, event: str, **fields: object) -> None:
        payload: Dict[str, object] = {"event": event, "time": int(self._now())}
        payload.update(fields)
        self._audit.append(payload)

    def _find_existing_identity(self, address: str, instance_id: str, client_ip: str) -> Tuple[Optional[QueueEntry], Optional[ActiveSlot]]:
        for active in self._active_by_queue_id.values():
            if (
                active.address == address
                and active.instance_id == instance_id
                and active.client_ip == client_ip
            ):
                return None, active
        for entry in self._queue:
            if (
                entry.address == address
                and entry.instance_id == instance_id
                and entry.client_ip == client_ip
            ):
                return entry, None
        return None, None

    def _wait_seconds_for_position(self, position: int) -> int:
        return max(0, int(position) * self.slot_estimate_sec // self.max_concurrent)

    def _active_payload(self, active: ActiveSlot) -> Dict[str, object]:
        expires_in = max(0, int(active.expires_at - self._now()))
        return {
            "status": "active",
            "queue_id": active.queue_id,
            "position": 0,
            "download_token": active.download_token,
            "expires_in": expires_in,
            "message": "Download slot active",
        }

    def _queued_payload(self, entry: QueueEntry, position: int) -> Dict[str, object]:
        wait_seconds = self._wait_seconds_for_position(position)
        return {
            "status": "queued",
            "queue_id": entry.queue_id,
            "position": position,
            "wait_seconds": wait_seconds,
            "message": f"Queue position #{position}, ~{wait_seconds // 60} min",
        }

    def _pop_queue_by_id(self, queue_id: str) -> Optional[QueueEntry]:
        entry = self._queue_index.pop(queue_id, None)
        if entry is None:
            return None
        self._queue = [item for item in self._queue if item.queue_id != queue_id]
        return entry

    def _promote_unlocked(self) -> None:
        while self._queue and len(self._active_by_queue_id) < self.max_concurrent:
            entry = self._queue.pop(0)
            self._queue_index.pop(entry.queue_id, None)
            token = secrets.token_urlsafe(24)
            now = self._now()
            active = ActiveSlot(
                queue_id=entry.queue_id,
                download_token=token,
                address=entry.address,
                instance_id=entry.instance_id,
                client_ip=entry.client_ip,
                started_at=now,
                expires_at=now + self.download_timeout,
                enqueued_at=entry.enqueued_at,
            )
            self._active_by_queue_id[active.queue_id] = active
            self._token_index[token] = active.queue_id
            self._add_audit(
                "promoted",
                queue_id=active.queue_id,
                address=active.address,
                instance_id=active.instance_id,
                waited_seconds=int(now - entry.enqueued_at),
            )

    def _cleanup_unlocked(self) -> None:
        now = self._now()
        expired_ids = [
            queue_id
            for queue_id, active in self._active_by_queue_id.items()
            if now > active.expires_at
        ]
        for queue_id in expired_ids:
            active = self._active_by_queue_id.pop(queue_id)
            self._token_index.pop(active.download_token, None)
            self._add_audit(
                "slot_expired",
                queue_id=active.queue_id,
                address=active.address,
                instance_id=active.instance_id,
            )
        self._promote_unlocked()

    def maintenance_tick(self) -> None:
        """Expire stale download slots and promote queued miners."""
        with self._lock:
            self._cleanup_unlocked()

    def join(self, address: str, instance_id: str, client_ip: str) -> Dict[str, object]:
        wallet = str(address or client_ip).strip() or client_ip
        miner_instance = str(instance_id or wallet).strip() or wallet
        ip = str(client_ip or "unknown").strip() or "unknown"
        with self._lock:
            self._cleanup_unlocked()
            queued, active = self._find_existing_identity(wallet, miner_instance, ip)
            if active is not None:
                return self._active_payload(active)
            if queued is not None:
                position = next((idx + 1 for idx, item in enumerate(self._queue) if item.queue_id == queued.queue_id), -1)
                if position > 0:
                    return self._queued_payload(queued, position)

            if len(self._active_by_queue_id) < self.max_concurrent:
                now = self._now()
                queue_id = secrets.token_urlsafe(18)
                token = secrets.token_urlsafe(24)
                active = ActiveSlot(
                    queue_id=queue_id,
                    download_token=token,
                    address=wallet,
                    instance_id=miner_instance,
                    client_ip=ip,
                    started_at=now,
                    expires_at=now + self.download_timeout,
                    enqueued_at=now,
                )
                self._active_by_queue_id[queue_id] = active
                self._token_index[token] = queue_id
                self._add_audit(
                    "slot_acquired",
                    queue_id=queue_id,
                    address=wallet,
                    instance_id=miner_instance,
                )
                return self._active_payload(active)

            queue_id = secrets.token_urlsafe(18)
            entry = QueueEntry(
                queue_id=queue_id,
                address=wallet,
                instance_id=miner_instance,
                client_ip=ip,
                enqueued_at=self._now(),
            )
            self._queue.append(entry)
            self._queue_index[queue_id] = entry
            position = len(self._queue)
            self._add_audit(
                "enqueued",
                queue_id=queue_id,
                address=wallet,
                instance_id=miner_instance,
                position=position,
            )
            return self._queued_payload(entry, position)

    def status(self, queue_id: str) -> Tuple[Dict[str, object], int]:
        lookup = str(queue_id or "").strip()
        if not lookup:
            return {"status": "invalid", "error": "queue_id required"}, 400
        with self._lock:
            self._cleanup_unlocked()
            active = self._active_by_queue_id.get(lookup)
            if active is not None:
                return self._active_payload(active), 200
            entry = self._queue_index.get(lookup)
            if entry is not None:
                position = next((idx + 1 for idx, item in enumerate(self._queue) if item.queue_id == lookup), -1)
                if position > 0:
                    return self._queued_payload(entry, position), 200
            return {
                "status": "not_found",
                "queue_id": lookup,
                "position": -1,
                "message": "Queue ticket not found. Rejoin the queue.",
            }, 404

    def heartbeat(self, download_token: str, client_ip: str) -> Tuple[Dict[str, object], int]:
        token = str(download_token or "").strip()
        if not token:
            return {"status": "invalid", "error": "download_token required"}, 400
        with self._lock:
            self._cleanup_unlocked()
            queue_id = self._token_index.get(token)
            if queue_id is None:
                return {"status": "not_found"}, 404
            active = self._active_by_queue_id.get(queue_id)
            if active is None:
                return {"status": "not_found"}, 404
            if active.client_ip != str(client_ip or "").strip():
                return {"status": "forbidden", "error": "token bound to a different client IP"}, 403
            active.expires_at = self._now() + self.download_timeout
            self._add_audit("heartbeat", queue_id=queue_id, address=active.address, instance_id=active.instance_id)
            return {
                "status": "ok",
                "queue_id": active.queue_id,
                "expires_in": max(0, int(active.expires_at - self._now())),
            }, 200

    def complete(
        self,
        *,
        queue_id: Optional[str] = None,
        download_token: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> Tuple[Dict[str, object], int]:
        token = str(download_token or "").strip()
        lookup_queue_id = str(queue_id or "").strip()
        ip = str(client_ip or "").strip()
        with self._lock:
            self._cleanup_unlocked()
            if token:
                lookup_queue_id = self._token_index.get(token, lookup_queue_id)
            active = self._active_by_queue_id.get(lookup_queue_id) if lookup_queue_id else None
            if active is not None:
                if ip and active.client_ip != ip:
                    return {"status": "forbidden", "error": "token bound to a different client IP"}, 403
                self._active_by_queue_id.pop(active.queue_id, None)
                self._token_index.pop(active.download_token, None)
                self._add_audit("complete", queue_id=active.queue_id, address=active.address, instance_id=active.instance_id)
                self._promote_unlocked()
                return {"status": "ok", "queue_id": active.queue_id}, 200
            if lookup_queue_id:
                entry = self._queue_index.get(lookup_queue_id)
                if entry is not None:
                    if ip and entry.client_ip != ip:
                        return {"status": "forbidden", "error": "queue ticket bound to a different client IP"}, 403
                    self._pop_queue_by_id(lookup_queue_id)
                    self._add_audit("queue_cancelled", queue_id=entry.queue_id, address=entry.address, instance_id=entry.instance_id)
                    return {"status": "ok", "queue_id": entry.queue_id}, 200
            return {"status": "not_found"}, 404

    def verify_token(self, download_token: str, client_ip: str, request_path: str) -> bool:
        token = str(download_token or "").strip()
        path = str(request_path or "").split("?", 1)[0].strip()
        ip = str(client_ip or "").strip()
        if not token or not path.startswith("/models/v") or not path.endswith("_full.pt"):
            return False
        with self._lock:
            self._cleanup_unlocked()
            queue_id = self._token_index.get(token)
            if queue_id is None:
                return False
            active = self._active_by_queue_id.get(queue_id)
            if active is None:
                return False
            return active.client_ip == ip

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            self._cleanup_unlocked()
            now = self._now()
            active = [
                {
                    "queue_id": slot.queue_id,
                    "address": slot.address,
                    "instance_id": slot.instance_id,
                    "elapsed_seconds": int(now - slot.started_at),
                    "expires_in": max(0, int(slot.expires_at - now)),
                }
                for slot in self._active_by_queue_id.values()
            ]
            queued = [
                {
                    "queue_id": entry.queue_id,
                    "address": entry.address,
                    "instance_id": entry.instance_id,
                    "position": idx + 1,
                    "waiting_seconds": int(now - entry.enqueued_at),
                }
                for idx, entry in enumerate(self._queue)
            ]
            return {
                "active_downloads": len(active),
                "max_concurrent": self.max_concurrent,
                "queue_length": len(queued),
                "active": active,
                "queue": queued,
                "audit": list(self._audit),
            }
