#!/usr/bin/env python3
"""Regression tests for atomic runtime auth/session state."""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MINER_DIR = REPO_ROOT / "miner"
if str(MINER_DIR) not in sys.path:
    sys.path.insert(0, str(MINER_DIR))

import alice_miner
import plan_b


class AtomicTokenHolderTests(unittest.TestCase):
    def test_headers_include_authorization_and_miner_id(self) -> None:
        holder = alice_miner.AtomicTokenHolder(
            token="secret-token",
            miner_id="miner-123",
            instance_id="instance-123",
            data_plane_url="https://runtime.example",
        )

        self.assertEqual(
            holder.headers,
            {
                "Authorization": "Bearer secret-token",
                "X-Miner-Id": "miner-123",
            },
        )

    def test_update_is_atomic_for_token_and_miner_id(self) -> None:
        holder = alice_miner.AtomicTokenHolder(token="t0", miner_id="m0")
        mismatches: list[tuple[str, str]] = []
        stop_event = threading.Event()

        def writer() -> None:
            for idx in range(1, 400):
                holder.update(
                    token=f"token-{idx}",
                    miner_id=f"miner-{idx}",
                    instance_id=f"instance-{idx}",
                    data_plane_url=f"https://runtime-{idx}.example",
                )
            stop_event.set()

        def reader() -> None:
            while not stop_event.is_set():
                snapshot = holder.snapshot()
                token = str(snapshot.get("token") or "")
                miner_id = str(snapshot.get("miner_id") or "")
                if not token or not miner_id:
                    continue
                if "-" not in token or "-" not in miner_id:
                    continue
                token_idx = token.rsplit("-", 1)[-1]
                miner_idx = miner_id.rsplit("-", 1)[-1]
                if token_idx != miner_idx:
                    mismatches.append((token, miner_id))
                    stop_event.set()
                    return

        writer_thread = threading.Thread(target=writer)
        readers = [threading.Thread(target=reader) for _ in range(6)]

        for reader_thread in readers:
            reader_thread.start()
        writer_thread.start()

        writer_thread.join()
        for reader_thread in readers:
            reader_thread.join(timeout=1)

        self.assertEqual(mismatches, [])

    def test_concurrent_reads_and_writes_are_safe(self) -> None:
        holder = alice_miner.AtomicTokenHolder(token="seed", miner_id="miner-seed")
        errors: list[BaseException] = []
        stop_event = threading.Event()

        def writer() -> None:
            try:
                for idx in range(200):
                    holder.update(
                        token=f"token-{idx}",
                        miner_id=f"miner-{idx}",
                        instance_id=f"instance-{idx}",
                        data_plane_url=f"https://runtime-{idx}.example",
                    )
                    time.sleep(0.001)
            except BaseException as exc:  # pragma: no cover - test safety guard
                errors.append(exc)
            finally:
                stop_event.set()

        def reader() -> None:
            try:
                while not stop_event.is_set():
                    _ = holder.token
                    _ = holder.headers
                    snapshot = holder.snapshot()
                    self.assertIn("updated_at", snapshot)
            except BaseException as exc:  # pragma: no cover - test safety guard
                errors.append(exc)
                stop_event.set()

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=writer))

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(errors, [])

    def test_runtime_session_update_refreshes_plan_b_and_heartbeat_views_together(self) -> None:
        session = alice_miner.RuntimeSession(
            "https://runtime.example",
            "miner-old",
            {"device_type": "cpu"},
            "token-old",
            instance_id="instance-old",
        )
        trainer = plan_b.LocalTrainer(
            model=None,
            device=mock.Mock(),
            ps_url="https://control.example",
            aggregator_url="https://runtime.example",
            miner_address="wallet-address",
            auth=session.auth,
            args=mock.Mock(batch_size=0, precision="fp32"),
            miner_instance_id="instance-old",
        )

        alice_miner._update_runtime_auth_state(
            session,
            data_plane_url="https://runtime-new.example",
            miner_id="miner-new",
            instance_id="instance-new",
            capabilities={"device_type": "cpu"},
            auth_token="token-new",
        )

        with mock.patch.object(alice_miner, "send_heartbeat", return_value="ok") as send_heartbeat:
            self.assertEqual(alice_miner.send_runtime_heartbeat(session), "ok")

        self.assertEqual(trainer.auth.token, "token-new")
        self.assertEqual(trainer._headers()["Authorization"], "Bearer token-new")
        self.assertEqual(trainer._runtime_data_plane_url(), "https://runtime-new.example")
        send_heartbeat.assert_called_once_with(
            "https://runtime-new.example",
            "miner-new",
            {"device_type": "cpu"},
            auth_token="token-new",
        )


if __name__ == "__main__":
    unittest.main()
