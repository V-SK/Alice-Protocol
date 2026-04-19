#!/usr/bin/env python3
"""Regression tests for runtime session recovery and non-auth retry behavior."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MINER_DIR = REPO_ROOT / "miner"
if str(MINER_DIR) not in sys.path:
    sys.path.insert(0, str(MINER_DIR))

import alice_miner


class RuntimeRecoveryTests(unittest.TestCase):
    def test_runtime_session_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "runtime_session.json"
            session = alice_miner.RuntimeSession(
                "https://runtime.example",
                "miner-1",
                {"device_type": "cpu", "memory_gb": 16.0},
                "token-1",
                instance_id="instance-1",
                control_plane_url="https://control.example",
            )

            alice_miner._save_runtime_session_cache(session, path=cache_path)
            loaded = alice_miner._load_runtime_session_cache(
                "https://control.example",
                path=cache_path,
            )

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded["auth_token"], "token-1")
            self.assertEqual(loaded["miner_id"], "miner-1")
            self.assertEqual(loaded["instance_id"], "instance-1")
            self.assertEqual(loaded["control_plane_url"], "https://control.example")

    def test_send_runtime_heartbeat_recovers_session_before_reregister(self) -> None:
        session = alice_miner.RuntimeSession(
            "https://runtime.example",
            "miner-1",
            {"device_type": "cpu", "memory_gb": 16.0},
            "token-1",
            instance_id="instance-1",
            control_plane_url="https://control.example",
        )
        with mock.patch.object(alice_miner, "send_heartbeat", return_value="auth_error") as send_mock:
            with mock.patch.object(alice_miner, "recover_runtime_session", return_value=True) as recover_mock:
                status = alice_miner.send_runtime_heartbeat(session)

        self.assertEqual(status, "ok")
        send_mock.assert_called_once()
        recover_mock.assert_called_once_with(session)

    def test_request_task_with_retry_keeps_session_on_repeated_failures(self) -> None:
        session = alice_miner.RuntimeSession(
            "https://runtime.example",
            "miner-1",
            {"device_type": "cpu", "memory_gb": 16.0},
            "token-1",
            instance_id="instance-1",
            control_plane_url="https://control.example",
        )
        with mock.patch.object(alice_miner, "request_task_detailed", return_value=(None, "failed")):
            with mock.patch.object(alice_miner.time, "sleep"):
                task, status = alice_miner.request_task_with_retry(
                    "https://control.example",
                    "miner-1",
                    {"device_type": "cpu", "memory_gb": 16.0},
                    auth_token="token-1",
                    runtime_session=session,
                    retry_delay=1,
                    max_attempts=3,
                )

        self.assertIsNone(task)
        self.assertEqual(status, "failed")


if __name__ == "__main__":
    unittest.main()
