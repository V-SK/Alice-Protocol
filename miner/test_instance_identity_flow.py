#!/usr/bin/env python3
"""Regression tests for miner client/runtime instance identity separation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MINER_DIR = REPO_ROOT / "miner"
if str(MINER_DIR) not in sys.path:
    sys.path.insert(0, str(MINER_DIR))

import alice_miner


class MinerInstanceIdentityFlowTests(unittest.TestCase):
    def test_client_instance_id_prefers_configured_label(self) -> None:
        self.assertEqual(
            alice_miner.resolve_client_instance_id("pc01-gpu0", "a2wallet"),
            "pc01-gpu0",
        )

    def test_client_instance_id_falls_back_to_wallet_address(self) -> None:
        self.assertEqual(
            alice_miner.resolve_client_instance_id(None, "a2wallet"),
            "a2wallet",
        )

    def test_runtime_identity_keeps_server_runtime_id_separate(self) -> None:
        runtime_miner_id, runtime_instance_id = alice_miner.extract_runtime_identity(
            {
                "miner_id": "a2wallet_gpu0_deadbeef",
                "instance_id": "a2wallet_gpu0_deadbeef",
            },
            client_instance_id="gpu0",
            wallet_address="a2wallet",
        )
        self.assertEqual(runtime_miner_id, "a2wallet_gpu0_deadbeef")
        self.assertEqual(runtime_instance_id, "a2wallet_gpu0_deadbeef")
        self.assertEqual(
            alice_miner.resolve_client_instance_id("gpu0", "a2wallet"),
            "gpu0",
        )


if __name__ == "__main__":
    unittest.main()
