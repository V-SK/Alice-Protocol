#!/usr/bin/env python3
"""Regression tests for Plan B model catch-up hotfixes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MINER_DIR = REPO_ROOT / "miner"
if str(MINER_DIR) not in sys.path:
    sys.path.insert(0, str(MINER_DIR))

import plan_b


class PlanBHotfixTests(unittest.TestCase):
    def test_apply_epoch_updates_redownloads_full_model_when_local_is_ahead_of_published_updates(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.model = object()
        trainer.current_model_version = 140
        trainer._publication_state = mock.Mock(
            return_value={
                "target_version": 149,
                "bootstrap_version": 149,
                "published_update_version": 136,
                "full_model_base_urls": ["https://dl.aliceprotocol.org/models"],
                "epoch_update_base_urls": ["https://dl.aliceprotocol.org/epoch_updates"],
            }
        )
        trainer._select_full_download_version = mock.Mock(return_value=149)
        trainer.download_full_model = mock.Mock()
        trainer._download_epoch_update_from_mirrors = mock.Mock()
        trainer._download_epoch_update_from_ps = mock.Mock()

        trainer.apply_epoch_updates()

        trainer._select_full_download_version.assert_called_once_with(149, 149)
        trainer.download_full_model.assert_called_once_with(
            149,
            mirror_urls=["https://dl.aliceprotocol.org/models"],
        )
        trainer._download_epoch_update_from_mirrors.assert_not_called()
        trainer._download_epoch_update_from_ps.assert_not_called()

    def test_wait_for_next_epoch_skips_wait_when_local_matches_live_full_model(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.current_model_version = 150
        trainer._publication_state = mock.Mock(
            return_value={
                "target_version": 150,
                "published_full_version": 150,
                "published_update_version": 140,
            }
        )

        with mock.patch.object(plan_b.time, "sleep") as sleep_mock:
            plan_b.wait_for_next_epoch(trainer, poll_interval_s=1)

        sleep_mock.assert_not_called()

    def test_wait_for_next_epoch_skips_wait_when_no_newer_published_artifact_exists(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.current_model_version = 150
        trainer._publication_state = mock.Mock(
            return_value={
                "target_version": 149,
                "published_full_version": 149,
                "published_update_version": 140,
            }
        )

        with mock.patch.object(plan_b.time, "sleep") as sleep_mock:
            plan_b.wait_for_next_epoch(trainer, poll_interval_s=1)

        sleep_mock.assert_not_called()

    def test_wait_for_next_epoch_still_waits_for_newer_published_version(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.current_model_version = 149
        trainer._publication_state = mock.Mock(
            return_value={
                "target_version": 150,
                "published_full_version": 150,
                "published_update_version": 140,
            }
        )

        with mock.patch.object(plan_b.time, "sleep") as sleep_mock:
            plan_b.wait_for_next_epoch(trainer, poll_interval_s=1)

        sleep_mock.assert_called_once_with(1)

    def test_apply_epoch_update_payload_accepts_int32_indices(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.model = torch.nn.Linear(5, 1, bias=False)
        trainer.model.weight.data.zero_()
        trainer.current_model_version = 7
        trainer._write_local_version_marker = mock.Mock()

        trainer._apply_epoch_update_payload(
            {
                "old_version": 7,
                "new_version": 8,
                "chunks": [
                    {
                        "name": "weight",
                        "indices": torch.tensor([1, 3], dtype=torch.int32),
                        "values": torch.tensor([0.5, -1.25], dtype=torch.float16),
                        "shape": (1, 5),
                        "indices_dtype": "int32",
                    }
                ],
            },
            from_version=7,
        )

        self.assertEqual(trainer.current_model_version, 8)
        self.assertAlmostEqual(float(trainer.model.weight.data.view(-1)[1]), 0.5, places=5)
        self.assertAlmostEqual(float(trainer.model.weight.data.view(-1)[3]), -1.25, places=5)
        trainer._write_local_version_marker.assert_called_once_with(8)

    def test_apply_epoch_update_payload_accepts_int64_indices(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.model = torch.nn.Linear(5, 1, bias=False)
        trainer.model.weight.data.zero_()
        trainer.current_model_version = 8
        trainer._write_local_version_marker = mock.Mock()

        trainer._apply_epoch_update_payload(
            {
                "old_version": 8,
                "new_version": 9,
                "chunks": [
                    {
                        "name": "weight",
                        "indices": torch.tensor([0, 4], dtype=torch.int64),
                        "values": torch.tensor([1.0, -0.75], dtype=torch.float16),
                        "shape": (1, 5),
                    }
                ],
            },
            from_version=8,
        )

        self.assertEqual(trainer.current_model_version, 9)
        self.assertAlmostEqual(float(trainer.model.weight.data.view(-1)[0]), 1.0, places=5)
        self.assertAlmostEqual(float(trainer.model.weight.data.view(-1)[4]), -0.75, places=5)
        trainer._write_local_version_marker.assert_called_once_with(9)

    def test_chunk_indices_for_apply_rejects_non_integral_dtype(self) -> None:
        with self.assertRaises(TypeError):
            plan_b._chunk_indices_for_apply({"indices": torch.tensor([1.0, 2.0], dtype=torch.float32)})


if __name__ == "__main__":
    unittest.main()
