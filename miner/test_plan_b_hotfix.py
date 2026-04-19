#!/usr/bin/env python3
"""Regression tests for Plan B model catch-up hotfixes."""

from __future__ import annotations

import sys
import tempfile
import threading
import unittest
import json
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MINER_DIR = REPO_ROOT / "miner"
if str(MINER_DIR) not in sys.path:
    sys.path.insert(0, str(MINER_DIR))

import plan_b


class PlanBHotfixTests(unittest.TestCase):
    def test_upload_parallelism_caps_by_layer_count(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)

        with mock.patch.object(plan_b, "UPLOAD_MAX_PARALLELISM", 4):
            self.assertEqual(trainer._upload_parallelism(0), 1)
            self.assertEqual(trainer._upload_parallelism(1), 1)
            self.assertEqual(trainer._upload_parallelism(3), 3)
            self.assertEqual(trainer._upload_parallelism(9), 4)

    def test_submit_param_diff_uses_parallel_upload_workers_and_finalize(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.auth = mock.Mock(miner_id="miner-runtime")
        trainer.miner_address = "a2miner"
        trainer.current_model_version = 261
        trainer._task_upload_headers = mock.Mock(return_value={"Authorization": "Bearer token"})
        trainer._runtime_data_plane_url = mock.Mock(return_value="http://agg")
        trainer._archive_spool = mock.Mock()
        trainer._cleanup_spool = mock.Mock()
        trainer._write_spool_manifest = mock.Mock()
        uploaded_layers = []
        uploaded_lock = threading.Lock()

        class FakeResponse:
            def __init__(self, status_code: int = 200) -> None:
                self.status_code = status_code
                self.text = ""

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

        class FakeSession:
            def post(self, url, data=None, headers=None, json=None, timeout=None):
                if url.endswith("/delta/upload_layer"):
                    with uploaded_lock:
                        uploaded_layers.append(str(headers["X-Layer-Name"]))
                    return FakeResponse(200)
                if url.endswith("/delta/finalize"):
                    return FakeResponse(200)
                raise AssertionError(url)

            def close(self) -> None:
                return None

        trainer._make_upload_session = mock.Mock(side_effect=lambda pool_size: FakeSession())

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for idx in range(5):
                path = Path(tmpdir) / f"layer_{idx}.pt"
                path.write_bytes(b"payload")
                paths.append(str(path))
            metadata = {
                "layer_files": paths,
                "completed_shards": 3,
                "batch_size": 8,
                "completed_effective_tokens": 1024,
                "param_diff_norm": 0.25,
                "model_version": 261,
            }
            with (
                mock.patch.object(plan_b, "UPLOAD_MAX_PARALLELISM", 3),
                mock.patch.object(plan_b.requests, "post", return_value=FakeResponse(200)),
            ):
                result = trainer.submit_param_diff(metadata)

        self.assertTrue(result)
        self.assertCountEqual(uploaded_layers, [f"layer_{i}" for i in range(5)])
        self.assertEqual(trainer._make_upload_session.call_count, 3)
        trainer._archive_spool.assert_not_called()
        trainer._cleanup_spool.assert_called_once()

    def test_submit_param_diff_archives_spool_on_terminal_upload_status(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.auth = mock.Mock(miner_id="miner-runtime")
        trainer.miner_address = "a2miner"
        trainer.current_model_version = 261
        trainer._task_upload_headers = mock.Mock(return_value={"Authorization": "Bearer token"})
        trainer._runtime_data_plane_url = mock.Mock(return_value="http://agg")
        trainer._archive_spool = mock.Mock()
        trainer._cleanup_spool = mock.Mock()
        trainer._write_spool_manifest = mock.Mock()

        class FakeResponse:
            def __init__(self, status_code: int) -> None:
                self.status_code = status_code
                self.text = ""

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

        class FakeSession:
            def post(self, url, data=None, headers=None, json=None, timeout=None):
                if url.endswith("/delta/upload_layer"):
                    return FakeResponse(410)
                raise AssertionError(url)

            def close(self) -> None:
                return None

        trainer._make_upload_session = mock.Mock(return_value=FakeSession())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "layer_0.pt"
            path.write_bytes(b"payload")
            metadata = {
                "layer_files": [str(path)],
                "completed_shards": 1,
                "batch_size": 4,
                "completed_effective_tokens": 128,
                "param_diff_norm": 0.1,
                "model_version": 261,
            }
            result = trainer.submit_param_diff(metadata)

        self.assertTrue(result)
        trainer._archive_spool.assert_called_once()
        trainer._cleanup_spool.assert_not_called()

    def test_spool_manifest_round_trips_task_context(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer._task_context = {
            "task_id": "task-123",
            "assignment_token": "lease-abc",
            "assignment_epoch": 284,
            "deadline": "2026-04-16T04:00:00Z",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            spool_dir = Path(tmpdir) / "v284"
            spool_dir.mkdir(parents=True, exist_ok=True)
            layer_path = spool_dir / "layer_0.pt"
            layer_path.write_bytes(b"payload")
            manifest = {
                "manifest_path": str(spool_dir / "manifest.json"),
                "layer_files": [str(layer_path)],
                "task_context": dict(trainer._task_context),
            }
            trainer._write_spool_manifest(manifest)
            loaded = trainer._read_spool_manifest(spool_dir / "manifest.json")

        assert loaded is not None
        self.assertEqual(loaded["task_context"]["task_id"], "task-123")
        self.assertEqual(loaded["task_context"]["assignment_token"], "lease-abc")
        self.assertEqual(loaded["task_context"]["assignment_epoch"], 284)

    def test_recover_pending_param_diff_restores_task_context(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer._task_context = None

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.param_diff_outbox_dir = Path(tmpdir)
            spool_dir = trainer.param_diff_outbox_dir / "v284"
            spool_dir.mkdir(parents=True, exist_ok=True)
            layer_path = spool_dir / "layer_0.pt"
            layer_path.write_bytes(b"payload")
            manifest_path = spool_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "state": "pending_upload",
                        "layer_files": [layer_path.name],
                        "task_context": {
                            "task_id": "task-restore",
                            "assignment_token": "lease-restore",
                            "assignment_epoch": 284,
                            "deadline": "2026-04-16T04:00:00Z",
                        },
                    }
                ),
                encoding="utf-8",
            )

            recovered = trainer.recover_pending_param_diff()

        assert recovered is not None
        self.assertEqual(trainer._task_context["task_id"], "task-restore")
        self.assertEqual(trainer._task_context["assignment_token"], "lease-restore")

    def test_preserve_high_precision_params_keeps_norm_scales_in_fp32(self) -> None:
        class ToyNorm(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.float16))

        class ToyInner(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = ToyNorm()

        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = ToyInner()
                self.linear = torch.nn.Linear(4, 4, bias=False, dtype=torch.float16)

        model = ToyModel()

        preserved = plan_b._preserve_high_precision_params(model)

        self.assertEqual(preserved, 1)
        self.assertEqual(model.model.norm.weight.dtype, torch.float32)
        self.assertEqual(model.linear.weight.dtype, torch.float16)

    def test_save_global_snapshot_keeps_norm_scales_in_fp32(self) -> None:
        class ToyNorm(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))

        class ToyInner(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = ToyNorm()

        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = ToyInner()
                self.linear = torch.nn.Linear(4, 4, bias=False, dtype=torch.float16)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
            trainer.model = ToyModel()
            trainer.snapshot_dir = Path(tmpdir)

            trainer.save_global_snapshot()

            norm_tensor = torch.load(
                trainer.snapshot_dir / "model_norm_weight.pt",
                map_location="cpu",
                weights_only=True,
            )
            linear_tensor = torch.load(
                trainer.snapshot_dir / "linear_weight.pt",
                map_location="cpu",
                weights_only=True,
            )

        self.assertEqual(norm_tensor.dtype, torch.float32)
        self.assertEqual(linear_tensor.dtype, torch.float16)

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

    def test_publication_state_uses_status_published_versions_when_model_info_omits_them(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.current_model_version = 284
        trainer._fetch_status = mock.Mock(
            return_value={
                "model_version": 285,
                "published_model_version": 284,
                "published_update_version_file": 284,
            }
        )
        trainer._fetch_model_info = mock.Mock(
            return_value={
                "version": 284,
                "base_urls": ["https://huggingface.example/models"],
            }
        )

        publication = trainer._publication_state(force=True)

        self.assertEqual(publication["target_version"], 285)
        self.assertEqual(publication["published_full_version"], 284)
        self.assertEqual(publication["published_update_version"], 284)
        self.assertEqual(
            publication["full_model_base_urls"],
            ["https://huggingface.example/models"],
        )
        self.assertEqual(
            publication["epoch_update_base_urls"],
            ["https://dl.aliceprotocol.org/epoch_updates"],
        )

    def test_apply_epoch_updates_waits_when_live_ahead_of_published_artifacts(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.model = object()
        trainer.current_model_version = 284
        trainer._publication_state = mock.Mock(
            return_value={
                "target_version": 285,
                "bootstrap_version": 284,
                "published_full_version": 284,
                "published_update_version": 284,
                "full_model_base_urls": ["https://dl.aliceprotocol.org/models"],
                "epoch_update_base_urls": ["https://dl.aliceprotocol.org/epoch_updates"],
            }
        )
        trainer.download_full_model = mock.Mock()
        trainer._download_epoch_update_from_mirrors = mock.Mock()
        trainer._download_epoch_update_from_ps = mock.Mock()

        trainer.apply_epoch_updates()

        trainer.download_full_model.assert_not_called()
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

    def test_wait_for_epoch_rollover_waits_for_epoch_change(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer._fetch_status = mock.Mock(
            side_effect=[
                {"local_epoch": 10, "epoch_remaining_s": 120},
                {"local_epoch": 10, "epoch_remaining_s": 45},
                {"local_epoch": 11, "epoch_remaining_s": 2980},
            ]
        )
        trainer._extract_epoch_number = plan_b.LocalTrainer._extract_epoch_number.__get__(
            trainer,
            plan_b.LocalTrainer,
        )
        trainer._extract_epoch_remaining_seconds = (
            plan_b.LocalTrainer._extract_epoch_remaining_seconds.__get__(
                trainer,
                plan_b.LocalTrainer,
            )
        )

        with (
            mock.patch.object(plan_b.time, "sleep") as sleep_mock,
            mock.patch.object(plan_b, "SUBMIT_WINDOW_S", 1),
            mock.patch.object(plan_b, "EPOCH_ROLLOVER_WAIT_BUFFER_S", 0),
        ):
            rolled_over = plan_b.wait_for_epoch_rollover(trainer, poll_interval_s=1)

        self.assertTrue(rolled_over)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_wait_for_epoch_rollover_times_out_without_epoch_change(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer._fetch_status = mock.Mock(
            side_effect=[
                {"local_epoch": 10, "epoch_remaining_s": 1},
                {"local_epoch": 10, "epoch_remaining_s": 0.5},
                {"local_epoch": 10, "epoch_remaining_s": 0.2},
                {"local_epoch": 10, "epoch_remaining_s": 0.1},
                {"local_epoch": 10, "epoch_remaining_s": 0.0},
                {"local_epoch": 10, "epoch_remaining_s": 0.0},
                {"local_epoch": 10, "epoch_remaining_s": 0.0},
            ]
        )
        trainer._extract_epoch_number = plan_b.LocalTrainer._extract_epoch_number.__get__(
            trainer,
            plan_b.LocalTrainer,
        )
        trainer._extract_epoch_remaining_seconds = (
            plan_b.LocalTrainer._extract_epoch_remaining_seconds.__get__(
                trainer,
                plan_b.LocalTrainer,
            )
        )

        with (
            mock.patch.object(plan_b.time, "sleep") as sleep_mock,
            mock.patch.object(plan_b, "SUBMIT_WINDOW_S", 1),
            mock.patch.object(plan_b, "EPOCH_ROLLOVER_WAIT_BUFFER_S", 0),
        ):
            rolled_over = plan_b.wait_for_epoch_rollover(trainer, poll_interval_s=1)

        self.assertFalse(rolled_over)
        self.assertGreaterEqual(sleep_mock.call_count, 1)

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

    def test_apply_epoch_update_payload_keeps_norm_scale_fp32_for_chunked_updates(self) -> None:
        class ToyNorm(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))

        class ToyInner(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = ToyNorm()

        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = ToyInner()

        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.model = ToyModel()
        trainer.current_model_version = 9
        trainer._write_local_version_marker = mock.Mock()

        trainer._apply_epoch_update_payload(
            {
                "old_version": 9,
                "new_version": 10,
                "chunks": [
                    {
                        "name": "model.norm.weight",
                        "indices": torch.tensor([1, 3], dtype=torch.int32),
                        "values": torch.tensor([0.25, -0.5], dtype=torch.float16),
                        "shape": (4,),
                        "indices_dtype": "int32",
                    }
                ],
            },
            from_version=9,
        )

        updated = trainer.model.model.norm.weight.detach()
        self.assertEqual(updated.dtype, torch.float32)
        self.assertAlmostEqual(float(updated[0]), 1.0, places=6)
        self.assertAlmostEqual(float(updated[1]), 1.25, places=6)
        self.assertAlmostEqual(float(updated[2]), 1.0, places=6)
        self.assertAlmostEqual(float(updated[3]), 0.5, places=6)
        trainer._write_local_version_marker.assert_called_once_with(10)

    def test_apply_epoch_update_payload_keeps_norm_scale_fp32_for_dense_updates(self) -> None:
        class ToyNorm(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))

        class ToyInner(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = ToyNorm()

        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = ToyInner()

        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.model = ToyModel()
        trainer.current_model_version = 10
        trainer._write_local_version_marker = mock.Mock()

        trainer._apply_epoch_update_payload(
            {
                "old_version": 10,
                "new_version": 11,
                "model.norm.weight": torch.tensor([0.0, 0.125, 0.0, -0.25], dtype=torch.float16),
            },
            from_version=10,
        )

        updated = trainer.model.model.norm.weight.detach()
        self.assertEqual(updated.dtype, torch.float32)
        self.assertAlmostEqual(float(updated[0]), 1.0, places=6)
        self.assertAlmostEqual(float(updated[1]), 1.125, places=6)
        self.assertAlmostEqual(float(updated[2]), 1.0, places=6)
        self.assertAlmostEqual(float(updated[3]), 0.75, places=6)
        trainer._write_local_version_marker.assert_called_once_with(11)

    def test_chunk_indices_for_apply_rejects_non_integral_dtype(self) -> None:
        with self.assertRaises(TypeError):
            plan_b._chunk_indices_for_apply({"indices": torch.tensor([1.0, 2.0], dtype=torch.float32)})


if __name__ == "__main__":
    unittest.main()
