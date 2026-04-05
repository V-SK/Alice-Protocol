#!/usr/bin/env python3
"""Plan B miner runtime with epoch-level local SGD and sparse delta upload."""

import contextlib
import gc
import io
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch

import alice_miner as miner_lib


SNAPSHOT_DIR = Path.home() / ".alice" / "global_snapshot"
DELTA_OUTBOX_DIR = Path.home() / ".alice" / "delta_outbox"
PLAN_B_MODEL_DIR = Path.home() / ".alice" / "plan_b_models"
STATUS_CACHE_TTL_S = 30
TRAINING_WINDOW_S = 3000
SUBMIT_WINDOW_S = 600
MAX_INCREMENTAL_CATCHUP_GAP = 5
BATCH_RESTORE_SUCCESS_SHARDS = 3
MAX_OOM_RETRIES_AT_BATCH1 = 3
MODEL_INFO_CACHE_TTL_S = 15


def _plan_b_log(message: str) -> None:
    print(f"[PLAN-B] {message}")


def _safe_layer_name(name: str) -> str:
    return name.replace(".", "_")


def _normalize_url(url: str) -> str:
    return str(url or "").strip().rstrip("/")


def _coerce_version(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _pick_first_version(*values: Any) -> Optional[int]:
    for value in values:
        parsed = _coerce_version(value)
        if parsed is not None:
            return parsed
    return None


def _parse_url_candidates(raw_value: Any) -> List[str]:
    urls: List[str] = []
    if isinstance(raw_value, list):
        for item in raw_value:
            url = _normalize_url(item)
            if url and url not in urls:
                urls.append(url)
    elif isinstance(raw_value, str):
        for part in raw_value.split(","):
            url = _normalize_url(part)
            if url and url not in urls:
                urls.append(url)
    return urls


MODEL_MIRRORS = [
    "https://huggingface.co/v102ss/alice-7b-model/resolve/main",
    "https://dl.aliceprotocol.org/models",
]
EPOCH_UPDATE_MIRRORS = [
    "https://dl.aliceprotocol.org/epoch_updates",
]


def _extract_tokens(shard_data: Any) -> torch.Tensor:
    if isinstance(shard_data, dict):
        token_tensor = shard_data.get("tokens")
        if token_tensor is None:
            token_tensor = shard_data.get("input_ids")
    elif torch.is_tensor(shard_data):
        token_tensor = shard_data
    else:
        token_tensor = None
    if token_tensor is None:
        raise ValueError(f"Unsupported shard format: {type(shard_data)}")
    return token_tensor.view(-1).long()


def _load_update_payload(raw_bytes: bytes) -> Dict[str, Any]:
    return torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)


def _version_marker_path() -> Path:
    return PLAN_B_MODEL_DIR / "current_version"


class LocalTrainer:
    def __init__(
        self,
        model: Optional[torch.nn.Module],
        device: torch.device,
        ps_url: str,
        aggregator_url: str,
        miner_address: str,
        token: str,
        args: Any,
        miner_instance_id: Optional[str] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.ps_url = _normalize_url(ps_url)
        self.aggregator_url = _normalize_url(aggregator_url)
        self.miner_address = str(miner_address)
        self.miner_instance_id = str(miner_instance_id or miner_address)
        self.token = str(token or "")
        self.args = args
        self.snapshot_dir = SNAPSHOT_DIR
        self.delta_outbox_dir = DELTA_OUTBOX_DIR
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.delta_outbox_dir.mkdir(parents=True, exist_ok=True)
        PLAN_B_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.current_model_version: Optional[int] = None
        self.assigned_batch_size: Optional[int] = None
        self.effective_batch_size: int = 2
        self.stable_shards_at_current_batch: int = 0
        self._last_batch_log: Optional[Tuple[int, int, str]] = None
        self._status_cache: Optional[Dict[str, Any]] = None
        self._status_cache_ts: float = 0.0
        self._model_info_cache: Optional[Dict[str, Any]] = None
        self._model_info_cache_ts: float = 0.0
        self.epoch_start_time: float = time.time()

    def mark_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

    def _write_local_version_marker(self, version: int) -> None:
        PLAN_B_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _version_marker_path().write_text(str(int(version)))

    def _read_local_version_marker(self) -> Optional[int]:
        path = _version_marker_path()
        if not path.exists():
            return None
        try:
            value = path.read_text().strip()
            return int(value) if value else None
        except Exception:
            return None

    def _spool_dir(self, model_version: Optional[int] = None) -> Path:
        version = int(model_version if model_version is not None else self.current_model_version or 0)
        return self.delta_outbox_dir / f"v{version}"

    def _spool_manifest_path(self, spool_dir: Optional[Path] = None) -> Path:
        return (spool_dir or self._spool_dir()) / "manifest.json"

    def _write_spool_manifest(self, manifest: Dict[str, Any]) -> None:
        serializable = dict(manifest)
        serializable["layer_files"] = [
            Path(str(path)).name
            for path in (manifest.get("layer_files") or [])
        ]
        manifest_path = Path(str(serializable["manifest_path"]))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        encoded = json.dumps(serializable, indent=2, sort_keys=True)
        tmp_path.write_text(encoded, encoding="utf-8")
        os.replace(tmp_path, manifest_path)

    def _read_spool_manifest(self, manifest_path: Path) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        spool_dir = manifest_path.parent
        layer_files = data.get("layer_files") or []
        if not isinstance(layer_files, list):
            return None
        materialized_files: List[str] = []
        for file_name in layer_files:
            path = spool_dir / str(file_name)
            if not path.exists():
                return None
            materialized_files.append(str(path))
        data["spool_dir"] = str(spool_dir)
        data["manifest_path"] = str(manifest_path)
        data["layer_files"] = materialized_files
        return data

    def recover_pending_delta(self) -> Optional[Dict[str, Any]]:
        manifests = sorted(
            self.delta_outbox_dir.glob("v*/manifest.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for manifest_path in manifests:
            manifest = self._read_spool_manifest(manifest_path)
            if manifest is None:
                continue
            if str(manifest.get("state") or "pending_upload") == "uploaded":
                continue
            return manifest
        return None

    def _cleanup_spool(self, metadata: Dict[str, Any]) -> None:
        spool_dir = Path(str(metadata.get("spool_dir") or ""))
        if not spool_dir.exists():
            return
        for path in sorted(spool_dir.glob("*"), reverse=True):
            with contextlib.suppress(Exception):
                path.unlink()
        with contextlib.suppress(Exception):
            spool_dir.rmdir()

    def _headers(self) -> Dict[str, str]:
        return miner_lib._auth_headers(self.token)

    def _target_batch_size(self) -> int:
        ps_cap = max(1, int(self.assigned_batch_size or 0) or 2)
        user_override = int(getattr(self.args, "batch_size", 0) or 0)
        if user_override > 0:
            return max(1, min(user_override, ps_cap))
        return ps_cap

    def _log_batch_decision(self, source: str) -> None:
        assigned = max(1, int(self.assigned_batch_size or 0) or 2)
        effective = max(1, int(self.effective_batch_size or 0) or 1)
        state = (assigned, effective, str(source))
        if state == self._last_batch_log:
            return
        self._last_batch_log = state
        _plan_b_log(
            f"Batch size selected: assigned_batch_size={assigned}, "
            f"effective_batch_size={effective}, source={source}"
        )

    def update_task_batch_size(self, task: Dict[str, Any]) -> int:
        previous_assigned = self.assigned_batch_size
        raw_assigned = int((task or {}).get("assigned_batch_size", 0) or 0)
        self.assigned_batch_size = raw_assigned if raw_assigned > 0 else 2
        target_batch = self._target_batch_size()
        user_override = int(getattr(self.args, "batch_size", 0) or 0)
        source = "ps_assignment"
        if user_override > 0:
            source = "manual_override_capped"
        if previous_assigned is None or previous_assigned != self.assigned_batch_size:
            self.effective_batch_size = target_batch
            self.stable_shards_at_current_batch = 0
        elif self.effective_batch_size <= 0 or self.effective_batch_size > target_batch:
            self.effective_batch_size = target_batch
            self.stable_shards_at_current_batch = 0
        self._log_batch_decision(source)
        return target_batch

    def _clear_device_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            with contextlib.suppress(Exception):
                torch.mps.empty_cache()

    def _is_oom_error(self, exc: BaseException) -> bool:
        message = str(exc).lower()
        return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message

    def save_global_snapshot(self) -> None:
        if self.model is None:
            raise RuntimeError("Plan B model is not initialized")
        files_written = 0
        for name, param in self.model.named_parameters():
            path = self.snapshot_dir / f"{_safe_layer_name(name)}.pt"
            tensor = param.detach().clone().cpu().half()
            torch.save(tensor, path)
            files_written += 1
        _plan_b_log(f"Saved global snapshot: {files_written} files -> {self.snapshot_dir}")

    def load_global_param(self, name: str) -> torch.Tensor:
        path = self.snapshot_dir / f"{_safe_layer_name(name)}.pt"
        return torch.load(path, map_location="cpu", weights_only=True).float()

    def train_shard_local(self, shard_data: Any) -> Tuple[Optional[float], int, bool]:
        if self.model is None:
            raise RuntimeError("Plan B model is not initialized")
        tokens = _extract_tokens(shard_data)
        seq_len = int(getattr(self.args, "seq_len", 128) or 128)
        max_batches = int(getattr(self.args, "max_batches", 10) or 10)
        current_batch_size = max(1, int(self.effective_batch_size or self._target_batch_size() or 2))
        oom_retries_at_batch1 = 0

        while True:
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            local_lr = float(self.args.local_lr)
            hooks: List[Any] = []
            start_idx = 0
            total_loss = 0.0
            num_batches = 0
            should_retry = False

            for param in self.model.parameters():
                if not param.requires_grad:
                    continue
                if not hasattr(param, "register_post_accumulate_grad_hook"):
                    raise RuntimeError(
                        "register_post_accumulate_grad_hook is unavailable in this PyTorch build"
                    )

                def _hook(
                    _: torch.Tensor,
                    *,
                    _param: torch.Tensor = param,
                    _lr: float = local_lr,
                ) -> None:
                    grad = _param.grad
                    if grad is None:
                        return
                    _param.data = (_param.data.float() - _lr * grad.float()).to(_param.dtype)
                    _param.grad = None

                hooks.append(param.register_post_accumulate_grad_hook(_hook))

            try:
                while start_idx < max(1, tokens.numel() - seq_len - 1) and num_batches < max_batches:
                    batch_inputs: List[torch.Tensor] = []
                    batch_labels: List[torch.Tensor] = []
                    for batch_offset in range(current_batch_size):
                        offset = start_idx + batch_offset * seq_len
                        if offset + seq_len + 1 > tokens.numel():
                            break
                        chunk = tokens[offset : offset + seq_len + 1]
                        batch_inputs.append(chunk[:-1])
                        batch_labels.append(chunk[1:])
                    if not batch_inputs:
                        break

                    try:
                        input_ids = torch.stack(batch_inputs).to(self.device)
                        labels = torch.stack(batch_labels).to(self.device)
                        use_amp = self.device.type in ("cuda", "mps") and str(getattr(self.args, "precision", "auto")) != "fp32"
                        autocast_dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
                        with (torch.autocast(device_type=self.device.type, dtype=autocast_dtype) if use_amp else contextlib.nullcontext()):
                            _, loss = self.model(input_ids, labels)

                        if loss is None or not torch.isfinite(loss):
                            _plan_b_log("Skipping invalid local loss batch")
                            self.model.zero_grad(set_to_none=True)
                            start_idx += len(batch_inputs) * seq_len
                            del input_ids, labels, loss
                            continue

                        loss.backward()
                        total_loss += float(loss.item())
                        num_batches += 1
                        start_idx += len(batch_inputs) * seq_len
                        del input_ids, labels, loss
                    except Exception as exc:
                        if not self._is_oom_error(exc):
                            raise
                        self._clear_device_cache()
                        old_batch_size = current_batch_size
                        new_batch_size = max(1, current_batch_size // 2)
                        self.stable_shards_at_current_batch = 0
                        if new_batch_size == current_batch_size:
                            oom_retries_at_batch1 += 1
                            _plan_b_log(
                                f"OOM at batch_size={current_batch_size}; retry {oom_retries_at_batch1}/{MAX_OOM_RETRIES_AT_BATCH1}"
                            )
                            if oom_retries_at_batch1 >= MAX_OOM_RETRIES_AT_BATCH1:
                                _plan_b_log("Repeated OOM at batch_size=1, skipping shard")
                                self.effective_batch_size = 1
                                return None, 1, False
                        else:
                            current_batch_size = new_batch_size
                            oom_retries_at_batch1 = 0
                            self.effective_batch_size = new_batch_size
                            _plan_b_log(f"OOM detected, reducing batch size: {old_batch_size} -> {new_batch_size}")
                        should_retry = True
                        break
            finally:
                for hook in hooks:
                    hook.remove()
                self.model.zero_grad(set_to_none=True)

            if should_retry:
                continue

            if num_batches <= 0:
                _plan_b_log("No valid local batches completed, skipping shard")
                self.stable_shards_at_current_batch = 0
                self.effective_batch_size = current_batch_size
                return None, current_batch_size, False

            self.effective_batch_size = current_batch_size
            self.stable_shards_at_current_batch += 1
            target_batch_size = self._target_batch_size()
            if self.stable_shards_at_current_batch >= BATCH_RESTORE_SUCCESS_SHARDS and current_batch_size < target_batch_size:
                restored_batch_size = min(current_batch_size * 2, target_batch_size)
                if restored_batch_size != current_batch_size:
                    _plan_b_log(
                        f"Batch size restored after stable training: {current_batch_size} -> {restored_batch_size}"
                    )
                    self.effective_batch_size = restored_batch_size
                    self.stable_shards_at_current_batch = 0

            avg_loss = total_loss / num_batches
            _plan_b_log(
                f"Local shard training complete: batches={num_batches}, avg_loss={avg_loss:.4f}, batch_size={current_batch_size}"
            )
            return avg_loss, current_batch_size, True

    def compute_and_compress_delta(self) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Plan B model is not initialized")
        if self.current_model_version is None:
            raise RuntimeError("Plan B current model version is unknown")
        existing = self.recover_pending_delta()
        if existing is not None and int(existing.get("model_version", -1) or -1) == int(self.current_model_version):
            _plan_b_log(
                f"Reusing pending delta spool for model v{self.current_model_version}: "
                f"{existing.get('spool_dir')}"
            )
            return existing

        total_entries = 0
        total_bytes = 0
        total_norm_sq = 0.0
        delta_norm_per_layer: Dict[str, float] = {}
        layer_files: List[str] = []
        ratio = float(getattr(self.args, "delta_compression_ratio", 0.005) or 0.005)
        spool_dir = self._spool_dir(self.current_model_version)
        if spool_dir.exists():
            for path in spool_dir.glob("*"):
                with contextlib.suppress(Exception):
                    path.unlink()
        spool_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._spool_manifest_path(spool_dir)

        for name, param in self.model.named_parameters():
            global_param = self.load_global_param(name)
            delta = param.detach().cpu().float() - global_param.float()
            flat = delta.flatten()
            if flat.numel() == 0:
                continue
            k = max(1, int(flat.numel() * ratio))
            _, topk_idx = torch.topk(flat.abs(), k)
            topk_vals = flat[topk_idx].float()
            layer_norm = float(delta.norm().item())
            delta_norm_per_layer[name] = layer_norm
            total_norm_sq += layer_norm ** 2

            payload = {
                "indices": topk_idx.to(torch.long),
                "values": topk_vals,
                "shape": tuple(param.shape),
            }
            safe_name = _safe_layer_name(name)
            output_path = spool_dir / f"{safe_name}.pt"
            torch.save(payload, output_path)
            total_entries += int(topk_idx.numel())
            total_bytes += output_path.stat().st_size
            layer_files.append(output_path.name)

        delta_norm = total_norm_sq ** 0.5
        metadata = {
            "state": "pending_upload",
            "created_at": time.time(),
            "spool_dir": str(spool_dir),
            "manifest_path": str(manifest_path),
            "base_model_version": int(self.current_model_version),
            "model_version": int(self.current_model_version),
            "total_entries": total_entries,
            "total_bytes": total_bytes,
            "delta_norm": delta_norm,
            "delta_norm_per_layer": delta_norm_per_layer,
            "layer_files": layer_files,
            "completed_shards": 0,
            "completed_effective_tokens": 0,
            "batch_size": 0,
        }
        self._write_spool_manifest(metadata)
        materialized = self._read_spool_manifest(manifest_path)
        if materialized is None:
            raise RuntimeError("Failed to materialize delta spool manifest")
        _plan_b_log(
            f"Compressed delta: layers={len(layer_files)}, entries={total_entries}, bytes={total_bytes}, "
            f"norm={delta_norm:.4f}, spool={spool_dir}"
        )
        return materialized

    def submit_delta(self, metadata: Dict[str, Any]) -> bool:
        manifest = dict(metadata)
        manifest["state"] = "uploading"
        self._write_spool_manifest(manifest)
        layer_files = metadata.get("layer_files") or []
        for filepath in layer_files:
            layer_name = Path(filepath).stem
            headers = {
                "X-Layer-Name": layer_name,
                "X-Miner-Id": self.miner_address,
            }
            headers.update(self._headers())
            try:
                with open(filepath, "rb") as handle:
                    resp = requests.post(
                        f"{self.aggregator_url}/delta/upload_layer",
                        data=handle.read(),
                        headers=headers,
                        timeout=120,
                    )
                if resp.status_code in (404, 501):
                    _plan_b_log(f"Delta upload endpoint unavailable ({resp.status_code}); deferring until Day 2")
                    manifest["state"] = "pending_upload"
                    self._write_spool_manifest(manifest)
                    return False
                resp.raise_for_status()
            except Exception as exc:
                _plan_b_log(f"Delta upload failed for {layer_name}: {exc}")
                manifest["state"] = "pending_upload"
                self._write_spool_manifest(manifest)
                return False

        finalize_payload = {
            "miner_id": self.miner_address,
            "completed_shards": int(metadata.get("completed_shards", 0) or 0),
            "batch_size": int(metadata.get("batch_size", 0) or 0),
            "completed_effective_tokens": int(metadata.get("completed_effective_tokens", 0) or 0),
            "delta_norm": float(metadata.get("delta_norm", 0.0) or 0.0),
            "model_version": int(metadata.get("model_version", self.current_model_version or 0) or 0),
        }
        try:
            resp = requests.post(
                f"{self.aggregator_url}/delta/finalize",
                json=finalize_payload,
                headers=self._headers(),
                timeout=30,
            )
            if resp.status_code in (404, 501):
                _plan_b_log(f"Delta finalize endpoint unavailable ({resp.status_code}); deferring until Day 2")
                manifest["state"] = "pending_upload"
                self._write_spool_manifest(manifest)
                return False
            resp.raise_for_status()
        except Exception as exc:
            _plan_b_log(f"Delta finalize failed: {exc}")
            manifest["state"] = "pending_upload"
            self._write_spool_manifest(manifest)
            return False

        manifest["state"] = "uploaded"
        self._write_spool_manifest(manifest)
        self._cleanup_spool(manifest)
        _plan_b_log("Delta upload complete")
        return True

    def _fetch_status(self, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        if not force and self._status_cache is not None and (now - self._status_cache_ts) < STATUS_CACHE_TTL_S:
            return self._status_cache
        candidates = [
            f"{self.ps_url}/status",
            f"{self.ps_url}/epoch/current",
            f"{self.ps_url}/health",
        ]
        for url in candidates:
            try:
                resp = requests.get(url, headers=self._headers(), timeout=10)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if isinstance(data, dict):
                    self._status_cache = data
                    self._status_cache_ts = now
                    return data
            except Exception:
                continue
        self._status_cache = {}
        self._status_cache_ts = now
        return {}

    def _fetch_model_info(self, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        if not force and self._model_info_cache is not None and (now - self._model_info_cache_ts) < MODEL_INFO_CACHE_TTL_S:
            return self._model_info_cache
        try:
            resp = requests.get(
                f"{self.ps_url}/model/info",
                headers=self._headers(),
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    self._model_info_cache = data
                    self._model_info_cache_ts = now
                    return data
        except Exception:
            pass
        self._model_info_cache = {}
        self._model_info_cache_ts = now
        return {}

    def _current_ps_model_version(self) -> Optional[int]:
        status = self._fetch_status(force=True)
        for key in ("model_version", "version", "current_version"):
            value = status.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return None

    def _full_model_path(self, version: int) -> Path:
        return PLAN_B_MODEL_DIR / f"full_model_v{version}.pt"

    def _epoch_update_path(self, version: int) -> Path:
        return PLAN_B_MODEL_DIR / f"update_v{version}.pt"

    def _publication_state(self, force: bool = False) -> Dict[str, Any]:
        status = self._fetch_status(force=force)
        info = self._fetch_model_info(force=force)

        target_version = _pick_first_version(
            info.get("live_version"),
            status.get("model_version"),
            info.get("model_version"),
            info.get("version"),
            status.get("version"),
            status.get("current_version"),
        ) or 0
        published_full_version = _pick_first_version(
            info.get("published_full_version"),
            info.get("version"),
            target_version,
        ) or target_version
        published_update_version = _pick_first_version(
            info.get("published_update_version"),
            target_version,
        ) or target_version
        if target_version > 0:
            published_full_version = max(0, min(published_full_version, target_version))
            published_update_version = max(0, min(published_update_version, target_version))

        full_model_base_urls = _parse_url_candidates(info.get("full_model_base_urls"))
        if not full_model_base_urls:
            full_model_base_urls = list(MODEL_MIRRORS)

        epoch_update_base_urls = _parse_url_candidates(info.get("epoch_update_base_urls"))
        if not epoch_update_base_urls:
            epoch_update_base_urls = list(EPOCH_UPDATE_MIRRORS)

        return {
            "status": status,
            "info": info,
            "target_version": target_version,
            "bootstrap_version": published_full_version,
            "published_full_version": published_full_version,
            "published_update_version": published_update_version,
            "full_model_base_urls": full_model_base_urls,
            "epoch_update_base_urls": epoch_update_base_urls,
        }

    def _download_full_model_direct(self, version: int, model_path: Path) -> None:
        ok = miner_lib.download_model_streaming(self.ps_url, model_path, auth_token=self.token)
        if not ok:
            raise RuntimeError(f"[PLAN-B] Full model download failed for version {version}")

    def _download_full_model_from_mirrors(
        self,
        version: int,
        model_path: Path,
        mirror_urls: Optional[List[str]] = None,
    ) -> None:
        model_filename = f"v{version}_full.pt"
        tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
        last_error: Optional[Exception] = None
        mirrors = mirror_urls or list(MODEL_MIRRORS)

        for mirror in mirrors:
            file_url = f"{mirror.rstrip('/')}/{model_filename}"
            _plan_b_log(f"[DOWNLOAD] Trying {file_url}")
            try:
                total_bytes = miner_lib._stream_download_with_resume(file_url, tmp_path, timeout_s=600)
                _ = torch.load(tmp_path, map_location="cpu", mmap=True, weights_only=True)
                os.replace(tmp_path, model_path)
                _plan_b_log(
                    f"[DOWNLOAD] Full model mirror download complete: {total_bytes / 1e9:.2f} GB"
                )
                return
            except Exception as exc:
                last_error = exc
                _plan_b_log(f"[DOWNLOAD] Mirror failed: {exc}")

        if last_error is not None:
            _plan_b_log("[DOWNLOAD] All mirrors failed, falling back to PS /model")
        self._download_full_model_direct(version, model_path)

    def _load_epoch_update_from_path(self, path: Path) -> Dict[str, Any]:
        return torch.load(path, map_location="cpu", weights_only=True)

    def _download_epoch_update_from_mirrors(
        self,
        version: int,
        base_urls: List[str],
    ) -> Optional[Dict[str, Any]]:
        update_path = self._epoch_update_path(version)
        tmp_path = update_path.with_suffix(update_path.suffix + ".tmp")
        if update_path.exists():
            try:
                return self._load_epoch_update_from_path(update_path)
            except Exception as exc:
                _plan_b_log(f"[DOWNLOAD] Cached epoch update v{version} invalid, re-downloading: {exc}")
                with contextlib.suppress(FileNotFoundError):
                    update_path.unlink()

        for base_url in base_urls:
            file_url = f"{base_url.rstrip('/')}/update_v{version}.pt"
            _plan_b_log(f"[DOWNLOAD] Trying epoch update mirror {file_url}")
            try:
                total_bytes = miner_lib._stream_download_with_resume(file_url, tmp_path, timeout_s=600)
                update = self._load_epoch_update_from_path(tmp_path)
                os.replace(tmp_path, update_path)
                _plan_b_log(
                    f"[DOWNLOAD] Epoch update mirror download complete: v{version} {total_bytes / 1e6:.1f} MB"
                )
                return update
            except Exception as exc:
                _plan_b_log(f"[DOWNLOAD] Epoch update mirror failed: {exc}")
        return None

    def _download_epoch_update_from_ps(self, from_version: int, version: int) -> Optional[Dict[str, Any]]:
        update_path = self._epoch_update_path(version)
        tmp_path = update_path.with_suffix(update_path.suffix + ".tmp")
        try:
            url = f"{self.ps_url.rstrip('/')}/model/epoch_update?from_version={int(from_version)}"
            try:
                total_bytes = miner_lib._stream_download_with_resume(url, tmp_path, timeout_s=600)
            except requests.HTTPError as exc:
                response = exc.response
                if response is not None and response.status_code in (404, 501):
                    _plan_b_log("Epoch update endpoint unavailable; using local model until next retry")
                    return None
                if response is not None and response.status_code == 410:
                    raise RuntimeError("expired")
                raise
            update = self._load_epoch_update_from_path(tmp_path)
            os.replace(tmp_path, update_path)
            _plan_b_log(
                f"[DOWNLOAD] Epoch update fallback download complete: v{version} {total_bytes / 1e6:.1f} MB"
            )
            return update
        except RuntimeError:
            raise
        except Exception as exc:
            _plan_b_log(f"Epoch update request failed, keeping current model: {exc}")
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            return None

    def _apply_epoch_update_payload(self, update: Dict[str, Any], from_version: int) -> None:
        if self.model is None:
            raise RuntimeError("Plan B model unexpectedly missing")

        named_params = dict(self.model.named_parameters())
        with torch.no_grad():
            chunks = update.get("chunks", [])
            if isinstance(chunks, list) and chunks:
                for chunk in chunks:
                    name = chunk.get("name")
                    param = named_params.get(name)
                    if param is None:
                        continue
                    indices = chunk["indices"].long()
                    values = chunk["values"].float()
                    param_flat = param.data.view(-1)
                    param_flat[indices.to(param.device)] += values.to(param.device)
            else:
                for name, delta in update.items():
                    if name in {"old_version", "new_version", "chunks"}:
                        continue
                    param = named_params.get(name)
                    if param is None or not torch.is_tensor(delta):
                        continue
                    param.data.add_(delta.to(param.device, dtype=param.dtype))
        self.current_model_version = int(update.get("new_version", from_version + 1))
        self._write_local_version_marker(self.current_model_version)
        _plan_b_log(f"Applied sparse epoch update v{self.current_model_version}")

    def _select_full_download_version(self, target_version: int, bootstrap_version: int) -> int:
        if bootstrap_version > 0 and target_version - bootstrap_version <= MAX_INCREMENTAL_CATCHUP_GAP:
            return bootstrap_version
        return target_version


    def _find_best_local_model(self, target_version: Optional[int]) -> Optional[int]:
        marker_version = self._read_local_version_marker()
        if marker_version is not None:
            marker_path = self._full_model_path(marker_version)
            if marker_path.exists() and (target_version is None or marker_version <= target_version):
                return marker_version
        best_version: Optional[int] = None
        for path in PLAN_B_MODEL_DIR.glob("full_model_v*.pt"):
            try:
                version = int(path.stem.split("_v", 1)[1])
            except (IndexError, ValueError):
                continue
            if target_version is not None and version > target_version:
                continue
            if best_version is None or version > best_version:
                best_version = version
        return best_version

    def _load_model_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> torch.nn.Module:
        alice_config = miner_lib.AliceConfig()
        embed_weight = state_dict.get("model.embed_tokens.weight")
        if not isinstance(embed_weight, torch.Tensor) or embed_weight.ndim != 2:
            raise RuntimeError("[PLAN-B] Invalid full-model checkpoint: missing model.embed_tokens.weight")
        alice_config.vocab_size = int(embed_weight.shape[0])
        alice_config.hidden_dim = int(embed_weight.shape[1])
        gate_weight = state_dict.get("model.layers.0.mlp.gate_proj.weight")
        if isinstance(gate_weight, torch.Tensor) and gate_weight.ndim == 2:
            alice_config.intermediate_size = int(gate_weight.shape[0])
        inv_freq = state_dict.get("model.layers.0.self_attn.rotary_emb.inv_freq")
        if isinstance(inv_freq, torch.Tensor) and inv_freq.ndim == 1 and int(inv_freq.shape[0]) > 0:
            alice_config.num_attention_heads = max(1, alice_config.hidden_dim // (2 * int(inv_freq.shape[0])))
        alice_config.head_dim = max(1, alice_config.hidden_dim // max(1, int(alice_config.num_attention_heads)))

        layer_indices = set()
        for key in state_dict.keys():
            if key.startswith("model.layers."):
                parts = key.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_indices.add(int(parts[2]))
        if layer_indices:
            alice_config.num_layers = max(layer_indices) + 1

        precision_mode = str(getattr(self.args, "precision", "auto"))
        build_dtype = (
            torch.float16
            if self.device.type in ("cuda", "mps") and precision_mode != "fp32"
            else torch.float32
        )
        prev_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(build_dtype)
            model = miner_lib.AliceForCausalLM(alice_config)
        finally:
            torch.set_default_dtype(prev_dtype)
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
        if self.device.type in ("cuda", "mps") and precision_mode != "fp32":
            model = model.half()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        with torch.no_grad():
            for param in model.parameters():
                param.data = param.data.to(self.device)
            for buf in model.buffers():
                buf.data = buf.data.to(self.device)
        return model

    def download_full_model(
        self,
        version: Optional[int] = None,
        mirror_urls: Optional[List[str]] = None,
    ) -> int:
        publication = self._publication_state(force=True)
        target_version = int(
            version
            if version is not None
            else publication.get("bootstrap_version") or publication.get("target_version") or 0
        )
        model_path = self._full_model_path(target_version)
        if not model_path.exists():
            _plan_b_log(f"Downloading full model for version {target_version}")
            self._download_full_model_from_mirrors(
                target_version,
                model_path,
                mirror_urls=mirror_urls or publication.get("full_model_base_urls"),
            )
        else:
            _plan_b_log(f"Using cached full model: {model_path}")
        state_dict = torch.load(model_path, map_location="cpu", mmap=True, weights_only=True)
        self.model = self._load_model_from_state_dict(state_dict)
        self.current_model_version = target_version
        self._write_local_version_marker(target_version)
        _plan_b_log(f"Loaded full model v{target_version}")
        return target_version

    def apply_epoch_updates(self) -> None:
        publication = self._publication_state(force=True)
        target_version = int(publication.get("target_version") or 0)
        bootstrap_version = int(publication.get("bootstrap_version") or 0)
        published_update_version = int(publication.get("published_update_version") or 0)
        full_model_base_urls = list(publication.get("full_model_base_urls") or MODEL_MIRRORS)
        epoch_update_base_urls = list(publication.get("epoch_update_base_urls") or EPOCH_UPDATE_MIRRORS)
        if target_version is None:
            return
        if self.model is None or self.current_model_version is None:
            local_version = self._find_best_local_model(target_version)
            if local_version is None:
                full_version = self._select_full_download_version(target_version, bootstrap_version)
                self.download_full_model(full_version, mirror_urls=full_model_base_urls)
                return
            initial_gap = target_version - local_version
            if initial_gap > MAX_INCREMENTAL_CATCHUP_GAP:
                _plan_b_log(
                    f"Local v{local_version} too far behind PS v{target_version}, downloading fresh model"
                )
                full_version = self._select_full_download_version(target_version, bootstrap_version)
                self.download_full_model(full_version, mirror_urls=full_model_base_urls)
                return
            _plan_b_log(
                f"Found local model v{local_version}, loading instead of downloading v{target_version}"
            )
            self.download_full_model(version=local_version)
        if target_version is None or target_version <= self.current_model_version:
            return
        available_update_version = min(target_version, max(0, published_update_version))
        gap = available_update_version - self.current_model_version
        if gap > MAX_INCREMENTAL_CATCHUP_GAP:
            _plan_b_log(
                f"Local v{self.current_model_version} too far behind published updates v{available_update_version}, downloading fresh model"
            )
            full_version = self._select_full_download_version(target_version, bootstrap_version)
            self.download_full_model(full_version, mirror_urls=full_model_base_urls)
            return

        while self.current_model_version is not None and target_version is not None and self.current_model_version < target_version:
            from_version = int(self.current_model_version)
            next_version = from_version + 1
            if published_update_version < next_version:
                if published_update_version < target_version:
                    _plan_b_log(
                        f"Published epoch updates currently stop at v{published_update_version}; waiting to catch up to live v{target_version}"
                    )
                return

            update = self._download_epoch_update_from_mirrors(next_version, epoch_update_base_urls)
            if update is None:
                try:
                    update = self._download_epoch_update_from_ps(from_version, next_version)
                except RuntimeError as exc:
                    if str(exc) != "expired":
                        raise
                    _plan_b_log(
                        f"Epoch update v{next_version} expired from local v{self.current_model_version}; downloading fresh full model"
                    )
                    full_version = self._select_full_download_version(target_version, bootstrap_version)
                    self.download_full_model(full_version, mirror_urls=full_model_base_urls)
                    return
            if update is None:
                return

            if update.get("old_version") not in (None, from_version):
                _plan_b_log(
                    f"Epoch update payload mismatch for v{next_version}; expected old_version={from_version}, got {update.get('old_version')}"
                )
                full_version = self._select_full_download_version(target_version, bootstrap_version)
                self.download_full_model(full_version, mirror_urls=full_model_base_urls)
                return
            self._apply_epoch_update_payload(update, from_version)

    def epoch_ending(self) -> bool:
        status = self._fetch_status()
        for key in ("remaining_seconds", "epoch_remaining_seconds", "epoch_remaining_s", "seconds_left"):
            value = status.get(key)
            if isinstance(value, (int, float)):
                return float(value) < SUBMIT_WINDOW_S
            if isinstance(value, str):
                with contextlib.suppress(ValueError):
                    return float(value) < SUBMIT_WINDOW_S
        return (time.time() - self.epoch_start_time) >= TRAINING_WINDOW_S


def notify_shard_complete(
    aggregator_url: str,
    miner_id: str,
    token: str,
    task: Dict[str, Any],
    avg_loss: float,
) -> bool:
    payload = {
        "task_id": task.get("task_id"),
        "shard_id": task.get("shard_id"),
        "loss": float(avg_loss),
    }
    try:
        resp = requests.post(
            f"{_normalize_url(aggregator_url)}/shard/complete",
            json=payload,
            headers=miner_lib._auth_headers(token),
            timeout=15,
        )
        if resp.status_code in (404, 501):
            _plan_b_log(f"Shard-complete endpoint unavailable ({resp.status_code}); telemetry deferred until Day 2")
            return False
        resp.raise_for_status()
        return True
    except Exception as exc:
        _plan_b_log(f"Shard-complete notify failed: {exc}")
        return False


def confirm_shard_complete(
    ps_url: str,
    miner_id: str,
    token: str,
    shard_id: int,
    epoch: Optional[int] = None,
) -> None:
    payload: Dict[str, Any] = {
        "miner_id": str(miner_id),
        "shard_id": int(shard_id),
    }
    if epoch is not None:
        payload["epoch"] = int(epoch)
    try:
        requests.post(
            f"{_normalize_url(ps_url)}/shard/confirm",
            json=payload,
            headers=miner_lib._auth_headers(token),
            timeout=5,
        )
    except Exception:
        pass


def wait_for_next_epoch(trainer: LocalTrainer, poll_interval_s: int = 15) -> None:
    start_version = trainer.current_model_version
    if start_version is None:
        return
    waited_s = 0
    while True:
        time.sleep(poll_interval_s)
        waited_s += poll_interval_s
        publication = trainer._publication_state(force=True)
        published_update_version = int(publication.get("published_update_version") or 0)
        published_full_version = int(publication.get("published_full_version") or 0)
        if max(published_update_version, published_full_version) > int(start_version):
            _plan_b_log(
                f"Detected published next version: full={published_full_version}, "
                f"update={published_update_version}, local={start_version}"
            )
            return
        if waited_s >= SUBMIT_WINDOW_S:
            live_version = int(publication.get("target_version") or 0)
            _plan_b_log(
                f"Still waiting for published next version after {waited_s}s: "
                f"live={live_version}, published_full={published_full_version}, "
                f"published_update={published_update_version}, local={start_version}"
            )
            waited_s = 0


def run_plan_b(args: Any) -> None:
    control_plane_url = _normalize_url(args.ps_url)
    args.model_dir = Path(getattr(args, "model_dir", miner_lib.DEFAULT_MODEL_DIR))
    capabilities = miner_lib.get_hardware_info(getattr(args, "device", None))
    wallet_address = str(args.address or "").strip()
    miner_instance_id = str(args.instance_id).strip() if args.instance_id else None
    lock_fp = miner_lib.acquire_single_instance_lock(miner_instance_id)
    heartbeat_stop: Optional[Any] = None
    heartbeat_re_register: Optional[Any] = None
    trainer: Optional[LocalTrainer] = None
    _ = lock_fp

    while True:
        try:
            route_info = miner_lib.resolve_runtime_route(control_plane_url)
            data_plane_url = str(route_info.get("base_url") or control_plane_url)
            miner_lib.log_runtime_route(route_info, control_plane_url)
            register_response = miner_lib.register_miner_with_retry(
                data_plane_url,
                wallet_address,
                miner_instance_id,
                capabilities,
                retry_seconds=30,
            )
            miner_instance_id = str(register_response.get("instance_id") or register_response.get("miner_id") or wallet_address)
            auth_token = str(register_response.get("token", "")).strip()
            if not auth_token:
                raise RuntimeError("[PLAN-B] Registration returned empty auth token")
            runtime_auth_state = miner_lib._build_runtime_auth_state(
                data_plane_url,
                miner_instance_id,
                capabilities,
                auth_token,
            )

            device = torch.device(capabilities["device_type"])
            trainer = LocalTrainer(
                model=None,
                device=device,
                ps_url=control_plane_url,
                aggregator_url=data_plane_url,
                miner_address=wallet_address,
                token=auth_token,
                args=args,
                miner_instance_id=miner_instance_id,
            )
            pending_delta = trainer.recover_pending_delta()
            if pending_delta is not None:
                _plan_b_log(f"Found pending delta spool at startup: {pending_delta.get('spool_dir')}")
                if not trainer.submit_delta(pending_delta):
                    _plan_b_log("Pending delta upload failed at startup; re-registering before training")
                    time.sleep(10)
                    continue
            trainer.apply_epoch_updates()
            heartbeat_stop, heartbeat_re_register, _thread = miner_lib.start_heartbeat_loop(
                runtime_auth_state,
            )

            while True:
                if heartbeat_re_register is not None and heartbeat_re_register.is_set():
                    _plan_b_log("Heartbeat requested re-registration")
                    break
                pending_delta = trainer.recover_pending_delta()
                if pending_delta is not None:
                    _plan_b_log(f"Retrying pending delta spool before next epoch: {pending_delta.get('spool_dir')}")
                    if not trainer.submit_delta(pending_delta):
                        _plan_b_log("Pending delta upload still failing; re-registering before new epoch")
                        break
                trainer.mark_epoch_start()
                trainer.apply_epoch_updates()
                trainer.save_global_snapshot()
                completed_shards = 0
                completed_effective_tokens = 0
                needs_re_registration = False

                while not trainer.epoch_ending():
                    if heartbeat_re_register is not None and heartbeat_re_register.is_set():
                        _plan_b_log("Re-registering before next shard assignment")
                        needs_re_registration = True
                        break
                    task, status = miner_lib.request_task_with_retry(
                        data_plane_url,
                        miner_instance_id,
                        capabilities,
                        auth_token=auth_token,
                        retry_delay=15,
                        max_attempts=5,
                    )
                    if status == "no_task":
                        time.sleep(5)
                        continue
                    if status == "re_register" or task is None:
                        _plan_b_log("Task request requires re-registration")
                        needs_re_registration = True
                        break

                    trainer.update_task_batch_size(task)
                    shard_id = task.get("shard_id")
                    _plan_b_log(f"Downloading shard {shard_id}")
                    shard_data = miner_lib.download_shard_streaming(data_plane_url, int(shard_id), auth_token=auth_token)
                    if shard_data is None:
                        _plan_b_log(f"Shard download failed for {shard_id}")
                        continue
                    avg_loss, shard_batch_size, completed = trainer.train_shard_local(shard_data)
                    if not completed or avg_loss is None:
                        _plan_b_log(f"Skipping shard {shard_id} after failed local training")
                        continue
                    notify_shard_complete(data_plane_url, wallet_address, auth_token, task, avg_loss)
                    confirm_shard_complete(
                        trainer.ps_url,
                        trainer.miner_address,
                        trainer.token,
                        int(shard_id),
                        task.get("epoch"),
                    )
                    completed_shards += 1
                    completed_effective_tokens += int(shard_batch_size)

                if heartbeat_re_register is not None and heartbeat_re_register.is_set():
                    needs_re_registration = True
                if needs_re_registration:
                    if completed_shards == 0:
                        _plan_b_log("Zero shards trained before re-registration, skipping delta submission")
                    break

                if completed_shards == 0:
                    _plan_b_log("Zero shards trained this epoch, skipping delta submission")
                    wait_for_next_epoch(trainer)
                    continue

                metadata = trainer.compute_and_compress_delta()
                metadata["completed_shards"] = completed_shards
                metadata["batch_size"] = int(
                    (completed_effective_tokens // completed_shards)
                    if completed_shards > 0
                    else max(1, int(trainer.effective_batch_size or trainer._target_batch_size() or 2))
                )
                metadata["completed_effective_tokens"] = int(completed_effective_tokens)
                if completed_shards > 0 and (completed_effective_tokens % completed_shards) != 0:
                    _plan_b_log(
                        f"Mixed shard batch sizes this epoch: completed_shards={completed_shards}, "
                        f"completed_effective_tokens={completed_effective_tokens}, "
                        f"reporting average batch_size={metadata['batch_size']}"
                    )
                trainer._write_spool_manifest(dict(metadata))
                success = trainer.submit_delta(metadata)
                if not success:
                    _plan_b_log("Delta upload failed, re-registering and retrying once")
                    try:
                        register_response = miner_lib.register_miner_with_retry(
                            data_plane_url,
                            wallet_address,
                            miner_instance_id,
                            capabilities,
                            retry_seconds=10,
                        )
                        miner_instance_id = str(
                            register_response.get("instance_id")
                            or register_response.get("miner_id")
                            or miner_instance_id
                            or wallet_address
                        )
                        new_token = str(register_response.get("token", "")).strip()
                        if new_token:
                            auth_token = new_token
                            trainer.token = new_token
                            miner_lib._update_runtime_auth_state(
                                runtime_auth_state,
                                data_plane_url=data_plane_url,
                                miner_id=miner_instance_id,
                                capabilities=capabilities,
                                auth_token=new_token,
                            )
                            success = trainer.submit_delta(metadata)
                            if success:
                                _plan_b_log("Delta upload succeeded after re-register")
                            else:
                                _plan_b_log("Delta upload still failed after re-register, skipping this epoch")
                        else:
                            _plan_b_log("Re-register returned empty token; skipping delta retry")
                    except Exception as exc:
                        _plan_b_log(f"Re-register for delta retry failed: {exc}")
                wait_for_next_epoch(trainer)
        except KeyboardInterrupt:
            if heartbeat_stop is not None:
                heartbeat_stop.set()
            _plan_b_log("Miner stopped by user")
            return
        except Exception as exc:
            _plan_b_log(f"Unexpected error: {exc}. Restarting in 30s")
            time.sleep(30)
        finally:
            if heartbeat_stop is not None:
                heartbeat_stop.set()
                heartbeat_stop = None
            heartbeat_re_register = None
