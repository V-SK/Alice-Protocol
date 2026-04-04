#!/usr/bin/env python3
"""Plan B miner runtime with epoch-level local SGD and sparse delta upload."""

import contextlib
import io
import os
import tempfile
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


def _plan_b_log(message: str) -> None:
    print(f"[PLAN-B] {message}")


def _safe_layer_name(name: str) -> str:
    return name.replace(".", "_")


def _normalize_url(url: str) -> str:
    return str(url or "").strip().rstrip("/")


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
        self._status_cache: Optional[Dict[str, Any]] = None
        self._status_cache_ts: float = 0.0
        self.epoch_start_time: float = time.time()

    def mark_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

    def _headers(self) -> Dict[str, str]:
        return miner_lib._auth_headers(self.token)

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

    def train_shard_local(self, shard_data: Any) -> float:
        if self.model is None:
            raise RuntimeError("Plan B model is not initialized")
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        local_lr = float(self.args.local_lr)
        hooks: List[Any] = []
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
        tokens = _extract_tokens(shard_data)
        seq_len = int(getattr(self.args, "seq_len", 128) or 128)
        max_batches = int(getattr(self.args, "max_batches", 10) or 10)
        batch_size = max(1, int(getattr(self.args, "batch_size", 0) or 2))
        start_idx = 0
        total_loss = 0.0
        num_batches = 0

        try:
            while start_idx < max(1, tokens.numel() - seq_len - 1) and num_batches < max_batches:
                batch_inputs: List[torch.Tensor] = []
                batch_labels: List[torch.Tensor] = []
                for batch_offset in range(batch_size):
                    offset = start_idx + batch_offset * seq_len
                    if offset + seq_len + 1 > tokens.numel():
                        break
                    chunk = tokens[offset : offset + seq_len + 1]
                    batch_inputs.append(chunk[:-1])
                    batch_labels.append(chunk[1:])
                if not batch_inputs:
                    break

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
        finally:
            for hook in hooks:
                hook.remove()
            self.model.zero_grad(set_to_none=True)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        _plan_b_log(f"Local shard training complete: batches={num_batches}, avg_loss={avg_loss:.4f}")
        return avg_loss

    def compute_and_compress_delta(self) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Plan B model is not initialized")
        total_entries = 0
        total_bytes = 0
        total_norm_sq = 0.0
        delta_norm_per_layer: Dict[str, float] = {}
        layer_files: List[str] = []
        ratio = float(getattr(self.args, "delta_compression_ratio", 0.005) or 0.005)

        for path in self.delta_outbox_dir.glob("*.pt"):
            with contextlib.suppress(Exception):
                path.unlink()

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
            output_path = self.delta_outbox_dir / f"{safe_name}.pt"
            torch.save(payload, output_path)
            total_entries += int(topk_idx.numel())
            total_bytes += output_path.stat().st_size
            layer_files.append(str(output_path))

        delta_norm = total_norm_sq ** 0.5
        _plan_b_log(
            f"Compressed delta: layers={len(layer_files)}, entries={total_entries}, bytes={total_bytes}, norm={delta_norm:.4f}"
        )
        return {
            "total_entries": total_entries,
            "total_bytes": total_bytes,
            "delta_norm": delta_norm,
            "delta_norm_per_layer": delta_norm_per_layer,
            "layer_files": layer_files,
            "model_version": self.current_model_version,
        }

    def submit_delta(self, metadata: Dict[str, Any]) -> bool:
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
                    return False
                resp.raise_for_status()
            except Exception as exc:
                _plan_b_log(f"Delta upload failed for {layer_name}: {exc}")
                return False

        finalize_payload = {
            "miner_id": self.miner_address,
            "completed_shards": int(metadata.get("completed_shards", 0) or 0),
            "batch_size": int(metadata.get("batch_size", 0) or 0),
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
                return False
            resp.raise_for_status()
        except Exception as exc:
            _plan_b_log(f"Delta finalize failed: {exc}")
            return False

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

    def _find_best_local_model(self, target_version: Optional[int]) -> Optional[int]:
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
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        return model.to(self.device)

    def download_full_model(self, version: Optional[int] = None) -> int:
        target_version = int(version if version is not None else self._current_ps_model_version() or 0)
        model_path = self._full_model_path(target_version)
        if not model_path.exists():
            _plan_b_log(f"Downloading full model for version {target_version}")
            ok = miner_lib.download_model_streaming(self.ps_url, model_path, auth_token=self.token)
            if not ok:
                raise RuntimeError("[PLAN-B] Full model download failed")
        else:
            _plan_b_log(f"Using cached full model: {model_path}")
        state_dict = torch.load(model_path, map_location="cpu", mmap=True, weights_only=True)
        self.model = self._load_model_from_state_dict(state_dict)
        self.current_model_version = target_version
        _plan_b_log(f"Loaded full model v{target_version}")
        return target_version

    def apply_epoch_updates(self) -> None:
        target_version = self._current_ps_model_version()
        if self.model is None or self.current_model_version is None:
            local_version = self._find_best_local_model(target_version)
            if local_version is None:
                self.download_full_model(target_version)
                return
            _plan_b_log(
                f"Found local model v{local_version}, loading instead of downloading v{target_version}"
            )
            self.download_full_model(version=local_version)
        if target_version is None or target_version <= self.current_model_version:
            return
        gap = target_version - self.current_model_version
        if gap > 20:
            _plan_b_log(f"Model gap {gap} > 20, falling back to full model download")
            self.download_full_model(target_version)
            return

        while self.current_model_version is not None and target_version is not None and self.current_model_version < target_version:
            from_version = int(self.current_model_version)
            try:
                resp = requests.get(
                    f"{self.ps_url}/model/epoch_update",
                    params={"from_version": from_version},
                    headers=self._headers(),
                    timeout=120,
                )
            except Exception as exc:
                _plan_b_log(f"Epoch update request failed, keeping current model: {exc}")
                return

            if resp.status_code in (404, 501):
                _plan_b_log("Epoch update endpoint unavailable; using local model until Day 2")
                return
            if resp.status_code == 410:
                _plan_b_log(
                    f"Epoch update expired; continuing with local model v{self.current_model_version} "
                    f"(gap={target_version - self.current_model_version})"
                )
                return
            resp.raise_for_status()
            update = _load_update_payload(resp.content)
            if self.model is None:
                raise RuntimeError("Plan B model unexpectedly missing")

            named_params = dict(self.model.named_parameters())
            with torch.no_grad():
                for chunk in update.get("chunks", []):
                    name = chunk.get("name")
                    param = named_params.get(name)
                    if param is None:
                        continue
                    indices = chunk["indices"].long()
                    values = chunk["values"].float()
                    param_flat = param.data.view(-1)
                    param_flat[indices.to(param.device)] += values.to(param.device)
            self.current_model_version = int(update.get("new_version", from_version + 1))
            _plan_b_log(f"Applied sparse epoch update v{self.current_model_version}")

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


def wait_for_next_epoch(trainer: LocalTrainer, poll_interval_s: int = 15) -> None:
    start_version = trainer.current_model_version
    for _ in range(max(1, SUBMIT_WINDOW_S // poll_interval_s)):
        time.sleep(poll_interval_s)
        latest = trainer._current_ps_model_version()
        if latest is not None and start_version is not None and latest > start_version:
            _plan_b_log(f"Detected new model version {latest}, continuing")
            return
    _plan_b_log("Waited for next epoch window; continuing with best-effort local timing")


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
            trainer.apply_epoch_updates()
            heartbeat_stop, heartbeat_re_register, _thread = miner_lib.start_heartbeat_loop(
                data_plane_url,
                miner_instance_id,
                capabilities,
                auth_token=auth_token,
            )

            while True:
                if heartbeat_re_register is not None and heartbeat_re_register.is_set():
                    _plan_b_log("Heartbeat requested re-registration")
                    break
                trainer.mark_epoch_start()
                trainer.apply_epoch_updates()
                trainer.save_global_snapshot()
                completed_shards = 0

                while not trainer.epoch_ending():
                    if heartbeat_re_register is not None and heartbeat_re_register.is_set():
                        _plan_b_log("Re-registering before next shard assignment")
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
                        break

                    shard_id = task.get("shard_id")
                    _plan_b_log(f"Downloading shard {shard_id}")
                    shard_data = miner_lib.download_shard_streaming(data_plane_url, int(shard_id), auth_token=auth_token)
                    if shard_data is None:
                        _plan_b_log(f"Shard download failed for {shard_id}")
                        continue
                    avg_loss = trainer.train_shard_local(shard_data)
                    notify_shard_complete(data_plane_url, wallet_address, auth_token, task, avg_loss)
                    completed_shards += 1

                if heartbeat_re_register is not None and heartbeat_re_register.is_set():
                    break

                metadata = trainer.compute_and_compress_delta()
                metadata["completed_shards"] = completed_shards
                metadata["batch_size"] = int(getattr(args, "batch_size", 0) or 2)
                trainer.submit_delta(metadata)
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
