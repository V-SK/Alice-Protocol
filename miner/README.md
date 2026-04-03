# Alice Miner

This is the standalone miner client for Alice Protocol.

## Quick Start

### macOS / Linux

```bash
./miner/bootstrap.sh --ps-url https://ps.aliceprotocol.org --address YOUR_ADDRESS
```

### Windows

```powershell
.\miner\bootstrap.ps1 -PsUrl https://ps.aliceprotocol.org -Address YOUR_ADDRESS
```

If you omit `--address`, the miner will create or reuse `~/.alice/wallet.json` automatically.

## Long-Running Mode

Foreground bootstrap is the default for first-run setup and debugging.

For long-running operation with automatic restart:

- Linux/macOS: `./miner/install-service.sh`
- Windows: `.\miner\install-service.ps1`

Linux systemd installation requires `sudo`.

Service logs are written to `~/.alice/logs/` on Unix-like systems and `%USERPROFILE%\.alice\logs\` on Windows.

Optional service overrides:

- Unix: `~/.alice/miner-service.env`
- Windows: `~\.alice\miner-service.ps1`

Use the service manager commands after installation:

- Linux/macOS: `./miner/start-service.sh`, `./miner/stop-service.sh`, `./miner/status-service.sh`, `./miner/uninstall-service.sh`
- Windows: `.\miner\start-service.ps1`, `.\miner\stop-service.ps1`, `.\miner\status-service.ps1`, `.\miner\uninstall-service.ps1`

## Wallet

Use your own wallet address:

```bash
./miner/bootstrap.sh \
  --ps-url https://ps.aliceprotocol.org \
  --address aYourAliceAddress
```

Create a local wallet explicitly:

```bash
python3 miner/alice_wallet.py create
```

If you do not pass `--address`, bootstrap will auto-create or reuse:

- `~/.alice/wallet.json`

To recover the saved address and mnemonic:

```bash
cat ~/.alice/wallet.json
```

Use a separate reward address:

```bash
./miner/bootstrap.sh \
  --ps-url https://ps.aliceprotocol.org \
  --address aYourAliceAddress \
  --reward-address aYourRewardAddress
```

## Multi-GPU

Run one miner process per GPU. There is no dedicated `--gpu` flag; use `CUDA_VISIBLE_DEVICES` and a unique `--instance-id` for each process.

Recommended pattern:

1. Run `./miner/bootstrap.sh` once to prepare the virtual environment and shared model cache.
2. Start additional GPU instances with `./miner/run_miner.sh`.

Example for 2 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 ./miner/bootstrap.sh \
  --ps-url https://ps.aliceprotocol.org \
  --address YOUR_ADDRESS \
  --instance-id gpu0

CUDA_VISIBLE_DEVICES=1 ./miner/run_miner.sh \
  --ps-url https://ps.aliceprotocol.org \
  --address YOUR_ADDRESS \
  --instance-id gpu1
```

All instances can share the same address. Rewards aggregate automatically to that wallet.

## Background Mode

Run in the background with `nohup`:

```bash
CUDA_VISIBLE_DEVICES=0 nohup ./miner/run_miner.sh \
  --ps-url https://ps.aliceprotocol.org \
  --address YOUR_ADDRESS \
  --instance-id gpu0 \
  > /tmp/miner_gpu0.log 2>&1 &
```

Stop all miners:

```bash
pkill -f alice_miner.py
```

Stop a specific instance:

```bash
kill "$(pgrep -f 'instance-id gpu0')"
```

## Epoch Reports

Miner epoch reports are written to:

- `~/.alice/reports/miner_epoch_reports.jsonl`
- `~/.alice/reports/epochs/miner_epoch_<epoch>.md`

Each report includes work completed, loss, gradient submission counts, and reward status (`confirmed` or `pending`).

## Hardware

See `docs/HARDWARE_REQUIREMENTS.md` for the full matrix.

Summary:
- CUDA GPU `>= 24GB`: recommended
- CUDA GPU `16GB`: supported, slower (`batch_size=1`)
- Apple Silicon `16GB`: supported, very slow (macOS swap, `batch_size=1`)
- Apple Silicon `>= 24GB`: supported
- Apple Silicon `>= 32GB`: recommended
- CPU `>= 32GB RAM`: supported but very slow
- `< 16GB`: not supported

CPU mining is supported but not recommended. Expect roughly `1/50 - 1/100` of GPU throughput and proportionally lower rewards.

For CUDA miners, registration uses the GPU's physical VRAM. If a node was previously downscaled after OOM recovery, the local runtime cap may still live in:

- `~/.alice/device_profile.json`
- `ALICE_MEMORY_CAP_GB`

That runtime cap is separate from physical VRAM and is only meant to tune local stability.
