# Alice Miner

This is the standalone miner client for Alice Protocol.

Current access policy:
- repository is private
- outside miner admission remains restricted

## Quick Start

### macOS / Linux

```bash
./miner/bootstrap.sh
```

### Windows

```powershell
.\miner\bootstrap.ps1
```

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

Create a local wallet:

```bash
python3 miner/alice_wallet.py create
```

Use an existing reward address:

```bash
python3 miner/alice_miner.py \
  --ps-url https://ps.aliceprotocol.org \
  --address aYourAliceAddress \
  --reward-address aYourRewardAddress
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
- Apple Silicon `>= 24GB`: supported
- CPU `>= 32GB RAM`: supported but very slow
- `< 20GB`: not supported

CPU mining is supported but not recommended. Expect roughly `1/50 - 1/100` of GPU throughput and proportionally lower rewards.

For CUDA miners, registration uses the GPU's physical VRAM. If a node was previously downscaled after OOM recovery, the local runtime cap may still live in:

- `~/.alice/device_profile.json`
- `ALICE_MEMORY_CAP_GB`

That runtime cap is separate from physical VRAM and is only meant to tune local stability.
