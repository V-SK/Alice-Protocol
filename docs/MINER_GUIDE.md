# Alice Miner Guide

This guide reflects the current standalone miner layout in `Alice-Protocol`.

Current rollout state:
- repository is private
- outside miner admission remains restricted

## 1. Install

macOS / Linux:

```bash
./miner/bootstrap.sh
```

Windows:

```powershell
.\miner\bootstrap.ps1
```

For long-running operation with automatic restart:

- Linux/macOS: `./miner/install-service.sh`
- Windows: `.\miner\install-service.ps1`

Linux systemd installation requires `sudo`.

## 2. Create or import an address

Create a new local wallet:

```bash
python3 miner/alice_wallet.py create
```

If you already have an Alice address, you can pass it directly with `--address`.

## 3. Start mining

The default bootstrap path will:

- create a repo-local virtual environment
- install missing dependencies
- create a local wallet if needed
- start the miner with `https://ps.aliceprotocol.org` by default

Manual launch is still available:

```bash
python3 miner/alice_miner.py \
  --ps-url https://ps.aliceprotocol.org \
  --address aYourAliceAddress \
  --device auto \
  --precision auto
```

Optional:

```bash
python3 miner/alice_miner.py \
  --ps-url https://ps.aliceprotocol.org \
  --address aControlAddress \
  --reward-address aRewardAddress
```

## 4. Network flow

The miner talks directly to the Parameter Server:
- `/register`
- `/task/request`
- `/task/complete`
- `/model`
- `/model/info`
- `/model/delta`

It does **not** connect directly to the aggregator.

## 5. Hardware guidance

- CUDA GPU `>= 24GB`: recommended
- CUDA GPU `16GB`: supported, slower (`batch_size=1`)
- Apple Silicon `16GB`: supported, very slow (macOS swap, `batch_size=1`)
- Apple Silicon `>= 24GB`: supported
- Apple Silicon `>= 32GB`: recommended
- CPU `>= 32GB RAM`: supported but very slow
- `< 16GB`: not supported

CPU mining is supported but not recommended.

## 6. Rewards

Rewards are paid to:
- `--reward-address` if provided
- otherwise `--address`

Reward timing depends on successful epoch settlement on chain.

## 7. Epoch reports

Miner writes local epoch reports to:

- `~/.alice/reports/miner_epoch_reports.jsonl`
- `~/.alice/reports/epochs/miner_epoch_<epoch>.md`

Each report records:

- tasks requested and trained
- batches trained
- gradients submitted, accepted, rejected
- average loss
- reward status (`confirmed`, `pending`)

## 8. Current release note

This repository is a private release-prep baseline. Cross-platform validation is still in progress before public access opens.

## 9. Managed service mode

Managed mode is recommended for long-running miners.

- Linux uses `systemd`
- macOS uses `launchd`
- Windows uses Task Scheduler

Service logs are written to `~/.alice/logs/` or `%USERPROFILE%\.alice\logs\`.

Optional overrides:

- Unix: `~/.alice/miner-service.env`
- Windows: `~\.alice\miner-service.ps1`

Service manager commands:

- Linux/macOS: `./miner/start-service.sh`, `./miner/stop-service.sh`, `./miner/status-service.sh`, `./miner/uninstall-service.sh`
- Windows: `.\miner\start-service.ps1`, `.\miner\stop-service.ps1`, `.\miner\status-service.ps1`, `.\miner\uninstall-service.ps1`
