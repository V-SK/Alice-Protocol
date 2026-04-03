# Alice Protocol

Public Alice Protocol miner and scorer distribution.

This repository contains:
- `miner/`: external miner client
- `scorer/`: external scorer worker
- `shared/`: shared Alice model runtime

## Repository Layout

```text
Alice-Protocol/
├── miner/
├── scorer/
├── shared/
├── core/
├── docs/
└── LICENSE
```

## Components

### Miner

Entry point: `miner/alice_miner.py`

Bootstrap:

```bash
./miner/bootstrap.sh --ps-url https://ps.aliceprotocol.org --address YOUR_ADDRESS
```

Windows:

```powershell
.\miner\bootstrap.ps1 -PsUrl https://ps.aliceprotocol.org -Address YOUR_ADDRESS
```

Managed service:

- Linux/macOS: `./miner/install-service.sh`
- Windows: `.\miner\install-service.ps1`

Guides:

- `miner/README.md`
- `docs/MINER_GUIDE.md`
- `docs/MINER_GUIDE_CN.md`

### Scorer

Entry point: `scorer/scoring_server.py`

Bootstrap:

```bash
./scorer/bootstrap.sh
```

Windows:

```powershell
.\scorer\bootstrap.ps1
```

Managed service:

- Linux/macOS: `./scorer/install-service.sh`
- Windows: `.\scorer\install-service.ps1`

## Documentation

- Miner:
  - `miner/README.md`
  - `docs/MINER_GUIDE.md`
  - `docs/MINER_GUIDE_CN.md`
- Scorer:
  - `docs/SCORER_GUIDE.md`
- Hardware:
  - `docs/HARDWARE_REQUIREMENTS.md`

## Notes

- Bootstrap is the default user entry point for both miner and scorer.
- Managed services are available for Linux, macOS, and Windows.
- Per-epoch local reports are written to `~/.alice/reports/`.
- Scorer bootstrap auto-fetches the held-out validation shards into `scorer/data/validation` by default.
- Multi-GPU mining uses one miner process per GPU via `CUDA_VISIBLE_DEVICES` plus unique `--instance-id` values.
- Miner rewards go to `--reward-address` when provided, otherwise to `--address`.
