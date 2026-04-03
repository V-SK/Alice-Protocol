# Alice Miner Guide

This guide reflects the current standalone miner layout in `Alice-Protocol`.

## 1. Install

macOS / Linux:

```bash
./miner/bootstrap.sh --ps-url https://ps.aliceprotocol.org --address YOUR_ADDRESS
```

Windows:

```powershell
.\miner\bootstrap.ps1 -PsUrl https://ps.aliceprotocol.org -Address YOUR_ADDRESS
```

For long-running operation with automatic restart:

- Linux/macOS: `./miner/install-service.sh`
- Windows: `.\miner\install-service.ps1`

Linux systemd installation requires `sudo`.

## 2. Create or import an address

### Use your own address (recommended)

```bash
./miner/bootstrap.sh \
  --ps-url https://ps.aliceprotocol.org \
  --address aYourAliceAddress
```

You can generate a new Alice address with:

```bash
python3 -c "
from substrateinterface import Keypair
mnemonic = Keypair.generate_mnemonic()
kp = Keypair.create_from_mnemonic(mnemonic, ss58_format=300)
print(f'Address:  {kp.ss58_address}')
print(f'Mnemonic: {mnemonic}')
print()
print('Write down the mnemonic. This is your backup.')
"
```

### Auto-create wallet

```bash
./miner/bootstrap.sh --ps-url https://ps.aliceprotocol.org
```

If you omit `--address`, bootstrap creates or reuses:

- `~/.alice/wallet.json`

To recover the saved address and mnemonic:

```bash
cat ~/.alice/wallet.json
```

Back up the mnemonic immediately. If it is lost, your funds are lost.

## 3. Start mining

The default bootstrap path will:

- create a repo-local virtual environment
- install missing dependencies
- create a local wallet if needed
- start the miner with `https://ps.aliceprotocol.org` by default

Manual launch is also available:

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

## 4. Multi-GPU mining

Run one miner process per GPU. There is no dedicated `--gpu` flag. Use `CUDA_VISIBLE_DEVICES` to bind each process and a unique `--instance-id` to distinguish them.

Recommended pattern on multi-GPU hosts:

1. Run `./miner/bootstrap.sh` once to prepare `.venv`, wallet defaults, and the shared model cache.
2. Launch additional GPU workers with `./miner/run_miner.sh`.

### 2 GPUs

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

### 4 GPUs

```bash
./miner/bootstrap.sh \
  --ps-url https://ps.aliceprotocol.org \
  --address YOUR_ADDRESS \
  --instance-id gpu0

for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i ./miner/run_miner.sh \
    --ps-url https://ps.aliceprotocol.org \
    --address YOUR_ADDRESS \
    --instance-id gpu${i} &
  sleep 5
done
```

### 8 GPUs

```bash
./miner/bootstrap.sh \
  --ps-url https://ps.aliceprotocol.org \
  --address YOUR_ADDRESS \
  --instance-id gpu0

for i in $(seq 1 7); do
  CUDA_VISIBLE_DEVICES=$i ./miner/run_miner.sh \
    --ps-url https://ps.aliceprotocol.org \
    --address YOUR_ADDRESS \
    --instance-id gpu${i} &
  sleep 5
done
```

Key points:

- All instances can use the same address; rewards aggregate to one wallet.
- Each instance must use a unique `--instance-id`.
- Later instances reuse the cached model, but `bootstrap.sh` still re-checks Python packages if you run it again.

## 5. Network flow

The miner talks directly to the Parameter Server:
- `/register`
- `/task/request`
- `/task/complete`
- `/model`
- `/model/info`
- `/model/delta`

It does **not** connect directly to the aggregator.

## 6. Background and service mode

Run in the background with `nohup`:

```bash
CUDA_VISIBLE_DEVICES=0 nohup ./miner/run_miner.sh \
  --ps-url https://ps.aliceprotocol.org \
  --address YOUR_ADDRESS \
  --instance-id gpu0 \
  > /tmp/miner_gpu0.log 2>&1 &
```

Install managed service mode:

- Linux/macOS: `./miner/install-service.sh`
- Windows: `.\miner\install-service.ps1`

Service manager commands:

- Linux/macOS: `./miner/start-service.sh`, `./miner/stop-service.sh`, `./miner/status-service.sh`, `./miner/uninstall-service.sh`
- Windows: `.\miner\start-service.ps1`, `.\miner\stop-service.ps1`, `.\miner\status-service.ps1`, `.\miner\uninstall-service.ps1`

Stop all miners:

```bash
pkill -f alice_miner.py
```

Stop one specific instance:

```bash
kill "$(pgrep -f 'instance-id gpu0')"
```

## 7. Hardware guidance

- CUDA GPU `>= 24GB`: recommended
- CUDA GPU `16GB`: supported, slower (`batch_size=1`)
- Apple Silicon `16GB`: supported, very slow (macOS swap, `batch_size=1`)
- Apple Silicon `>= 24GB`: supported
- Apple Silicon `>= 32GB`: recommended
- CPU `>= 32GB RAM`: supported but very slow
- `< 16GB`: not supported

CPU mining is supported but not recommended.

For CUDA miners, the client now reports both:

- physical VRAM: the GPU's total memory
- runtime memory cap: any temporary local cap learned from prior OOM recovery

The PS admission check uses physical VRAM so a 24GB card is not misclassified as an unsupported GPU just because a local runtime cap was lowered.

If you are reusing a machine from an older private beta and the miner still behaves like it has a much smaller GPU, inspect:

- `~/.alice/device_profile.json`
- `ALICE_MEMORY_CAP_GB`

Those are local runtime throttles, not the source of truth for physical VRAM.

## 8. Rewards

Rewards are paid to:
- `--reward-address` if provided
- otherwise `--address`

Reward timing depends on successful epoch settlement on chain.

## 9. Epoch reports

Miner writes local epoch reports to:

- `~/.alice/reports/miner_epoch_reports.jsonl`
- `~/.alice/reports/epochs/miner_epoch_<epoch>.md`

Each report records:

- tasks requested and trained
- batches trained
- gradients submitted, accepted, rejected
- average loss
- reward status (`confirmed`, `pending`)

## 10. FAQ

**Can multiple GPUs share the same address?**  
Yes. Rewards from all instances using the same address aggregate automatically to one wallet.

**Do I need to run `bootstrap.sh` for every GPU?**  
No. The first run should prepare the environment and shared model cache. Additional GPU instances on the same machine should usually use `./miner/run_miner.sh`.

**Where is my wallet file?**  
`~/.alice/wallet.json`

**How do I check earnings?**  
Use the Alice explorer or run:

```bash
python3 miner/alice_wallet.py balance --address YOUR_ADDRESS
```
