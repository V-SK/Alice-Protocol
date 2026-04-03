# Alice Protocol

Decentralized AI training network. Train a 7B language model from scratch using distributed GPU compute.

## Quick Start

### One command to start mining:

```bash
git clone https://github.com/V-SK/Alice-Protocol.git
cd Alice-Protocol
./miner/bootstrap.sh --ps-url https://ps.aliceprotocol.org
```

That's it. The script will:
- Detect your GPU (CUDA / MPS / CPU)
- Install dependencies
- Create a wallet automatically
- Download the model (~14GB)
- Start training

### Check your wallet:

```bash
cat ~/.alice/wallet.json
```

This file contains your `address` and `mnemonic`. **Write down the mnemonic and keep it safe — it's the only way to recover your wallet.**

### Check your balance:

```bash
python3 miner/alice_wallet.py balance --address YOUR_ADDRESS
```

Or visit the explorer: https://aliceprotocol.org

---

## Mine with your own address

If you already have an Alice wallet:

```bash
./miner/bootstrap.sh \
  --ps-url https://ps.aliceprotocol.org \
  --address YOUR_ADDRESS
```

Generate a new address manually:

```bash
python3 -c "
from substrateinterface import Keypair
mnemonic = Keypair.generate_mnemonic()
kp = Keypair.create_from_mnemonic(mnemonic, ss58_format=300)
print(f'Address:  {kp.ss58_address}')
print(f'Mnemonic: {mnemonic}')
print()
print('Write down the mnemonic. This is your only backup.')
"
```

---

## Multi-GPU Mining

Run one miner instance per GPU:

```bash
# 2 GPUs
CUDA_VISIBLE_DEVICES=0 ./miner/bootstrap.sh --ps-url https://ps.aliceprotocol.org --address YOUR_ADDRESS --instance-id gpu0 &
CUDA_VISIBLE_DEVICES=1 ./miner/bootstrap.sh --ps-url https://ps.aliceprotocol.org --address YOUR_ADDRESS --instance-id gpu1 &
```

```bash
# 4 GPUs
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i ./miner/bootstrap.sh \
    --ps-url https://ps.aliceprotocol.org \
    --address YOUR_ADDRESS \
    --instance-id gpu${i} &
  sleep 5
done
```

```bash
# 8 GPUs
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i ./miner/bootstrap.sh \
    --ps-url https://ps.aliceprotocol.org \
    --address YOUR_ADDRESS \
    --instance-id gpu${i} &
  sleep 5
done
```

- All instances can use the **same address** — rewards combine automatically
- Each instance **must** have a unique `--instance-id`
- Verify with `nvidia-smi` — you should see one python process per GPU

---

## Hardware Requirements

| GPU VRAM | Batch Size | Effective Tokens/shard |
|----------|-----------|----------------------|
| 80GB (A100/H100) | 32 | 32 |
| 48GB (A6000) | 16 | 16 |
| 32GB (V100-32G) | 8 | 8 |
| 24GB (RTX 3090/4090) | 4 | 4 |
| 16GB (RTX 4060 Ti) | 1 | 1 |
| <16GB | — | Not supported |

Rewards scale with effective tokens (shards × batch_size). Larger GPUs earn proportionally more per shard.

- **RAM**: 16GB+ system memory
- **Disk**: 50GB+ free space
- **Network**: Stable internet connection

### Apple Silicon (MPS)

```bash
./miner/bootstrap.sh --ps-url https://ps.aliceprotocol.org --device mps
```

Requires 16GB+ unified memory.

### Windows

```powershell
.\miner\bootstrap.ps1
```

---

## Wallet

Standalone wallet CLI:

```bash
git clone https://github.com/V-SK/alice-wallet.git
cd alice-wallet
pip install -r requirements.txt
python3 cli.py --help
```

GitHub: https://github.com/V-SK/alice-wallet

---

## Run as Service

```bash
# Linux/macOS
./miner/install-service.sh

# Windows
.\miner\install-service.ps1
```

---

## Run a Scorer

Scorers validate miner gradients and earn 6% of epoch rewards.

```bash
./scorer/bootstrap.sh \
  --scorer-address YOUR_STAKED_ADDRESS \
  --public-endpoint http://YOUR_PUBLIC_IP:8090
```

Requires 5,000 ALICE staked on-chain. See `docs/SCORER_GUIDE.md`.

---

## Documentation

- [Miner Guide](docs/MINER_GUIDE.md)
- [Miner Guide (中文)](docs/MINER_GUIDE_CN.md)
- [Scorer Guide](docs/SCORER_GUIDE.md)
- [Hardware Requirements](docs/HARDWARE_REQUIREMENTS.md)

## Links

- Website: https://aliceprotocol.org
- Explorer: https://aliceprotocol.org
- Discord: https://discord.gg/mRz3BThDyh
- Twitter: https://x.com/Alice_AI102
- Wallet: https://github.com/V-SK/alice-wallet

## FAQ

**Q: What is Score: N/A?**
Scoring is asynchronous. Your gradient was accepted and will be scored in the background. N/A means "not yet scored at time of submission." Your rewards are based on effective tokens, not individual scores.

**Q: Can multiple GPUs share the same address?**
Yes. Use different `--instance-id` for each GPU. Rewards combine automatically.

**Q: How are rewards calculated?**
Rewards = your effective tokens / total network effective tokens × miner pool (89% of epoch reward). Effective tokens = shards completed × batch_size assigned by your GPU's VRAM.

**Q: What are we training?**
Alice-7B, a 7 billion parameter language model trained from scratch on SlimPajama. No corporate pretrained weights — every parameter is computed by miners.

**Q: My miner shows errors after a server restart?**
Run `git pull origin main` and restart. The latest version has auto-recovery for session expiry.

## License

MIT
