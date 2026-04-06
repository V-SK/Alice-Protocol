# Alice Protocol

Decentralized AI training network for Alice-7B. Miners train local shards and earn ALICE in proportion to effective tokens.

## Quick Start

```bash
git clone https://github.com/V-SK/Alice-Protocol.git
cd Alice-Protocol
./miner/bootstrap.sh --ps-url https://ps.aliceprotocol.org
```

The default miner mode is now `Plan B`.

Bootstrap will:

- detect your hardware
- install dependencies
- create or reuse `~/.alice/wallet.json`
- download the current model if needed
- start the miner

To use your own wallet:

```bash
./miner/bootstrap.sh \
  --ps-url https://ps.aliceprotocol.org \
  --address YOUR_ADDRESS
```

## Mining Model

Plan B is the default production path.

- The miner receives shard assignments from the network
- The runtime trains locally and uploads a compressed delta at epoch end
- Rewards are based on effective tokens, not a user-chosen batch size

Plan A is still available as a legacy fallback:

```bash
python3 miner/alice_miner.py --address YOUR_ADDRESS --mode plan_a
```

## Hardware and Batch Assignment

Batch size is assigned by the network based on the miner's hardware profile. A manual `--batch-size` override can only reduce the assigned cap; it cannot increase it.

| GPU VRAM | Assigned Batch Size | Tokens per Shard |
|----------|---------------------|------------------|
| 16GB     | 1                   | 10,240           |
| 24GB     | 4                   | 40,960           |
| 32GB     | 8                   | 81,920           |
| 48GB     | 16                  | 163,840          |
| 80GB     | 32                  | 327,680          |

Other requirements:

- RAM: 16GB+
- Disk: 50GB+
- Stable internet connection

## Rewards

Reward accounting uses the assigned batch size:

```text
effective_tokens = completed_shards * assigned_batch_size
tokens_trained = effective_tokens * 10240
your_share = your_effective_tokens / total_network_effective_tokens
your_reward = epoch_reward * 89% * your_share
```

This means larger GPUs earn proportionally more per shard because they train more tokens per shard.

## Common Commands

Default Plan B:

```bash
python3 miner/alice_miner.py --address YOUR_ADDRESS
```

## Separate Reward Address (Cloud GPU Safe Pattern)

Why:

- Use `--address` as the control/sign-in identity the miner registers with.
- Use `--reward-address` as the payout destination when the miner runs on rented or shared cloud GPUs.
- This keeps the reward wallet separate from the operational wallet you expose on the remote host.

How:

- If `--reward-address` is omitted, rewards go to `--address`.
- If `--reward-address` is set, rewards go to that address while the miner still authenticates as `--address`.
- The safe pattern is to keep `--address` on the cloud worker and direct rewards to a separate cold or payout wallet.

Example:

```bash
python3 miner/alice_miner.py \
  --address YOUR_CONTROL_ADDRESS \
  --reward-address YOUR_PAYOUT_ADDRESS
```

Multi-GPU:

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

Background mode:

```bash
CUDA_VISIBLE_DEVICES=0 nohup ./miner/run_miner.sh \
  --ps-url https://ps.aliceprotocol.org \
  --address YOUR_ADDRESS \
  --instance-id gpu0 \
  > /tmp/miner_gpu0.log 2>&1 &
```

## Links

- Website: https://aliceprotocol.org
- Explorer: https://aliceprotocol.org/explorer
- Wallet CLI: https://github.com/V-SK/alice-wallet

## Notes

- Plan B is the default miner path.
- Plan A is deprecated and should only be used for compatibility or debugging.
- If a local model is too far behind, the miner will refresh the full model instead of staying on stale weights.
