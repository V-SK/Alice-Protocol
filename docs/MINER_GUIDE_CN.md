# Alice Miner 使用指南

本指南对应当前 `Alice-Protocol` 仓库中的独立矿工版本。

当前状态：
- 仓库仍为 private
- 外部矿工接入仍受限制

## 1. 安装

macOS / Linux：

```bash
./miner/bootstrap.sh
```

Windows：

```powershell
.\miner\bootstrap.ps1
```

## 2. 创建或导入地址

本地创建新钱包：

```bash
python3 miner/alice_wallet.py create
```

如果你已有 Alice 地址，也可以直接在启动时传入 `--address`。

## 3. 启动挖矿

默认 bootstrap 会：

- 创建仓库本地 `.venv`
- 自动安装缺失依赖
- 若本地没有钱包则自动创建
- 默认连接 `https://ps.aliceprotocol.org` 启动 miner

仍可手动启动：

```bash
python3 miner/alice_miner.py \
  --ps-url https://ps.aliceprotocol.org \
  --address a你的Alice地址 \
  --device auto \
  --precision auto
```

如需把奖励发到单独地址：

```bash
python3 miner/alice_miner.py \
  --ps-url https://ps.aliceprotocol.org \
  --address a控制地址 \
  --reward-address a奖励地址
```

## 4. 网络连接

矿工直接连接 Parameter Server：
- `/register`
- `/task/request`
- `/task/complete`
- `/model`
- `/model/info`
- `/model/delta`

矿工不会直接连接 Aggregator。

## 5. 硬件要求

- CUDA GPU `>= 24GB`：推荐
- CUDA GPU `16GB`：支持，但更慢（`batch_size=1`）
- Apple Silicon `16GB`：支持，但会很慢（依赖 macOS swap，`batch_size=1`）
- Apple Silicon `>= 24GB`：支持
- Apple Silicon `>= 32GB`：推荐
- CPU `>= 32GB RAM`：支持，但非常慢
- `< 16GB`：不支持

CPU 挖矿可运行，但通常只有 GPU 的 `1/50 - 1/100` 速度，提交更少，奖励也会显著更低。

## 6. 奖励地址

奖励优先发送到：
- `--reward-address`
- 若未设置，则回到 `--address`

## 7. Epoch 汇报

Miner 会把每个 epoch 的本地汇报写到：

- `~/.alice/reports/miner_epoch_reports.jsonl`
- `~/.alice/reports/epochs/miner_epoch_<epoch>.md`

其中包括训练量、梯度提交数、平均 loss，以及奖励状态（`confirmed` / `pending`）。

## 8. 当前发布说明

本仓库目前仍是 private 的发布整理基线。在完成三平台验证之前，不会开放外部接入。
