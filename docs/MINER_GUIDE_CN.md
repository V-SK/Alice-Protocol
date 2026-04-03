# Alice Miner 使用指南

本指南对应当前 `Alice-Protocol` 仓库中的独立矿工版本。

## 1. 安装

macOS / Linux：

```bash
./miner/bootstrap.sh --ps-url https://ps.aliceprotocol.org --address YOUR_ADDRESS
```

Windows：

```powershell
.\miner\bootstrap.ps1 -PsUrl https://ps.aliceprotocol.org -Address YOUR_ADDRESS
```

如需长期运行并自动重启：

- Linux/macOS：`./miner/install-service.sh`
- Windows：`.\miner\install-service.ps1`

Linux 的 systemd 安装需要 `sudo`。

## 2. 创建或导入地址

### 使用你自己的地址（推荐）

```bash
./miner/bootstrap.sh \
  --ps-url https://ps.aliceprotocol.org \
  --address a你的Alice地址
```

如需生成新的 Alice 地址，可以运行：

```bash
python3 -c "
from substrateinterface import Keypair
mnemonic = Keypair.generate_mnemonic()
kp = Keypair.create_from_mnemonic(mnemonic, ss58_format=300)
print(f'Address:  {kp.ss58_address}')
print(f'Mnemonic: {mnemonic}')
print()
print('请立即备份助记词。这是唯一恢复方式。')
"
```

### 自动创建钱包

```bash
./miner/bootstrap.sh --ps-url https://ps.aliceprotocol.org
```

如果未传 `--address`，bootstrap 会自动创建或复用：

- `~/.alice/wallet.json`

找回地址和助记词：

```bash
cat ~/.alice/wallet.json
```

请立即备份助记词。丢失后资金无法找回。

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

## 4. 多 GPU 挖矿

Alice 采用“一张 GPU 一个 miner 进程”的方式。当前没有专门的 `--gpu` 参数，正确做法是使用 `CUDA_VISIBLE_DEVICES` 绑定每个进程，再用唯一的 `--instance-id` 区分实例。

推荐流程：

1. 先运行一次 `./miner/bootstrap.sh`，准备 `.venv`、默认钱包和共享模型缓存
2. 同机后续 GPU 实例优先使用 `./miner/run_miner.sh`

### 2 张 GPU

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

### 4 张 GPU

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

### 8 张 GPU

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

要点：

- 多个实例可以共用同一个地址，奖励会自动聚合
- 每个实例必须使用不同的 `--instance-id`
- 后续实例会复用模型缓存，但如果你再次运行 `bootstrap.sh`，它仍会重新检查 Python 依赖

## 5. 网络连接

矿工直接连接 Parameter Server：
- `/register`
- `/task/request`
- `/task/complete`
- `/model`
- `/model/info`
- `/model/delta`

矿工不会直接连接 Aggregator。

## 6. 后台运行 / Service 模式

Linux 后台运行示例：

```bash
CUDA_VISIBLE_DEVICES=0 nohup ./miner/run_miner.sh \
  --ps-url https://ps.aliceprotocol.org \
  --address YOUR_ADDRESS \
  --instance-id gpu0 \
  > /tmp/miner_gpu0.log 2>&1 &
```

安装 service：

- Linux/macOS：`./miner/install-service.sh`
- Windows：`.\miner\install-service.ps1`

service 管理命令：

- Linux/macOS：`./miner/start-service.sh`、`./miner/stop-service.sh`、`./miner/status-service.sh`、`./miner/uninstall-service.sh`
- Windows：`.\miner\start-service.ps1`、`.\miner\stop-service.ps1`、`.\miner\status-service.ps1`、`.\miner\uninstall-service.ps1`

停止全部 miner：

```bash
pkill -f alice_miner.py
```

停止单个实例：

```bash
kill "$(pgrep -f 'instance-id gpu0')"
```

## 7. 硬件要求

- CUDA GPU `>= 24GB`：推荐
- CUDA GPU `16GB`：支持，但更慢（`batch_size=1`）
- Apple Silicon `16GB`：支持，但会很慢（依赖 macOS swap，`batch_size=1`）
- Apple Silicon `>= 24GB`：支持
- Apple Silicon `>= 32GB`：推荐
- CPU `>= 32GB RAM`：支持，但非常慢
- `< 16GB`：不支持

CPU 挖矿可运行，但通常只有 GPU 的 `1/50 - 1/100` 速度，提交更少，奖励也会显著更低。

## 8. 奖励地址

奖励优先发送到：
- `--reward-address`
- 若未设置，则回到 `--address`

## 9. Epoch 汇报

Miner 会把每个 epoch 的本地汇报写到：

- `~/.alice/reports/miner_epoch_reports.jsonl`
- `~/.alice/reports/epochs/miner_epoch_<epoch>.md`

其中包括训练量、梯度提交数、平均 loss，以及奖励状态（`confirmed` / `pending`）。

## 10. 常见问题

**多张 GPU 能不能共用一个地址？**  
可以。多个实例使用同一个地址时，奖励会自动聚合到同一个钱包。

**每张 GPU 都要重新跑 `bootstrap.sh` 吗？**  
不需要。首次运行建议用 `bootstrap.sh` 完成环境和模型缓存准备，后续实例优先用 `./miner/run_miner.sh`。

**钱包文件在哪里？**  
`~/.alice/wallet.json`

**怎么查看收益？**  
可以在 Alice explorer 里搜索地址，或运行：

```bash
python3 miner/alice_wallet.py balance --address YOUR_ADDRESS
```
