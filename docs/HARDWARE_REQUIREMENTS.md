# Hardware Requirements

## Miner

| Profile | Requirement | Support Level | Expected Performance |
|---|---|---|---|
| CUDA GPU | `>= 24GB VRAM` | Recommended | Best performance |
| CUDA GPU | `16GB VRAM` | Supported, slower | `batch_size=1`, lower throughput |
| Apple Silicon | `>= 32GB unified memory` | Recommended | Good performance |
| Apple Silicon | `24GB unified memory` | Supported | Reduced throughput |
| Apple Silicon | `16GB unified memory` | Supported, very slow | Uses macOS swap, `batch_size=1` |
| CPU | `>= 32GB RAM` | Supported but not recommended | Roughly `1/50 - 1/100` of GPU throughput |
| Low memory | `< 16GB VRAM / RAM` | Not supported | Likely unstable or unusable |

### CPU mining

CPU mining is supported but not recommended. Training speed is typically `50-100x` slower than GPU, which means:
- far fewer gradient submissions per epoch
- proportionally lower rewards
- much longer warm-up and model load time

Minimum recommendation for CPU mining: `32GB RAM`.

### Apple Silicon notes

- `16GB` Apple Silicon is supported, but macOS will rely on swap and training will be very slow.
- `24GB+` Apple Silicon is supported.
- `32GB+` Apple Silicon is recommended.

## Scorer

| Platform | Requirement | Default Dtype | Support Level |
|---|---|---|---|
| Linux x86 | `>= 32GB RAM` | `float32` | Recommended |
| Linux x86 | `24-32GB RAM` | `float16` | Supported, slower |
| Mac ARM | `>= 24GB unified memory` | `float16` | Supported |
| Windows | `>= 32GB RAM` | `float32` | Experimental |

### Scorer notes

- x86 CPU with `float32` is the preferred path, but still slower than GPU-style workloads.
- x86 CPU with `float16` is usually slower and should only be used as a low-memory fallback.
- Mac ARM defaults to `float16`.
- Windows scorer has no known architectural blocker, but remains experimental until full validation is complete.

## Storage and Network

- Miner model download: expect a large initial model checkpoint download
- Scorer model checkpoint: expect tens of GB of disk headroom
- Stable outbound HTTPS access to the PS is required
- Scorer endpoints must be reachable by the aggregator
- Miner and scorer both write local epoch reports to `~/.alice/reports/`
