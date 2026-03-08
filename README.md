# Balancing Coverage and Draft Latency in Vocabulary Trimming for Faster Speculative Decoding

This repository contains the reference implementation for the paper:

> **Balancing Coverage and Draft Latency in Vocabulary Trimming for Faster Speculative Decoding**
Paper link: https://arxiv.org/abs/2603.05210
> 
# Citation

If you find this paper useful or use this code, please cite:
```bibtex
@article{shoham2026balancing,
  title={Balancing Coverage and Draft Latency in Vocabulary Trimming for Faster Speculative Decoding},
  author={Shoham, Ofir Ben},
  journal={arXiv preprint arXiv:2603.05210},
  year={2026}
}
```

## Overview

Speculative decoding accelerates LLM inference by using a lightweight draft model to propose candidate tokens verified in parallel by the target model. The draft model's language modeling (LM) head, which projects hidden states to vocabulary logits, often dominates draft latency — accounting for over 60% of FLOPs for models like LLaMA-3-8B.

We formulate draft vocabulary selection as a **constrained optimization problem** that balances token coverage against draft model latency:

$$k^* = \arg\max_{k \in [k_{\min}, k_{\max}]} U(k) \quad \text{s.t.} \quad C(k) \geq c_{\min}$$

where the utility function combines coverage and latency reduction:

$$U(k) = \alpha \cdot C(k) + (1 - \alpha) \cdot R(k)$$

Token coverage $C(k)$ is computed over **assistant response tokens only** (matching training loss masks), and latency reduction $R(k)$ is estimated using architecture-aware FLOPs. The optimization is performed using Optuna's **Tree-structured Parzen Estimator (TPE)**.

## Installation

```bash
git clone https://github.com/Ofir408/Balancing-Coverage-and-Draft-Latency-in-Vocabulary-Trimming-for-Faster-Speculative-Decoding.git
cd Balancing-Coverage-and-Draft-Latency-in-Vocabulary-Trimming-for-Faster-Speculative-Decoding
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python optimize_vocab_size.py \
    --dataset-path ./data/train.jsonl \
    --tokenizer-path meta-llama/Llama-3.1-8B-Instruct \
    --chat-template llama3 \
    --n-calls 100
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-path` | *required* | Path to training dataset (JSONL with `conversations` field) |
| `--tokenizer-path` | *required* | HuggingFace tokenizer path or name |
| `--chat-template` | `llama3` | Chat template (`llama3`, `llama4`, `qwen`, `deepseek`, `phi3`, `phi4`, `gemma`) |
| `--is-preformatted` | `False` | Set if dataset already contains formatted text |
| `--coverage-weight` | `0.5` | Weight for coverage in utility function ($\alpha$) |
| `--min-coverage` | `0.0` | Minimum acceptable token coverage (0.0 = no constraint) |
| `--min-vocab-size` | `50` | Lower bound of search space |
| `--max-vocab-size` | `50000` | Upper bound of search space |
| `--n-calls` | `30` | Number of TPE optimization trials |
| `--sample-size` | all | Subset of dataset to use |
| `--config-path` | `None` | Path to draft model config JSON to auto-update `draft_vocab_size` |
| `--output-dir` | `./cache/vocab_optimization` | Directory for results and plots |
| `--random-state` | `42` | Seed for reproducibility |

### Dataset Format

The script expects JSONL files with a `conversations` field:

```json
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Or preformatted text (use `--is-preformatted`):

```json
{"text": "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi!<|eot_id|>"}
```

### Output

The script produces:

1. **`summary.json`** — Optimal vocabulary size, coverage, latency reduction, and utility score.
2. **`full_results.json`** — All evaluated configurations with detailed metrics.
3. **`optimization_results.png`** — Four-panel visualization:
   - Coverage vs. vocabulary size
   - Latency reduction vs. vocabulary size
   - Utility score vs. vocabulary size
   - Pareto front (coverage vs. latency)

### Using the Result

After optimization, use the `draft_vocab_size` in your EAGLE-3 draft model training config. For example:

```json
{
    "hidden_size": 4096,
    "vocab_size": 128256,
    "draft_vocab_size": 13264
}
```

For training the draft model with [SpecForge](https://github.com/sgl-project/specforge):

```bash
python scripts/train_eagle3.py --config configs/your_config.json
```

## Reproducing Paper Results

### Open-PerfectBlend

To reproduce the vocabulary optimization that yields a 13,264-token draft vocabulary:

```bash
python optimize_vocab_size.py \
    --dataset-path /path/to/open-perfectblend.jsonl \
    --tokenizer-path meta-llama/Llama-3.1-8B-Instruct \
    --chat-template llama3 \
    --sample-size 50000 \
    --n-calls 100 \
    --random-state 42 \
    --output-dir ./results/open-perfectblend
```



