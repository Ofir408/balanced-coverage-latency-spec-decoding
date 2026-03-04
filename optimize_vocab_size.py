"""
Vocabulary Size Optimization for Speculative Decoding Draft Models

Finds the optimal draft vocabulary size that balances token coverage
and draft model latency for speculative decoding.

Coverage is computed from tokens that contribute to training loss (loss_mask=1),
i.e., only assistant response tokens, aligning with standard instruction tuning.

This is the reference implementation for:
  "Balancing Coverage and Draft Latency in Vocabulary Trimming
   for Faster Speculative Decoding" (Ben Shoham, 2025)

Usage:
    python optimize_vocab_size.py \
        --dataset-path ./data/train.jsonl \
        --tokenizer-path meta-llama/Llama-3.1-8B-Instruct \
        --chat-template llama3 \
        --n-calls 100

Output:
    The optimal draft_vocab_size to use in your config file, e.g.:
    {
        ...
        "vocab_size": 128256,
        "draft_vocab_size": <optimal_value>
    }
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError(
        "optuna is required. Install with: pip install optuna"
    )


# ============================================================================
# Minimal chat template handling (standalone, no specforge imports)
# ============================================================================

CHAT_TEMPLATES = {
    "llama3": {
        "assistant_header": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "user_header": "<|start_header_id|>user<|end_header_id|>",
        "end_of_turn_token": "<|eot_id|>",
    },
    "llama4": {
        "assistant_header": "<|header_start|>assistant<|header_end|>\n\n",
        "user_header": "<|header_start|>user<|header_end|>",
        "end_of_turn_token": "<|eot|>",
    },
    "qwen": {
        "assistant_header": "<|im_start|>assistant\n",
        "user_header": "<|im_start|>user\n",
        "end_of_turn_token": "<|im_end|>\n",
    },
    "deepseek": {
        "assistant_header": "Assistant:",
        "user_header": "User:",
        "end_of_turn_token": "",
    },
    "phi3": {
        "assistant_header": "<|assistant|>\n",
        "user_header": "<|user|>\n",
        "end_of_turn_token": "<|end|>\n",
    },
    "phi4": {
        "assistant_header": "<|im_start|>assistant<|im_sep|>",
        "user_header": "<|im_start|>user<|im_sep|>",
        "end_of_turn_token": "<|im_end|>",
    },
    "gemma": {
        "assistant_header": "<start_of_turn>model\n",
        "user_header": "<start_of_turn>user\n",
        "end_of_turn_token": "<end_of_turn>\n",
    },
}


def get_available_templates() -> List[str]:
    return list(CHAT_TEMPLATES.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find optimal draft_vocab_size for EAGLE3 training"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the training dataset (jsonl format with 'conversations' field)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path or name of the target model tokenizer",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="llama3",
        help=f"Chat template for tokenization",
    )
    parser.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the dataset contains preformatted text",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=131072,
        help="Max sequence length for tokenization (default: 131072 to capture all assistant tokens)",
    )
    parser.add_argument(
        "--target-vocab-size",
        type=int,
        default=None,
        help="Target model vocabulary size (auto-detected from tokenizer if not specified)",
    )
    parser.add_argument(
        "--min-vocab-size",
        type=int,
        default=50,
        help="Minimum draft vocabulary size to consider (default: 50)",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=50000,
        help="Maximum draft vocabulary size to consider (default: 50000)",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=30,
        help="Number of optimization iterations (default: 30)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="Minimum acceptable token coverage (default: 0.0, no constraint)",
    )
    parser.add_argument(
        "--coverage-weight",
        type=float,
        default=0.5,
        help="Weight for coverage in utility (default: 0.5)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to use from dataset (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./cache/vocab_optimization",
        help="Directory to save optimization results",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to EAGLE3 config file to update with optimal draft_vocab_size",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of processes for dataset preprocessing (default: 8)",
    )
    return parser.parse_args()


def load_model_config(config_path: str) -> dict:
    """Load model architecture parameters from config file."""
    with open(config_path, "r") as f:
        return json.load(f)


def compute_loss_mask_for_text(
    text: str,
    tokenizer,
    chat_template: str,
    max_length: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize text and compute loss mask (1 for assistant tokens, 0 otherwise).
    
    For preformatted text, identifies assistant response spans using the chat template.
    """
    template = CHAT_TEMPLATES.get(chat_template, CHAT_TEMPLATES["llama3"])
    
    # Tokenize with offset mapping
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    
    input_ids = encoding.input_ids[0]
    offsets = encoding.offset_mapping[0]
    
    # Create loss mask - mark assistant response tokens
    loss_mask = torch.zeros(len(input_ids), dtype=torch.long)
    
    assistant_header = template["assistant_header"]
    user_header = template["user_header"]
    end_of_turn = template["end_of_turn_token"]
    
    # Build regex pattern to find assistant responses
    user_sep = re.escape(end_of_turn) + re.escape(user_header) if user_header else re.escape(end_of_turn)
    assistant_sep = re.escape(end_of_turn) + re.escape(assistant_header) if assistant_header else ""
    
    # Find assistant response spans
    if assistant_header:
        pattern = re.escape(assistant_header) + r"(.*?)(?=" + user_sep + "|$)"
        
        for match in re.finditer(pattern, text, re.DOTALL):
            start_char = match.start(1)
            end_char = match.end(1)
            
            # Mark tokens in this span
            for idx, (token_start, token_end) in enumerate(offsets.tolist()):
                if token_end <= start_char:
                    continue
                if token_start >= end_char:
                    continue
                loss_mask[idx] = 1
    else:
        # If no template, count all tokens
        loss_mask.fill_(1)
    
    return input_ids, loss_mask


def format_conversation_with_template(
    conversations: List[dict],
    chat_template: str,
) -> str:
    """
    Format structured conversations using the chat template.
    
    Converts conversations like:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    To formatted text with proper headers.
    """
    template = CHAT_TEMPLATES.get(chat_template, CHAT_TEMPLATES["llama3"])
    
    formatted_parts = []
    for msg in conversations:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        
        if role == "user":
            formatted_parts.append(f"{template['user_header']}{content}{template['end_of_turn_token']}")
        elif role == "assistant":
            formatted_parts.append(f"{template['assistant_header']}{content}{template['end_of_turn_token']}")
        elif role == "system":
            # Handle system messages (prepend to first user message or add separately)
            formatted_parts.append(content)
    
    return "".join(formatted_parts)


def compute_token_frequencies(
    dataset_path: str,
    tokenizer,
    chat_template: str,
    max_length: int = 131072,
    sample_size: Optional[int] = None,
    is_preformatted: bool = False,
    num_proc: int = 8,
) -> Counter:
    """
    Compute token frequencies from training data.
    
    IMPORTANT: Only counts tokens where loss_mask=1 (assistant responses),
    matching the behavior of generate_vocab_mapping_file in train_eagle3.py.
    """
    # Load dataset
    dataset = hf_load_dataset("json", data_files=dataset_path)["train"]
    
    if sample_size is not None and sample_size < len(dataset):
        dataset = dataset.select(range(sample_size))
        print(f"Using {sample_size} samples from dataset")
    
    token_frequencies = Counter()
    
    # Always count only loss_mask=1 tokens (assistant responses)
    # This matches the training script behavior
    print("Computing token frequencies (loss_mask=1 tokens only, matching training)...")
    
    for item in tqdm(dataset, desc="Processing"):
        # Get text
        if is_preformatted:
            if "text" in item:
                text = item["text"]
            elif "conversations" in item:
                # For preformatted, conversations might contain the full text
                text = " ".join(msg.get("content", "") for msg in item["conversations"])
            else:
                continue
        else:
            # For structured conversations, apply chat template to get proper formatting
            if "conversations" not in item:
                continue
            conversations = item["conversations"]
            if not conversations:
                continue
            # Format with chat template so we can find assistant headers
            text = format_conversation_with_template(conversations, chat_template)
        
        if not text.strip():
            continue
        
        # Compute loss mask for assistant tokens (same logic as training)
        input_ids, loss_mask = compute_loss_mask_for_text(
            text, tokenizer, chat_template, max_length
        )
        
        # Count only tokens where loss_mask=1 (assistant responses)
        masked_ids = input_ids[loss_mask == 1]
        if len(masked_ids) > 0:
            unique_ids, counts = masked_ids.unique(return_counts=True)
            batch_token_dict = dict(zip(unique_ids.tolist(), counts.tolist()))
            token_frequencies.update(batch_token_dict)
    
    return token_frequencies


def compute_coverage(token_frequencies: Counter, draft_vocab_size: int) -> float:
    """
    Compute token coverage for a given draft vocabulary size.
    
    Coverage = (sum of frequencies of top-k tokens) / (total frequency)
    
    This is the fraction of training tokens covered by the draft vocabulary.
    """
    total_frequency = sum(token_frequencies.values())
    if total_frequency == 0:
        return 0.0

    top_k = token_frequencies.most_common(draft_vocab_size)
    top_k_frequency = sum(freq for _, freq in top_k)

    return top_k_frequency / total_frequency


def estimate_lm_head_flops(hidden_size: int, vocab_size: int) -> float:
    """Estimate FLOPs for LM head: Linear(hidden_size -> vocab_size)."""
    return 2 * hidden_size * vocab_size


def estimate_eagle3_draft_model_flops(
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    draft_vocab_size: int,
    num_aux_hidden_layers: int = 3,
) -> dict:
    """
    Estimate FLOPs for EAGLE3 draft model forward pass (per token).
    
    EAGLE3 draft model architecture:
    1. FC projection: concat(aux_hidden_states) -> hidden_size
    2. One decoder layer (attention + MLP)
    3. LM head: hidden_size -> draft_vocab_size
    
    Returns dict with breakdown and total FLOPs.
    """
    head_dim = hidden_size // num_attention_heads
    
    # FC projection: (hidden_size * num_aux_layers) -> hidden_size
    fc_flops = 2 * (hidden_size * num_aux_hidden_layers) * hidden_size
    
    # Attention projections (note: EAGLE3 uses hidden_size*2 input for Q/K/V)
    # Q: (hidden_size*2) -> (num_heads * head_dim)
    # K: (hidden_size*2) -> (num_kv_heads * head_dim)
    # V: (hidden_size*2) -> (num_kv_heads * head_dim)
    # O: (num_heads * head_dim) -> hidden_size
    qkv_input_size = hidden_size * 2
    q_flops = 2 * qkv_input_size * (num_attention_heads * head_dim)
    k_flops = 2 * qkv_input_size * (num_key_value_heads * head_dim)
    v_flops = 2 * qkv_input_size * (num_key_value_heads * head_dim)
    o_flops = 2 * (num_attention_heads * head_dim) * hidden_size
    attn_proj_flops = q_flops + k_flops + v_flops + o_flops
    
    # MLP: gate_proj, up_proj (hidden -> intermediate), down_proj (intermediate -> hidden)
    mlp_flops = 2 * hidden_size * intermediate_size * 3  # 3 projections
    
    # LM head: hidden_size -> draft_vocab_size
    lm_head_flops = estimate_lm_head_flops(hidden_size, draft_vocab_size)
    
    total_flops = fc_flops + attn_proj_flops + mlp_flops + lm_head_flops
    
    return {
        "fc": fc_flops,
        "attention": attn_proj_flops,
        "mlp": mlp_flops,
        "lm_head": lm_head_flops,
        "total": total_flops,
        "lm_head_ratio": lm_head_flops / total_flops,
    }


def estimate_latency_reduction(
    draft_vocab_size: int,
    target_vocab_size: int,
    hidden_size: int = 4096,
    intermediate_size: int = 14336,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 8,
) -> float:
    """
    Estimate relative latency reduction from vocabulary trimming for EAGLE3.
    
    This accounts for the full draft model architecture:
    - FC projection layer
    - 1 decoder layer (attention + MLP)
    - LM head
    
    Returns value between 0 (no reduction) and ~0.5 (typical max for EAGLE3).
    """
    # Compute FLOPs with trimmed vocab
    flops_trimmed = estimate_eagle3_draft_model_flops(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        draft_vocab_size=draft_vocab_size,
    )
    
    # Compute FLOPs with full vocab
    flops_full = estimate_eagle3_draft_model_flops(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        draft_vocab_size=target_vocab_size,
    )
    
    # Latency reduction = 1 - (trimmed_flops / full_flops)
    return 1.0 - (flops_trimmed["total"] / flops_full["total"])


def utility_function(
    coverage: float,
    latency_reduction: float,
    coverage_weight: float = 0.7,
) -> float:
    """Compute utility combining coverage and latency reduction."""
    latency_weight = 1.0 - coverage_weight
    return coverage_weight * coverage + latency_weight * latency_reduction


class VocabOptimizer:
    """Vocabulary size optimizer for EAGLE3 draft models using Optuna."""

    def __init__(
        self,
        token_frequencies: Counter,
        target_vocab_size: int,
        min_vocab_size: int = 1000,
        max_vocab_size: int = 50000,
        min_coverage: float = 0.95,
        coverage_weight: float = 0.7,
        # Model architecture params for accurate latency estimation
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
    ):
        self.token_frequencies = token_frequencies
        self.target_vocab_size = target_vocab_size
        self.min_vocab_size = min_vocab_size
        self.max_vocab_size = min(max_vocab_size, target_vocab_size)
        self.min_coverage = min_coverage
        self.coverage_weight = coverage_weight
        
        # Model architecture for accurate FLOPs estimation
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        
        self.evaluation_cache: Dict[int, Tuple[float, float, float]] = {}
        
        # Print model info for latency estimation
        flops_full = estimate_eagle3_draft_model_flops(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            draft_vocab_size=target_vocab_size,
        )
        print(f"\nEAGLE3 Draft Model FLOPs breakdown (full vocab={target_vocab_size:,}):")
        print(f"  FC projection:  {flops_full['fc']/1e6:,.1f}M ({flops_full['fc']/flops_full['total']*100:.1f}%)")
        print(f"  Attention:      {flops_full['attention']/1e6:,.1f}M ({flops_full['attention']/flops_full['total']*100:.1f}%)")
        print(f"  MLP:            {flops_full['mlp']/1e6:,.1f}M ({flops_full['mlp']/flops_full['total']*100:.1f}%)")
        print(f"  LM Head:        {flops_full['lm_head']/1e6:,.1f}M ({flops_full['lm_head']/flops_full['total']*100:.1f}%)")
        print(f"  Total:          {flops_full['total']/1e6:,.1f}M")

    def evaluate(self, vocab_size: int) -> Tuple[float, float, float]:
        """Evaluate a vocabulary size. Returns (coverage, latency_reduction, utility)."""
        if vocab_size in self.evaluation_cache:
            return self.evaluation_cache[vocab_size]

        coverage = compute_coverage(self.token_frequencies, vocab_size)
        latency_reduction = estimate_latency_reduction(
            draft_vocab_size=vocab_size,
            target_vocab_size=self.target_vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
        )
        utility = utility_function(coverage, latency_reduction, self.coverage_weight)

        self.evaluation_cache[vocab_size] = (coverage, latency_reduction, utility)
        return coverage, latency_reduction, utility

    def objective(self, trial: "optuna.Trial") -> float:
        """
        Optuna objective function to maximize utility.
        
        Returns very low utility (-1.0) when coverage is below min_coverage,
        ensuring Optuna learns to avoid those regions.
        """
        vocab_size = trial.suggest_int("vocab_size", self.min_vocab_size, self.max_vocab_size)
        coverage, latency_reduction, utility = self.evaluate(vocab_size)
        
        # Report intermediate values for pruning
        trial.set_user_attr("coverage", coverage)
        trial.set_user_attr("latency_reduction", latency_reduction)
        
        # Return very low utility if coverage constraint is not met
        if coverage < self.min_coverage:
            return -1.0  # Low value to discourage this region
        
        return utility

    def optimize(self, n_trials: int = 100, random_state: int = 42) -> Dict:
        """
        Find optimal vocabulary size using Optuna.
        
        Uses TPE (Tree-structured Parzen Estimator) sampler which is efficient
        for hyperparameter optimization and learns from previous trials.
        """

        print(f"\n{'='*60}")
        print("Running Optuna Optimization")
        print(f"{'='*60}")
        print(f"Search space: [{self.min_vocab_size:,}, {self.max_vocab_size:,}]")
        print(f"Minimum coverage constraint: {self.min_coverage:.2%}")
        print(f"Number of trials: {n_trials}")

        # Suppress Optuna's verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Create study with TPE sampler
        sampler = TPESampler(seed=random_state)
        study = optuna.create_study(
            direction="maximize",  # Maximize utility
            sampler=sampler,
        )
        
        # Run optimization with progress bar
        with tqdm(total=n_trials, desc="Optuna trials") as pbar:
            def callback(study, trial):
                pbar.update(1)
                if trial.value is not None and trial.value > -1.0:
                    pbar.set_postfix({
                        "best": f"{study.best_value:.4f}",
                        "vocab": study.best_params.get("vocab_size", "?"),
                    })
            
            study.optimize(
                self.objective,
                n_trials=n_trials,
                callbacks=[callback],
                show_progress_bar=False,
            )

        # Get best result
        best_trial = study.best_trial
        optimal_vocab_size = best_trial.params["vocab_size"]
        coverage, latency_reduction, utility = self.evaluate(optimal_vocab_size)
        
        # If best result doesn't meet coverage, find smallest vocab that does
        if coverage < self.min_coverage:
            print(f"\nWarning: Best trial doesn't meet min_coverage ({coverage:.2%} < {self.min_coverage:.2%})")
            print("Searching for smallest vocab size meeting coverage constraint...")
            
            # Find all evaluated vocab sizes that meet coverage
            valid = [(vs, c, lr, u) for vs, (c, lr, u) in self.evaluation_cache.items() 
                     if c >= self.min_coverage]
            
            if valid:
                # Sort by utility descending, pick best
                valid.sort(key=lambda x: x[3], reverse=True)
                optimal_vocab_size, coverage, latency_reduction, utility = valid[0]
                print(f"Found: vocab_size={optimal_vocab_size:,} with coverage={coverage:.2%}")
            else:
                print("No vocab size in cache meets coverage. Using max vocab size.")
                optimal_vocab_size = self.max_vocab_size
                coverage, latency_reduction, utility = self.evaluate(optimal_vocab_size)

        print(f"\n✓ Optimization complete!")
        print(f"  Best vocab_size: {optimal_vocab_size:,}")
        print(f"  Coverage: {coverage:.2%}")
        print(f"  Utility: {utility:.4f}")

        return {
            "method": "optuna",
            "optimal_vocab_size": int(optimal_vocab_size),
            "coverage": coverage,
            "latency_reduction": latency_reduction,
            "utility": utility,
            "n_trials": n_trials,
            "all_evaluations": self.evaluation_cache.copy(),
        }


def plot_results(result: Dict, output_dir: str, target_vocab_size: int):
    """Generate and save visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    evaluations = result["all_evaluations"]
    vocab_sizes = sorted(evaluations.keys())
    coverages = [evaluations[v][0] for v in vocab_sizes]
    latency_reductions = [evaluations[v][1] for v in vocab_sizes]
    utilities = [evaluations[v][2] for v in vocab_sizes]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Coverage vs Vocab Size
    ax1 = axes[0, 0]
    ax1.plot(vocab_sizes, coverages, "b-", linewidth=2, label="Coverage")
    ax1.axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
    ax1.axvline(
        x=result["optimal_vocab_size"],
        color="g",
        linestyle="--",
        label=f"Optimal: {result['optimal_vocab_size']:,}",
    )
    ax1.set_xlabel("Draft Vocabulary Size")
    ax1.set_ylabel("Token Coverage")
    ax1.set_title("Token Coverage vs Draft Vocab Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Latency Reduction vs Vocab Size
    ax2 = axes[0, 1]
    ax2.plot(vocab_sizes, latency_reductions, "orange", linewidth=2)
    ax2.axvline(
        x=result["optimal_vocab_size"],
        color="g",
        linestyle="--",
        label=f"Optimal: {result['optimal_vocab_size']:,}",
    )
    ax2.set_xlabel("Draft Vocabulary Size")
    ax2.set_ylabel("Latency Reduction (relative)")
    ax2.set_title("Latency Reduction vs Draft Vocab Size")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Utility vs Vocab Size
    ax3 = axes[1, 0]
    ax3.plot(vocab_sizes, utilities, "purple", linewidth=2)
    ax3.axvline(
        x=result["optimal_vocab_size"],
        color="g",
        linestyle="--",
        label=f"Optimal: {result['optimal_vocab_size']:,}",
    )
    ax3.set_xlabel("Draft Vocabulary Size")
    ax3.set_ylabel("Utility Score")
    ax3.set_title("Utility Score vs Draft Vocab Size")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Pareto Front
    ax4 = axes[1, 1]
    scatter = ax4.scatter(
        latency_reductions, coverages, c=vocab_sizes, cmap="viridis", alpha=0.7
    )
    optimal_idx = vocab_sizes.index(result["optimal_vocab_size"])
    ax4.scatter(
        latency_reductions[optimal_idx],
        coverages[optimal_idx],
        color="red",
        s=200,
        marker="*",
        label=f"Optimal: {result['optimal_vocab_size']:,}",
        zorder=5,
    )
    ax4.set_xlabel("Latency Reduction")
    ax4.set_ylabel("Token Coverage")
    ax4.set_title("Pareto Front: Coverage vs Latency")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label="Draft Vocab Size")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_results.png"), dpi=150)
    plt.close()

    print(f"\nPlots saved to {output_dir}/optimization_results.png")


def save_results(result: Dict, output_dir: str, target_vocab_size: int):
    """Save optimization results to files."""
    os.makedirs(output_dir, exist_ok=True)

    result_to_save = {k: v for k, v in result.items() if k != "skopt_result"}
    if "all_evaluations" in result_to_save:
        result_to_save["all_evaluations"] = {
            str(k): {"coverage": v[0], "latency_reduction": v[1], "utility": v[2]}
            for k, v in result_to_save["all_evaluations"].items()
        }

    # Convert numpy types to native Python types for JSON serialization
    def to_python(val):
        if hasattr(val, "item"):
            return val.item()
        if isinstance(val, dict):
            return {k: to_python(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [to_python(v) for v in val]
        return val

    summary = {
        "method": result["method"],
        "optimal_draft_vocab_size": to_python(result["optimal_vocab_size"]),
        "target_vocab_size": to_python(target_vocab_size),
        "vocab_reduction_ratio": to_python(1 - result["optimal_vocab_size"] / target_vocab_size),
        "coverage": to_python(result["coverage"]),
        "latency_reduction": to_python(result["latency_reduction"]),
        "utility": to_python(result["utility"]),
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "full_results.json"), "w") as f:
        json.dump(to_python(result_to_save), f, indent=2)

    print(f"\nResults saved to {output_dir}/")


def update_config_file(config_path: str, optimal_vocab_size: int):
    """Update the draft_vocab_size in the config file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Convert numpy int64 to Python int if needed
    if hasattr(optimal_vocab_size, "item"):
        optimal_vocab_size = optimal_vocab_size.item()
    
    old_value = config.get("draft_vocab_size", "not set")
    config["draft_vocab_size"] = int(optimal_vocab_size)
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Updated {config_path}")
    print(f"   draft_vocab_size: {old_value} -> {optimal_vocab_size}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model architecture from config file (required for accurate latency estimation)
    model_config = {}
    if args.config_path and os.path.exists(args.config_path):
        print(f"Loading draft model config from {args.config_path}...")
        model_config = load_model_config(args.config_path)
    
    # Get model architecture parameters from config (with sensible defaults)
    hidden_size = model_config.get("hidden_size", 4096)
    intermediate_size = model_config.get("intermediate_size", 14336)
    num_attention_heads = model_config.get("num_attention_heads", 32)
    num_key_value_heads = model_config.get("num_key_value_heads", 8)
    
    print(f"\nModel architecture (for latency estimation):")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  num_attention_heads: {num_attention_heads}")
    print(f"  num_key_value_heads: {num_key_value_heads}")

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Determine target vocab size (priority: CLI > config > tokenizer)
    target_vocab_size = args.target_vocab_size or model_config.get("vocab_size") or len(tokenizer)
    print(f"Target vocabulary size: {target_vocab_size:,}")

    max_vocab_size = min(args.max_vocab_size, target_vocab_size)

    # Compute token frequencies from training data
    print(f"\nAnalyzing training data: {args.dataset_path}")
    token_frequencies = compute_token_frequencies(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        sample_size=args.sample_size,
        is_preformatted=args.is_preformatted,
        num_proc=args.num_proc,
    )
    
    total_tokens = sum(token_frequencies.values())
    unique_tokens = len(token_frequencies)
    print(f"\nToken statistics (loss_mask=1 only):")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Unique tokens: {unique_tokens:,}")

    # Create optimizer with model architecture for accurate latency estimation
    optimizer = VocabOptimizer(
        token_frequencies=token_frequencies,
        target_vocab_size=target_vocab_size,
        min_vocab_size=args.min_vocab_size,
        max_vocab_size=max_vocab_size,
        min_coverage=args.min_coverage,
        coverage_weight=args.coverage_weight,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )

    # Run optimization with Optuna
    result = optimizer.optimize(
        n_trials=args.n_calls,
        random_state=args.random_state,
    )

    # Compute FLOPs for optimal vocab size
    flops_optimal = estimate_eagle3_draft_model_flops(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        draft_vocab_size=result["optimal_vocab_size"],
    )
    flops_full = estimate_eagle3_draft_model_flops(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        draft_vocab_size=target_vocab_size,
    )

    # Print results
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Method: {result['method']}")
    print(f"Optimal draft_vocab_size: {result['optimal_vocab_size']:,}")
    print(f"Token coverage: {result['coverage']:.4f} ({result['coverage']:.2%})")
    print(f"\nLatency analysis:")
    print(f"  Vocab reduction:    {target_vocab_size:,} -> {result['optimal_vocab_size']:,} ({(1 - result['optimal_vocab_size']/target_vocab_size):.1%})")
    print(f"  LM head FLOPs:      {flops_full['lm_head']/1e6:.1f}M -> {flops_optimal['lm_head']/1e6:.1f}M ({(1 - flops_optimal['lm_head']/flops_full['lm_head']):.1%} reduction)")
    print(f"  Total draft FLOPs:  {flops_full['total']/1e6:.1f}M -> {flops_optimal['total']/1e6:.1f}M ({result['latency_reduction']:.1%} reduction)")
    print(f"{'='*60}")

    # Generate plots
    plot_results(result, args.output_dir, target_vocab_size)

    # Save results
    save_results(result, args.output_dir, target_vocab_size)

    # Optionally update config file
    if args.config_path:
        update_config_file(args.config_path, result["optimal_vocab_size"])

    print(f"\n✅ Optimization complete!")
    print(f"   Optimal draft_vocab_size: {result['optimal_vocab_size']:,}")
    print(f"   Token coverage: {result['coverage']:.2%}")
    print(f"\n   Add to your config:")
    print(f'   "draft_vocab_size": {result["optimal_vocab_size"]}')


if __name__ == "__main__":
    main()
