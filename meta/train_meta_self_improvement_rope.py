#!/usr/bin/env python3
"""
Prototype meta self-improvement loop with model-capacity growth.

Highlights:
1. Minimal causal Transformer LM (no pretrained checkpoints).
2. RoPE-based self-attention.
3. Iterative compositional pseudo-label self-improvement.
4. Saturation detection on validation accuracy.
5. Automatic model growth; next stage starts from examples previous stage solved.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ExampleKey = Tuple[int, int, int]  # (digits, a, b)
NUMERIC_PATTERN = re.compile(r"[-+]?\d+")


@dataclass(frozen=True)
class AdditionExample:
    a: int
    b: int
    result: int
    digits: int
    has_carry: bool
    target_override: Optional[str] = None

    def key(self) -> ExampleKey:
        return (self.digits, self.a, self.b)

    def prompt(self) -> str:
        return f"{self.a}+{self.b}="

    def true_target(self) -> str:
        return str(self.result)

    def training_target(self) -> str:
        if self.target_override is not None:
            return self.target_override
        return str(self.result)

    def as_ground_truth(self) -> "AdditionExample":
        return AdditionExample(
            a=self.a,
            b=self.b,
            result=self.result,
            digits=self.digits,
            has_carry=self.has_carry,
            target_override=None,
        )

    def with_override(self, override: Optional[str]) -> "AdditionExample":
        return AdditionExample(
            a=self.a,
            b=self.b,
            result=self.result,
            digits=self.digits,
            has_carry=self.has_carry,
            target_override=override,
        )

    def formatted_a(self) -> str:
        return str(self.a).zfill(self.digits)

    def formatted_b(self) -> str:
        return str(self.b).zfill(self.digits)


def has_carry(a: int, b: int) -> bool:
    carry = 0
    while a > 0 or b > 0 or carry > 0:
        total = (a % 10) + (b % 10) + carry
        if total >= 10:
            return True
        carry = total // 10
        a //= 10
        b //= 10
    return False


def generate_addition_pair(num_digits: int, rng: random.Random, *, allow_carry: bool = True) -> AdditionExample:
    if num_digits <= 0:
        raise ValueError("num_digits must be positive.")
    low = 10 ** (num_digits - 1) if num_digits > 1 else 0
    high = 10**num_digits - 1
    for _ in range(50_000):
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        carry = has_carry(a, b)
        if allow_carry or not carry:
            return AdditionExample(a=a, b=b, result=a + b, digits=num_digits, has_carry=carry)
    raise RuntimeError(f"Failed to sample num_digits={num_digits} (allow_carry={allow_carry}).")


def compose_examples(*examples: AdditionExample) -> AdditionExample:
    if len(examples) < 2:
        raise ValueError("Need at least two component examples for composition.")
    a_str = "".join(ex.formatted_a() for ex in examples)
    b_str = "".join(ex.formatted_b() for ex in examples)
    a_val = int(a_str)
    b_val = int(b_str)
    digits = len(a_str)
    return AdditionExample(
        a=a_val,
        b=b_val,
        result=a_val + b_val,
        digits=digits,
        has_carry=has_carry(a_val, b_val),
    )


def component_has_carry_out(example: AdditionExample) -> bool:
    # True when this component would emit carry into the next more-significant component.
    return example.result >= (10 ** example.digits)


def has_boundary_carry_from_components(components: Sequence[AdditionExample]) -> bool:
    # Detect carry propagation across component boundaries (least-significant component first).
    if len(components) <= 1:
        return False
    carry = 0
    remaining = len(components)
    for ex in reversed(components):
        a_val = ex.a
        b_val = ex.b
        for _ in range(ex.digits):
            total = (a_val % 10) + (b_val % 10) + carry
            carry = 1 if total >= 10 else 0
            a_val //= 10
            b_val //= 10
        remaining -= 1
        if remaining > 0 and carry:
            return True
    return False


def bucket_by_digits(examples: Sequence[AdditionExample]) -> Dict[int, List[AdditionExample]]:
    buckets: Dict[int, List[AdditionExample]] = {}
    for ex in examples:
        buckets.setdefault(ex.digits, []).append(ex)
    return buckets


def compose_to_length(
    buckets: Dict[int, List[AdditionExample]],
    target_digits: int,
    rng: random.Random,
    *,
    allow_carry: bool = True,
    max_attempts: int = 4_000,
) -> Tuple[AdditionExample, List[AdditionExample]]:
    if target_digits <= 0:
        raise ValueError("target_digits must be positive.")
    digit_keys = sorted(k for k, vals in buckets.items() if vals)
    if not digit_keys:
        raise ValueError("No base examples available for composition.")

    for _ in range(max_attempts):
        remaining = target_digits
        components: List[AdditionExample] = []
        while remaining > 0:
            choices = [d for d in digit_keys if d <= remaining]
            if not choices:
                break
            d = rng.choice(choices)
            components.append(rng.choice(buckets[d]))
            remaining -= d
        if remaining == 0 and len(components) >= 2:
            composed = compose_examples(*components)
            if allow_carry or not composed.has_carry:
                return composed, components
    raise RuntimeError(f"Failed to compose target_digits={target_digits}.")


def sample_unique_examples(
    *,
    digits: int,
    count: int,
    rng: random.Random,
    occupied: set[ExampleKey],
    allow_carry: bool = True,
) -> List[AdditionExample]:
    examples: List[AdditionExample] = []
    attempts = 0
    while len(examples) < count:
        attempts += 1
        ex = generate_addition_pair(digits, rng, allow_carry=allow_carry)
        key = ex.key()
        if key in occupied:
            if attempts > 200_000:
                raise RuntimeError(f"Unable to sample enough unique pairs for digits={digits}.")
            continue
        occupied.add(key)
        examples.append(ex)
        attempts = 0
    return examples


def max_unique_pairs_for_digits(digits: int) -> int:
    if digits <= 0:
        raise ValueError("digits must be positive.")
    low = 10 ** (digits - 1) if digits > 1 else 0
    count = (10**digits) - low
    return count * count


def count_occupied_for_digits(occupied: set[ExampleKey], digits: int) -> int:
    return sum(1 for key in occupied if key[0] == digits)


def build_initial_train(
    *,
    min_digits: int,
    max_digits: int,
    per_digit_count: int,
    rng: random.Random,
    occupied: set[ExampleKey],
) -> List[AdditionExample]:
    train: List[AdditionExample] = []
    for digits in range(min_digits, max_digits + 1):
        train.extend(
            sample_unique_examples(
                digits=digits,
                count=per_digit_count,
                rng=rng,
                occupied=occupied,
                allow_carry=True,
            )
        )
    rng.shuffle(train)
    return train


def build_validation(
    *,
    min_digits: int,
    max_digits: int,
    per_digit_count: int,
    rng: random.Random,
    occupied: set[ExampleKey],
) -> List[AdditionExample]:
    val: List[AdditionExample] = []
    for digits in range(min_digits, max_digits + 1):
        val.extend(
            sample_unique_examples(
                digits=digits,
                count=per_digit_count,
                rng=rng,
                occupied=occupied,
                allow_carry=True,
            )
        )
    rng.shuffle(val)
    return val


def build_composed_pool(
    *,
    component_examples: Sequence[AdditionExample],
    min_digits: int,
    max_digits: int,
    per_digit_count: int,
    rng: random.Random,
    occupied: set[ExampleKey],
    boundary_mode: Literal["allow", "no_boundary", "with_carry_filtered"],
) -> Tuple[List[AdditionExample], Dict[ExampleKey, List[ExampleKey]]]:
    pool: List[AdditionExample] = []
    component_map: Dict[ExampleKey, List[ExampleKey]] = {}
    if boundary_mode == "no_boundary":
        # Restrict components to those that do not emit carry to adjacent components.
        eligible_components = [ex for ex in component_examples if not component_has_carry_out(ex)]
    elif boundary_mode in ("allow", "with_carry_filtered"):
        eligible_components = list(component_examples)
    else:
        raise ValueError(f"Unsupported boundary_mode={boundary_mode!r}")
    buckets = bucket_by_digits(eligible_components)

    for digits in range(min_digits, max_digits + 1):
        created = 0
        attempts = 0
        while created < per_digit_count:
            attempts += 1
            composed, components = compose_to_length(buckets, digits, rng, allow_carry=True)
            if boundary_mode == "no_boundary" and has_boundary_carry_from_components(components):
                if attempts > 200_000:
                    raise RuntimeError(
                        f"Unable to create boundary-safe composed examples for digits={digits}."
                    )
                continue
            key = composed.key()
            if key in occupied:
                if attempts > 200_000:
                    raise RuntimeError(f"Unable to create unique composed examples for digits={digits}.")
                continue
            occupied.add(key)
            pool.append(composed)
            component_map[key] = [c.key() for c in components]
            created += 1
            attempts = 0
    rng.shuffle(pool)
    return pool, component_map


def extend_composed_pool(
    *,
    component_examples: Sequence[AdditionExample],
    min_digits: int,
    max_digits: int,
    per_digit_count: int,
    rng: random.Random,
    occupied: set[ExampleKey],
    boundary_mode: Literal["allow", "no_boundary", "with_carry_filtered"],
    max_attempts_per_digit: int = 200_000,
) -> Tuple[List[AdditionExample], Dict[ExampleKey, List[ExampleKey]]]:
    # Best-effort incremental pool expansion used during stalled frontier rounds.
    if per_digit_count <= 0 or max_digits < min_digits:
        return [], {}

    if boundary_mode == "no_boundary":
        eligible_components = [ex for ex in component_examples if not component_has_carry_out(ex)]
    elif boundary_mode in ("allow", "with_carry_filtered"):
        eligible_components = list(component_examples)
    else:
        raise ValueError(f"Unsupported boundary_mode={boundary_mode!r}")

    buckets = bucket_by_digits(eligible_components)
    added: List[AdditionExample] = []
    added_map: Dict[ExampleKey, List[ExampleKey]] = {}

    for digits in range(min_digits, max_digits + 1):
        created = 0
        attempts = 0
        while created < per_digit_count and attempts < max_attempts_per_digit:
            attempts += 1
            try:
                composed, components = compose_to_length(buckets, digits, rng, allow_carry=True)
            except RuntimeError:
                break
            if boundary_mode == "no_boundary" and has_boundary_carry_from_components(components):
                continue
            key = composed.key()
            if key in occupied:
                continue
            occupied.add(key)
            added.append(composed)
            added_map[key] = [c.key() for c in components]
            created += 1
            attempts = 0

    rng.shuffle(added)
    return added, added_map


def sample_unique_composed_against_keys(
    *,
    component_examples: Sequence[AdditionExample],
    min_digits: int,
    max_digits: int,
    target_unique_count: int,
    rng: random.Random,
    occupied: set[ExampleKey],
    excluded_keys: set[ExampleKey],
    boundary_mode: Literal["allow", "no_boundary", "with_carry_filtered"],
    max_attempts: int = 1_000_000,
) -> Tuple[List[AdditionExample], Dict[ExampleKey, List[ExampleKey]]]:
    """Generate composed examples unique vs occupied+excluded with hash-set filtering."""
    if target_unique_count <= 0 or max_digits < min_digits:
        return [], {}

    if boundary_mode == "no_boundary":
        eligible_components = [ex for ex in component_examples if not component_has_carry_out(ex)]
    elif boundary_mode in ("allow", "with_carry_filtered"):
        eligible_components = list(component_examples)
    else:
        raise ValueError(f"Unsupported boundary_mode={boundary_mode!r}")

    buckets = bucket_by_digits(eligible_components)
    if not buckets:
        return [], {}

    active_targets = [d for d in range(min_digits, max_digits + 1)]
    if not active_targets:
        return [], {}

    added: List[AdditionExample] = []
    added_map: Dict[ExampleKey, List[ExampleKey]] = {}
    added_keys: set[ExampleKey] = set()
    blocked_digits: set[int] = set()
    attempts = 0

    while len(added) < target_unique_count and attempts < max_attempts:
        available_targets = [d for d in active_targets if d not in blocked_digits]
        if not available_targets:
            break
        target_digits = rng.choice(available_targets)
        attempts += 1
        try:
            composed, components = compose_to_length(buckets, target_digits, rng, allow_carry=True)
        except RuntimeError:
            blocked_digits.add(target_digits)
            continue

        if boundary_mode == "no_boundary" and has_boundary_carry_from_components(components):
            continue

        key = composed.key()
        if key in occupied or key in excluded_keys or key in added_keys:
            continue

        occupied.add(key)
        added_keys.add(key)
        added.append(composed)
        added_map[key] = [c.key() for c in components]

    rng.shuffle(added)
    return added, added_map


def dedupe_examples(examples: Iterable[AdditionExample]) -> List[AdditionExample]:
    deduped: List[AdditionExample] = []
    seen: set[ExampleKey] = set()
    for ex in examples:
        key = ex.key()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    return deduped


def count_examples_per_digit(examples: Sequence[AdditionExample]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for ex in examples:
        counts[ex.digits] = counts.get(ex.digits, 0) + 1
    return counts


def sample_per_digit_cap(
    examples: Sequence[AdditionExample],
    *,
    per_digit_cap: int,
    rng: random.Random,
) -> List[AdditionExample]:
    """Sample up to per_digit_cap examples inside each digit bucket."""
    deduped = dedupe_examples(examples)
    if per_digit_cap <= 0:
        return deduped
    by_digits: Dict[int, List[AdditionExample]] = {}
    for ex in deduped:
        by_digits.setdefault(ex.digits, []).append(ex)
    sampled: List[AdditionExample] = []
    for _, bucket in sorted(by_digits.items()):
        if len(bucket) > per_digit_cap:
            sampled.extend(rng.sample(bucket, per_digit_cap))
        else:
            sampled.extend(bucket)
    rng.shuffle(sampled)
    return sampled


def merge_seed_and_pseudo(seed_examples: Sequence[AdditionExample], pseudo_examples: Sequence[AdditionExample]) -> List[AdditionExample]:
    merged = dedupe_examples(list(seed_examples) + list(pseudo_examples))
    return merged


def downsample_seed_for_learned_levels(
    seed_examples: Sequence[AdditionExample],
    *,
    max_learned_digits: int,
    retain_per_digit: int,
    rng: random.Random,
) -> Tuple[List[AdditionExample], int]:
    """Downsample older/easier digit buckets while keeping higher-digit buckets intact."""
    deduped = dedupe_examples(seed_examples)
    if retain_per_digit <= 0 or max_learned_digits < 1:
        return deduped, 0

    keep: List[AdditionExample] = []
    by_digits: Dict[int, List[AdditionExample]] = {}
    for ex in deduped:
        if ex.digits <= max_learned_digits:
            by_digits.setdefault(ex.digits, []).append(ex)
        else:
            keep.append(ex)

    for _, bucket in sorted(by_digits.items()):
        if len(bucket) > retain_per_digit:
            keep.extend(rng.sample(bucket, retain_per_digit))
        else:
            keep.extend(bucket)

    kept = dedupe_examples(keep)
    return kept, (len(deduped) - len(kept))


def select_next_round_pseudo(
    pseudo_candidates: Sequence[AdditionExample],
    *,
    seed_examples: Sequence[AdditionExample],
    new_unique_quota_per_digit: int,
    rng: random.Random,
) -> Tuple[List[AdditionExample], Dict[str, object]]:
    deduped_candidates = dedupe_examples(pseudo_candidates)
    seed_keys = {ex.key() for ex in seed_examples}
    unique_candidates = [ex for ex in deduped_candidates if ex.key() not in seed_keys]
    overlap_count = len(deduped_candidates) - len(unique_candidates)

    if new_unique_quota_per_digit > 0:
        selected = sample_per_digit_cap(
            unique_candidates,
            per_digit_cap=new_unique_quota_per_digit,
            rng=rng,
        )
    else:
        selected = list(deduped_candidates)

    selected_per_digit: Dict[str, int] = {}
    for ex in selected:
        key = str(ex.digits)
        selected_per_digit[key] = selected_per_digit.get(key, 0) + 1

    stats = {
        "new_unique_quota_per_digit": new_unique_quota_per_digit,
        "candidate_dedup_total": len(deduped_candidates),
        "unique_candidate_total": len(unique_candidates),
        "overlap_with_seed_total": overlap_count,
        "selected_total": len(selected),
        "selected_per_digit": selected_per_digit,
    }
    return selected, stats


class AdditionTokenizer:
    def __init__(self) -> None:
        tokens = ["<pad>", "<eos>", "+", "="] + [str(i) for i in range(10)]
        self.stoi = {tok: idx for idx, tok in enumerate(tokens)}
        self.itos = {idx: tok for tok, idx in self.stoi.items()}
        self.pad_id = self.stoi["<pad>"]
        self.eos_id = self.stoi["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode_prompt(self, a: int, b: int) -> List[int]:
        text = f"{a}+{b}="
        return [self.stoi[ch] for ch in text]

    def encode_digits(self, value: str) -> List[int]:
        if not value or not all(ch.isdigit() for ch in value):
            raise ValueError(f"Target must be non-empty digits, got: {value!r}")
        return [self.stoi[ch] for ch in value]

    def decode_ids(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        chars: List[str] = []
        for token_id in token_ids:
            token = self.itos.get(int(token_id))
            if token is None:
                continue
            if skip_special_tokens and token in ("<pad>", "<eos>"):
                continue
            chars.append(token)
        return "".join(chars)


class AdditionTokenDataset(Dataset):
    def __init__(self, examples: Sequence[AdditionExample], tokenizer: AdditionTokenizer, *, use_true_targets: bool) -> None:
        self.rows: List[Tuple[List[int], List[int], ExampleKey, int]] = []
        for ex in examples:
            prompt_ids = tokenizer.encode_prompt(ex.a, ex.b)
            target_text = ex.true_target() if use_true_targets else ex.training_target()
            target_ids = tokenizer.encode_digits(target_text)
            input_ids = prompt_ids + target_ids + [tokenizer.eos_id]
            labels = ([-100] * len(prompt_ids)) + target_ids + [tokenizer.eos_id]
            self.rows.append((input_ids, labels, ex.key(), ex.digits))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int], ExampleKey, int]:
        return self.rows[index]


def collate_batch(batch: Sequence[Tuple[List[int], List[int], ExampleKey, int]], pad_id: int) -> Dict[str, object]:
    max_len = max(len(item[0]) for item in batch)
    input_rows: List[List[int]] = []
    label_rows: List[List[int]] = []
    keys: List[ExampleKey] = []
    digits: List[int] = []

    for input_ids, labels, key, d in batch:
        pad_len = max_len - len(input_ids)
        input_rows.append(input_ids + [pad_id] * pad_len)
        label_rows.append(labels + [-100] * pad_len)
        keys.append(key)
        digits.append(d)

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "keys": keys,
        "digits": digits,
    }


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D], cos/sin: [1, 1, T, D/2]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
    return out


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even.")
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self.max_seq_cached = 0

    def _build_cache(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self.max_seq_cached >= seq_len
            and self.cos_cached.device == device
            and self.cos_cached.dtype == dtype
        ):
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        self.cos_cached = cos
        self.sin_cached = sin
        self.max_seq_cached = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        self._build_cache(seq_len, device=q.device, dtype=q.dtype)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


@dataclass(frozen=True)
class ModelStageConfig:
    d_model: int
    n_heads: int
    n_layers: int


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, rope_base: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q, k)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: float, dropout: float, rope_base: float) -> None:
        super().__init__()
        hidden = int(ffn_mult * d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, rope_base)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        stage_cfg: ModelStageConfig,
        ffn_mult: float,
        dropout: float,
        rope_base: float,
        context_window: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_window = context_window
        self.tok_emb = nn.Embedding(vocab_size, stage_cfg.d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=stage_cfg.d_model,
                    n_heads=stage_cfg.n_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    rope_base=rope_base,
                )
                for _ in range(stage_cfg.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(stage_cfg.d_model)
        self.lm_head = nn.Linear(stage_cfg.d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.tok_emb(input_ids)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is None:
            return logits, None
        if logits.size(1) < 2:
            return logits, None
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return logits, loss


def filter_examples_by_digits(
    examples: Sequence[AdditionExample],
    *,
    min_digits: Optional[int] = None,
    max_digits: Optional[int] = None,
) -> List[AdditionExample]:
    filtered: List[AdditionExample] = []
    for ex in examples:
        if min_digits is not None and ex.digits < min_digits:
            continue
        if max_digits is not None and ex.digits > max_digits:
            continue
        filtered.append(ex)
    return filtered


def evaluate_exact_teacher_forced(
    model: MiniTransformerLM,
    examples: Sequence[AdditionExample],
    tokenizer: AdditionTokenizer,
    *,
    device: torch.device,
    batch_size: int,
    use_true_targets: bool,
) -> Tuple[float, Dict[int, float], Dict[ExampleKey, bool]]:
    if not examples:
        return math.nan, {}, {}

    dataset = AdditionTokenDataset(examples, tokenizer, use_true_targets=use_true_targets)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_id),
    )
    model.eval()
    total = 0
    correct = 0
    digit_totals: Dict[int, int] = {}
    digit_correct: Dict[int, int] = {}
    success_map: Dict[ExampleKey, bool] = {}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            keys = batch["keys"]
            digits = batch["digits"]

            logits, _ = model(input_ids, labels=None)
            preds = logits.argmax(dim=-1)
            shift_preds = preds[:, :-1]
            shift_labels = labels[:, 1:]
            valid_mask = shift_labels.ne(-100)
            answer_mask = valid_mask & shift_labels.ne(tokenizer.eos_id)

            for row_idx, key in enumerate(keys):
                row_mask = answer_mask[row_idx]
                d = int(digits[row_idx])
                digit_totals[d] = digit_totals.get(d, 0) + 1
                total += 1

                if not bool(row_mask.any().item()):
                    success = False
                else:
                    success = bool(torch.all(shift_preds[row_idx][row_mask] == shift_labels[row_idx][row_mask]).item())
                success_map[key] = success
                if success:
                    correct += 1
                    digit_correct[d] = digit_correct.get(d, 0) + 1

    per_digit = {
        d: (digit_correct.get(d, 0) / count if count > 0 else math.nan)
        for d, count in digit_totals.items()
    }
    overall = correct / total if total > 0 else math.nan
    return overall, per_digit, success_map


def train_one_round(
    model: MiniTransformerLM,
    examples: Sequence[AdditionExample],
    tokenizer: AdditionTokenizer,
    *,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    num_epochs: int,
) -> float:
    if not examples:
        return math.nan

    dataset = AdditionTokenDataset(examples, tokenizer, use_true_targets=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_id),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    running_loss = 0.0
    step_count = 0

    for _ in range(num_epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids, labels=labels)
            if loss is None:
                continue
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            running_loss += float(loss.detach().item())
            step_count += 1

    return running_loss / max(step_count, 1)


def distill_warmstart_round(
    student: MiniTransformerLM,
    teacher: MiniTransformerLM,
    examples: Sequence[AdditionExample],
    tokenizer: AdditionTokenizer,
    *,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    num_epochs: int,
    alpha: float,
    temperature: float,
) -> float:
    if not examples or num_epochs <= 0:
        return math.nan

    dataset = AdditionTokenDataset(examples, tokenizer, use_true_targets=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_id),
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    teacher.eval()
    student.train()
    running_loss = 0.0
    step_count = 0

    for _ in range(num_epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            valid_mask = labels[:, 1:].ne(-100)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits, _ = teacher(input_ids, labels=None)
            student_logits, _ = student(input_ids, labels=None)

            if student_logits.size(1) < 2:
                continue

            shift_student = student_logits[:, :-1, :].contiguous()
            shift_teacher = teacher_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            hard_loss = F.cross_entropy(
                shift_student.view(-1, shift_student.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            teacher_probs = F.softmax(shift_teacher / temperature, dim=-1)
            student_log_probs = F.log_softmax(shift_student / temperature, dim=-1)
            kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
            mask = valid_mask.float()
            denom = mask.sum().clamp_min(1.0)
            soft_loss = (kl_per_token * mask).sum() / denom
            soft_loss = soft_loss * (temperature * temperature)

            loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(student.parameters(), max_norm=grad_clip)
            optimizer.step()
            running_loss += float(loss.detach().item())
            step_count += 1

    return running_loss / max(step_count, 1)


def select_successful_examples(
    examples: Sequence[AdditionExample],
    success_map: Dict[ExampleKey, bool],
) -> List[AdditionExample]:
    selected: List[AdditionExample] = []
    seen: set[ExampleKey] = set()
    for ex in examples:
        key = ex.key()
        if key in seen:
            continue
        if success_map.get(key, False):
            selected.append(ex.as_ground_truth())
            seen.add(key)
    return selected


def extract_numeric_answer(text: str) -> Optional[str]:
    matches = NUMERIC_PATTERN.findall(text)
    if not matches:
        return None
    best: Optional[str] = None
    best_len = -1
    for token in matches:
        candidate = token.strip()
        length = len(candidate.lstrip("+-"))
        if length > best_len or (length == best_len and candidate != best):
            best = candidate
            best_len = length
    return best


def generate_prediction_map(
    model: MiniTransformerLM,
    examples: Sequence[AdditionExample],
    tokenizer: AdditionTokenizer,
    *,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
) -> Dict[ExampleKey, str]:
    unique: Dict[ExampleKey, AdditionExample] = {}
    for ex in examples:
        key = ex.key()
        if key not in unique:
            unique[key] = ex
    values = list(unique.values())
    predictions: Dict[ExampleKey, str] = {}

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            prompt_rows = [tokenizer.encode_prompt(ex.a, ex.b) for ex in batch]
            prompt_lens = [len(row) for row in prompt_rows]
            max_prompt_len = max(prompt_lens)
            total_len = max_prompt_len + max_new_tokens
            input_ids = torch.full(
                (len(batch), total_len),
                fill_value=tokenizer.pad_id,
                dtype=torch.long,
                device=device,
            )
            for row_idx, row in enumerate(prompt_rows):
                input_ids[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)

            done = [False] * len(batch)
            generated: List[List[int]] = [[] for _ in batch]
            lengths = list(prompt_lens)

            for _ in range(max_new_tokens):
                active_len = max(lengths)
                if active_len <= 0:
                    break
                logits, _ = model(input_ids[:, :active_len], labels=None)
                gather_rows = torch.arange(len(batch), device=device)
                gather_cols = torch.tensor([max(0, l - 1) for l in lengths], dtype=torch.long, device=device)
                next_ids = logits[gather_rows, gather_cols, :].argmax(dim=-1).tolist()

                for idx, token_id in enumerate(next_ids):
                    if done[idx]:
                        continue
                    if lengths[idx] >= total_len:
                        done[idx] = True
                        continue
                    input_ids[idx, lengths[idx]] = int(token_id)
                    lengths[idx] += 1
                    if token_id == tokenizer.eos_id:
                        done[idx] = True
                        continue
                    generated[idx].append(token_id)
                if all(done):
                    break

            for idx, ex in enumerate(batch):
                text = tokenizer.decode_ids(generated[idx], skip_special_tokens=True)
                pred = extract_numeric_answer(text)
                if pred is not None:
                    predictions[ex.key()] = pred.strip()
    if model_was_training:
        model.train()
    return predictions


def evaluate_exact_autoregressive(
    model: MiniTransformerLM,
    examples: Sequence[AdditionExample],
    tokenizer: AdditionTokenizer,
    *,
    device: torch.device,
    batch_size: int,
    use_true_targets: bool,
) -> Tuple[float, Dict[int, float], Dict[ExampleKey, bool]]:
    if not examples:
        return math.nan, {}, {}

    def expected_target(ex: AdditionExample) -> str:
        return ex.true_target() if use_true_targets else ex.training_target()

    max_new_tokens = max(len(expected_target(ex)) for ex in examples) + 2
    prediction_map = generate_prediction_map(
        model,
        examples,
        tokenizer,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    total = 0
    correct = 0
    digit_totals: Dict[int, int] = {}
    digit_correct: Dict[int, int] = {}
    success_map: Dict[ExampleKey, bool] = {}

    for ex in examples:
        key = ex.key()
        d = int(ex.digits)
        target = expected_target(ex)
        pred = prediction_map.get(key)
        success = pred == target
        success_map[key] = success
        total += 1
        digit_totals[d] = digit_totals.get(d, 0) + 1
        if success:
            correct += 1
            digit_correct[d] = digit_correct.get(d, 0) + 1

    per_digit = {
        d: (digit_correct.get(d, 0) / count if count > 0 else math.nan)
        for d, count in digit_totals.items()
    }
    overall = correct / total if total > 0 else math.nan
    return overall, per_digit, success_map


def build_composed_pseudo_dataset(
    composed_pool: Sequence[AdditionExample],
    component_map: Dict[ExampleKey, List[ExampleKey]],
    component_prediction_targets: Dict[ExampleKey, str],
    component_example_lookup: Dict[ExampleKey, AdditionExample],
    *,
    target_max_digits: int,
    max_pseudo_per_round_per_digit: int,
    rng: random.Random,
    boundary_mode: Literal["allow", "no_boundary", "with_carry_filtered"],
) -> Tuple[List[AdditionExample], Dict[str, int]]:
    pseudo: List[AdditionExample] = []
    candidates = 0
    kept = 0
    missing = 0
    filtered_boundary = 0
    filter_boundary_carry = boundary_mode == "with_carry_filtered"

    for ex in composed_pool:
        if ex.digits > target_max_digits:
            continue
        candidates += 1
        component_keys = component_map.get(ex.key(), [])
        if not component_keys:
            missing += 1
            continue
        if filter_boundary_carry:
            components: List[AdditionExample] = []
            for key in component_keys:
                component_ex = component_example_lookup.get(key)
                if component_ex is None:
                    components = []
                    break
                components.append(component_ex)
            if not components:
                missing += 1
                continue
            if has_boundary_carry_from_components(components):
                filtered_boundary += 1
                continue
        stitched: List[str] = []
        good = True
        for key in component_keys:
            value = component_prediction_targets.get(key)
            if value is None:
                good = False
                break
            stitched.append(value)
        if not good:
            missing += 1
            continue
        override = "".join(stitched)
        if not override.isdigit():
            missing += 1
            continue
        pseudo.append(ex.with_override(override))
        kept += 1

    pseudo = sample_per_digit_cap(
        pseudo,
        per_digit_cap=max_pseudo_per_round_per_digit,
        rng=rng,
    )

    stats = {
        "candidate_total": candidates,
        "retained_total": kept,
        "retained_after_per_digit_cap_total": len(pseudo),
        "max_pseudo_per_round_per_digit": max_pseudo_per_round_per_digit,
        "missing_total": missing,
        "filtered_boundary_total": filtered_boundary,
        "invalid_total": 0,
    }
    return dedupe_examples(pseudo), stats


def encode_key(key: ExampleKey) -> str:
    return f"{key[0]}|{key[1]}|{key[2]}"


def decode_key(raw: str) -> ExampleKey:
    parts = raw.split("|")
    if len(parts) != 3:
        raise ValueError(f"Invalid encoded key: {raw!r}")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def serialize_example(ex: AdditionExample) -> Dict[str, object]:
    return {
        "a": ex.a,
        "b": ex.b,
        "result": ex.result,
        "digits": ex.digits,
        "has_carry": ex.has_carry,
        "target_override": ex.target_override,
    }


def deserialize_example(payload: Dict[str, object]) -> AdditionExample:
    return AdditionExample(
        a=int(payload["a"]),
        b=int(payload["b"]),
        result=int(payload["result"]),
        digits=int(payload["digits"]),
        has_carry=bool(payload["has_carry"]),
        target_override=payload.get("target_override"),
    )


def save_examples_jsonl(path: Path, examples: Sequence[AdditionExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            json.dump(serialize_example(ex), handle)
            handle.write("\n")


def append_examples_jsonl(path: Path, examples: Sequence[AdditionExample]) -> None:
    if not examples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for ex in examples:
            json.dump(serialize_example(ex), handle)
            handle.write("\n")


def load_examples_jsonl(path: Path) -> List[AdditionExample]:
    examples: List[AdditionExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            examples.append(deserialize_example(payload))
    return examples


def load_component_map_json(path: Path) -> Dict[ExampleKey, List[ExampleKey]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    component_map: Dict[ExampleKey, List[ExampleKey]] = {}
    for key_raw, comps_raw in raw.items():
        key = decode_key(str(key_raw))
        comps = [decode_key(str(item)) for item in comps_raw]
        component_map[key] = comps
    return component_map


def append_component_map_jsonl(path: Path, component_map: Dict[ExampleKey, List[ExampleKey]]) -> None:
    if not component_map:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for key, comps in component_map.items():
            payload = {
                "key": encode_key(key),
                "components": [encode_key(comp) for comp in comps],
            }
            json.dump(payload, handle)
            handle.write("\n")


def load_component_map_augments_jsonl(path: Path) -> Dict[ExampleKey, List[ExampleKey]]:
    component_map: Dict[ExampleKey, List[ExampleKey]] = {}
    if not path.exists():
        return component_map
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            key = decode_key(str(payload["key"]))
            comps = [decode_key(str(item)) for item in payload["components"]]
            component_map[key] = comps
    return component_map


def parse_stage_configs(raw: str) -> List[ModelStageConfig]:
    configs: List[ModelStageConfig] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        normalized = chunk.replace(":", "x")
        parts = normalized.split("x")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid stage spec '{chunk}'. Use d_modelxn_headsxn_layers, e.g. 96x4x2."
            )
        d_model, n_heads, n_layers = (int(parts[0]), int(parts[1]), int(parts[2]))
        if d_model <= 0 or n_heads <= 0 or n_layers <= 0:
            raise ValueError(f"Invalid non-positive stage spec '{chunk}'.")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads in '{chunk}'.")
        if (d_model // n_heads) % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE in '{chunk}'.")
        configs.append(ModelStageConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers))
    if not configs:
        raise ValueError("At least one stage config is required.")
    return configs


def instantiate_model(
    tokenizer: AdditionTokenizer,
    stage_cfg: ModelStageConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> MiniTransformerLM:
    model = MiniTransformerLM(
        vocab_size=tokenizer.vocab_size,
        stage_cfg=stage_cfg,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        rope_base=args.rope_base,
        context_window=args.context_window,
    )
    model.to(device)
    return model


def max_supported_digits_for_context_window(context_window: int) -> int:
    # Longest training row is "a+b=" + target + "<eos>" where:
    # prompt length ~= 2*digits + 2, target length <= digits + 1, eos=1.
    # So 3*digits + 4 must fit in context_window.
    return max(1, (context_window - 4) // 3)


def resolve_expand_span(
    *,
    current_max_digits: int,
    fixed_expand_num_digits: int,
    expand_num_digits_division_factor: float,
) -> int:
    if fixed_expand_num_digits > 0:
        return fixed_expand_num_digits
    raw = ((current_max_digits * 2) - current_max_digits) / expand_num_digits_division_factor
    adaptive_span = max(1, int(math.floor(raw)))
    # Prevent abrupt frontier jumps; adaptive growth cannot exceed the current frontier width.
    return min(adaptive_span, current_max_digits)


def build_frontier_expansion_schedule(
    *,
    initial_max_digits: int,
    num_rounds: int,
    fixed_expand_num_digits: int,
    expand_num_digits_division_factor: float,
    max_digits_cap: int,
) -> Tuple[List[int], List[int]]:
    targets: List[int] = []
    spans: List[int] = []
    frontier = initial_max_digits

    for _ in range(num_rounds):
        span = resolve_expand_span(
            current_max_digits=frontier,
            fixed_expand_num_digits=fixed_expand_num_digits,
            expand_num_digits_division_factor=expand_num_digits_division_factor,
        )
        next_frontier = min(max_digits_cap, frontier + span)
        if next_frontier <= frontier:
            break
        targets.append(next_frontier)
        spans.append(span)
        frontier = next_frontier

    return targets, spans


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meta self-improvement prototype with a minimal RoPE Transformer and capacity growth."
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/runs/meta_self_improvement/rope_prototype")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")

    parser.add_argument("--initial-min-digits", type=int, default=3)
    parser.add_argument("--initial-max-digits", type=int, default=6)
    parser.add_argument("--initial-train-per-digit", type=int, default=300)
    parser.add_argument("--validation-per-digit", type=int, default=80)
    parser.add_argument("--composed-train-per-digit", type=int, default=300)
    parser.add_argument(
        "--composed-boundary-mode",
        type=str,
        choices=("allow", "no_boundary", "with_carry_filtered"),
        default="no_boundary",
        help=(
            "How to compose pseudo-data components: allow boundary carry, avoid it by construction, "
            "or allow-and-filter boundary-carry pseudo labels."
        ),
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=50,
        help="Number of frontier expansion steps after initial mastery.",
    )
    parser.add_argument(
        "--expand-num-digits",
        type=int,
        default=0,
        help=(
            "Fixed frontier expansion span per step. Use 0 to enable adaptive expansion "
            "span: ((current_max_digits*2 - current_max_digits) / expand_num_digits_division_factor), "
            "capped at current_max_digits."
        ),
    )
    parser.add_argument(
        "--expand-num-digits-division-factor",
        type=float,
        default=2.0,
        help=(
            "Division factor used by adaptive expansion span when --expand-num-digits=0."
        ),
    )
    parser.add_argument(
        "--max-pseudo-per-round-per-digit",
        type=int,
        default=1000,
        help=(
            "If > 0, keep up to this many pseudo examples per digit bucket each round "
            "(across all digits <= current frontier)."
        ),
    )
    parser.add_argument(
        "--new-pseudo-unique-quota-per-digit",
        type=int,
        default=1000,
        help=(
            "If > 0, keep up to this many seed-unique pseudo examples per digit bucket "
            "for the next training round."
        ),
    )
    parser.add_argument(
        "--min-unique-candidate-per-digit",
        type=int,
        default=200,
        help=(
            "Require at least this many seed-unique pseudo candidates per digit bucket each round "
            "(converted to a total target using active pseudo digits). "
            "If unmet, composed pool is augmented before pseudo selection."
        ),
    )
    parser.add_argument(
        "--unique-candidate-augment-per-digit",
        type=int,
        default=0,
        help=(
            "When enforcing min-unique-candidate-per-digit, add this many composed examples per edge digit "
            "per augmentation event. 0 falls back to stall-bootstrap-composed-per-digit or composed-train-per-digit."
        ),
    )
    parser.add_argument(
        "--unique-candidate-augment-max-events-per-edge-digit",
        type=int,
        default=4,
        help=(
            "Maximum augmentation events while enforcing min-unique-candidate-per-digit. "
            "Each event augments every edge digit by --unique-candidate-augment-per-digit."
        ),
    )
    parser.add_argument(
        "--min-unique-candidate-hard-fail",
        action="store_true",
        help="Abort the run if the computed unique-candidate requirement cannot be met after augmentation attempts.",
    )
    parser.add_argument(
        "--min-unique-candidate-min-component-acc",
        type=float,
        default=0.80,
        help=(
            "Only enforce unique-candidate requirements (and hard-fail) when component prediction coverage "
            "reaches this threshold."
        ),
    )
    parser.add_argument(
        "--seed-retention-per-digit",
        type=int,
        default=1000,
        help=(
            "After initial mastery, downsample learned (older) digit buckets in seed to this many "
            "examples per digit. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-total-rounds",
        type=int,
        default=80,
        help="Safety cap on total rounds including initial bootstrapping rounds.",
    )

    parser.add_argument("--stage-configs", type=str, default="96x4x2,128x4x3,160x5x4")
    parser.add_argument("--ffn-mult", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rope-base", type=float, default=10_000.0)
    parser.add_argument("--context-window", type=int, default=512)

    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument(
        "--initial-bootstrapping-epochs",
        type=int,
        default=10,
        help="Epochs per round before initial distribution is mastered.",
    )
    parser.add_argument("--growth-warmup-epochs", type=int, default=2)
    parser.add_argument(
        "--growth-distill-epochs",
        type=int,
        default=0,
        help="Optional distillation epochs from previous stage into the grown stage.",
    )
    parser.add_argument(
        "--growth-distill-alpha",
        type=float,
        default=0.7,
        help="Hard-label loss weight for growth distillation (1-alpha scales soft loss).",
    )
    parser.add_argument(
        "--growth-distill-temperature",
        type=float,
        default=1.0,
        help="Temperature for growth distillation soft targets.",
    )
    parser.add_argument(
        "--growth-distill-learning-rate",
        type=float,
        default=0.0,
        help="Learning rate for growth distillation; 0 uses --learning-rate.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument(
        "--full-eval-interval",
        type=int,
        default=0,
        help=(
            "Run full-range validation every N rounds. 0 disables per-round full validation "
            "and keeps frontier-only evaluation."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--bootstrap-new-examples-per-digit",
        type=int,
        default=0,
        help="Before initial mastery, sample this many fresh initial-range examples per digit each round.",
    )
    parser.add_argument(
        "--stall-bootstrap-composed-per-digit",
        type=int,
        default=0,
        help=(
            "When frontier validation stalls below expansion threshold, add this many fresh composed "
            "examples per edge-digit before considering capacity growth."
        ),
    )
    parser.add_argument(
        "--stall-bootstrap-patience",
        type=int,
        default=3,
        help="Stale frontier rounds required before adding extra composed bootstrap data.",
    )
    parser.add_argument(
        "--stall-bootstrap-max-events-per-edge-digit",
        type=int,
        default=3,
        help=(
            "Maximum stall-bootstrap injections each edge digit can receive before allowing "
            "capacity-growth escalation."
        ),
    )

    parser.add_argument("--saturation-patience", type=int, default=2)
    parser.add_argument("--saturation-delta", type=float, default=0.002)
    parser.add_argument("--min-rounds-before-growth", type=int, default=2)
    parser.add_argument(
        "--min-rounds-per-frontier-before-growth",
        type=int,
        default=2,
        help="Require this many rounds at the current frontier before allowing capacity growth.",
    )
    parser.add_argument("--min-growth-seed-examples", type=int, default=256)
    parser.add_argument(
        "--initial-mastery-threshold",
        type=float,
        default=0.98,
        help="Required accuracy on initial digit range before frontier expansion begins.",
    )
    parser.add_argument(
        "--frontier-expand-threshold",
        type=float,
        default=0.90,
        help="Increase frontier digits only when current frontier accuracy reaches this threshold.",
    )
    parser.add_argument(
        "--frontier-mastery-threshold",
        type=float,
        default=0.95,
        help="If frontier accuracy is above this, avoid growth even if improvements are small.",
    )
    parser.add_argument(
        "--allow-growth-before-initial-mastery",
        action="store_true",
        help="Permit model growth before initial distribution is mastered.",
    )
    parser.add_argument(
        "--initial-mastery-patience",
        type=int,
        default=2,
        help="Require this many consecutive rounds over initial-mastery-threshold.",
    )
    parser.add_argument(
        "--log-round-breakdown",
        action="store_true",
        help="Log per-round timing breakdown for train/eval/pseudo blocks.",
    )
    parser.add_argument(
        "--log-round-breakdown-sync-cuda",
        action="store_true",
        help=(
            "Synchronize CUDA around timed blocks for more accurate measurements. "
            "Adds small overhead when --log-round-breakdown is enabled."
        ),
    )
    parser.add_argument(
        "--resume-from-latest-state",
        action="store_true",
        help="Resume from continuation state saved in output-dir (or --continuation-state-path).",
    )
    parser.add_argument(
        "--continuation-state-path",
        type=str,
        default="",
        help="Path for continuation state checkpoint (defaults to <output-dir>/continuation_state.pt).",
    )
    parser.add_argument(
        "--disable-continuation-save",
        action="store_true",
        help="Disable periodic continuation checkpoint writes.",
    )
    parser.add_argument(
        "--continuation-save-interval",
        type=int,
        default=5,
        help="Save continuation state every N rounds (ignored when continuation save is disabled).",
    )
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.initial_min_digits < 1:
        raise ValueError("initial_min_digits must be >= 1.")
    if args.initial_max_digits < args.initial_min_digits:
        raise ValueError("initial_max_digits must be >= initial_min_digits.")
    if args.num_rounds < 0:
        raise ValueError("num_rounds must be non-negative.")
    if args.expand_num_digits < 0:
        raise ValueError("expand_num_digits must be >= 0 (0 enables adaptive expansion).")
    if args.expand_num_digits_division_factor <= 0:
        raise ValueError("expand_num_digits_division_factor must be positive.")
    if args.initial_train_per_digit <= 0:
        raise ValueError("initial_train_per_digit must be positive.")
    if args.validation_per_digit <= 0:
        raise ValueError("validation_per_digit must be positive.")
    if args.composed_train_per_digit <= 0:
        raise ValueError("composed_train_per_digit must be positive.")
    if args.max_pseudo_per_round_per_digit < 0:
        raise ValueError("max_pseudo_per_round_per_digit must be >= 0.")
    if args.new_pseudo_unique_quota_per_digit < 0:
        raise ValueError("new_pseudo_unique_quota_per_digit must be >= 0.")
    if args.min_unique_candidate_per_digit < 0:
        raise ValueError("min_unique_candidate_per_digit must be >= 0.")
    if args.unique_candidate_augment_per_digit < 0:
        raise ValueError("unique_candidate_augment_per_digit must be >= 0.")
    if args.unique_candidate_augment_max_events_per_edge_digit < 1:
        raise ValueError("unique_candidate_augment_max_events_per_edge_digit must be >= 1.")
    if not (0.0 <= args.min_unique_candidate_min_component_acc <= 1.0):
        raise ValueError("min_unique_candidate_min_component_acc must be in [0, 1].")
    if args.full_eval_interval < 0:
        raise ValueError("full_eval_interval must be >= 0.")
    if args.seed_retention_per_digit < 0:
        raise ValueError("seed_retention_per_digit must be >= 0.")
    if not (0.0 <= args.initial_mastery_threshold <= 1.0):
        raise ValueError("initial_mastery_threshold must be in [0, 1].")
    if not (0.0 <= args.frontier_expand_threshold <= 1.0):
        raise ValueError("frontier_expand_threshold must be in [0, 1].")
    if not (0.0 <= args.frontier_mastery_threshold <= 1.0):
        raise ValueError("frontier_mastery_threshold must be in [0, 1].")
    if args.initial_mastery_patience < 1:
        raise ValueError("initial_mastery_patience must be >= 1.")
    if args.bootstrap_new_examples_per_digit < 0:
        raise ValueError("bootstrap_new_examples_per_digit must be >= 0.")
    if args.stall_bootstrap_composed_per_digit < 0:
        raise ValueError("stall_bootstrap_composed_per_digit must be >= 0.")
    if args.stall_bootstrap_patience < 1:
        raise ValueError("stall_bootstrap_patience must be >= 1.")
    if args.stall_bootstrap_max_events_per_edge_digit < 1:
        raise ValueError("stall_bootstrap_max_events_per_edge_digit must be >= 1.")
    if args.min_rounds_per_frontier_before_growth < 1:
        raise ValueError("min_rounds_per_frontier_before_growth must be >= 1.")
    if args.context_window < 8:
        raise ValueError("context_window must be >= 8.")
    if args.growth_distill_epochs < 0:
        raise ValueError("growth_distill_epochs must be >= 0.")
    if not (0.0 <= args.growth_distill_alpha <= 1.0):
        raise ValueError("growth_distill_alpha must be in [0, 1].")
    if args.growth_distill_temperature <= 0:
        raise ValueError("growth_distill_temperature must be positive.")
    if args.growth_distill_learning_rate < 0:
        raise ValueError("growth_distill_learning_rate must be >= 0.")
    if (not args.disable_continuation_save) and args.continuation_save_interval < 1:
        raise ValueError("continuation_save_interval must be >= 1.")
    if args.log_round_breakdown_sync_cuda:
        args.log_round_breakdown = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    stage_cfgs = parse_stage_configs(args.stage_configs)
    tokenizer = AdditionTokenizer()
    max_context_digits = max_supported_digits_for_context_window(args.context_window)
    if args.initial_max_digits > max_context_digits:
        raise ValueError(
            "initial_max_digits exceeds context-window capacity "
            f"(initial_max_digits={args.initial_max_digits}, max_supported={max_context_digits})."
        )
    frontier_targets, frontier_spans = build_frontier_expansion_schedule(
        initial_max_digits=args.initial_max_digits,
        num_rounds=args.num_rounds,
        fixed_expand_num_digits=args.expand_num_digits,
        expand_num_digits_division_factor=args.expand_num_digits_division_factor,
        max_digits_cap=max_context_digits,
    )
    planned_expansion_rounds = len(frontier_targets)
    if args.max_total_rounds < planned_expansion_rounds + 1:
        raise ValueError(
            "max_total_rounds should be at least planned_expansion_rounds + 1 "
            f"(planned={planned_expansion_rounds})."
        )
    final_max_digits = frontier_targets[-1] if frontier_targets else args.initial_max_digits
    frontier_span_digits = resolve_expand_span(
        current_max_digits=args.initial_max_digits,
        fixed_expand_num_digits=args.expand_num_digits,
        expand_num_digits_division_factor=args.expand_num_digits_division_factor,
    )

    summary_path = output_dir / "summary.json"
    continuation_state_path = (
        Path(args.continuation_state_path)
        if args.continuation_state_path
        else (output_dir / "continuation_state.pt")
    )
    initial_train_path = data_dir / "initial_train.jsonl"
    composed_pool_path = data_dir / "composed_pool.jsonl"
    validation_path = data_dir / "validation.jsonl"
    component_map_path = data_dir / "composed_component_map.json"
    composed_pool_augments_path = data_dir / "composed_pool_augments.jsonl"
    composed_component_map_augments_path = data_dir / "composed_component_map_augments.jsonl"

    occupied: set[ExampleKey] = set()
    round_idx = 0
    latest_full_val_acc = math.nan
    round_records: List[Dict[str, object]] = []
    growth_events: List[Dict[str, object]] = []
    stop_reason = "max_total_rounds_reached"
    stop_requested = False
    signal_stop_reason: Optional[str] = None

    if args.resume_from_latest_state:
        if not continuation_state_path.exists():
            raise FileNotFoundError(
                f"Continuation state not found: {continuation_state_path}. "
                "Run without --resume-from-latest-state or provide --continuation-state-path."
            )
        for required in (initial_train_path, composed_pool_path, validation_path, component_map_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"Missing dataset artifact for resume: {required}"
                )

        initial_train = load_examples_jsonl(initial_train_path)
        composed_pool = load_examples_jsonl(composed_pool_path)
        if composed_pool_augments_path.exists():
            composed_pool.extend(load_examples_jsonl(composed_pool_augments_path))
        composed_pool = dedupe_examples(composed_pool)
        validation = load_examples_jsonl(validation_path)
        component_map = load_component_map_json(component_map_path)
        component_map.update(load_component_map_augments_jsonl(composed_component_map_augments_path))

        component_example_lookup = {ex.key(): ex for ex in initial_train}
        occupied.update(ex.key() for ex in initial_train)
        occupied.update(ex.key() for ex in validation)
        occupied.update(ex.key() for ex in composed_pool)

        state = torch.load(continuation_state_path, map_location="cpu")
        stage_idx = int(state["stage_idx"])
        if stage_idx < 0 or stage_idx >= len(stage_cfgs):
            raise ValueError(
                f"Continuation state stage index out of range: {stage_idx} (available stages={len(stage_cfgs)})."
            )
        model = instantiate_model(tokenizer, stage_cfgs[stage_idx], args, device)
        model.load_state_dict(state["model_state_dict"])

        seed_examples = dedupe_examples(
            [deserialize_example(payload) for payload in state.get("seed_examples", [])]
        )
        pseudo_examples = dedupe_examples(
            [deserialize_example(payload) for payload in state.get("pseudo_examples", [])]
        )
        pending_growth_raw = state.get("pending_growth_seed_examples")
        pending_growth_seed = (
            dedupe_examples([deserialize_example(payload) for payload in pending_growth_raw])
            if pending_growth_raw is not None
            else None
        )

        growth_pending = bool(state.get("growth_pending", False))
        stage_best_frontier_val = float(state.get("stage_best_frontier_val", -1.0))
        stale_rounds = int(state.get("stale_rounds", 0))
        rounds_in_current_frontier = int(state.get("rounds_in_current_frontier", 0))
        stall_bootstrap_events_in_frontier = int(state.get("stall_bootstrap_events_in_frontier", 0))
        expansion_step = int(state.get("expansion_step", 0))
        frontier_max_digits = int(state.get("frontier_max_digits", args.initial_max_digits))
        frontier_span_digits = int(state.get("frontier_span_digits", frontier_span_digits))
        initial_mastered = bool(state.get("initial_mastered", False))
        initial_mastery_streak = int(state.get("initial_mastery_streak", 0))
        trained_at_final_frontier = bool(state.get("trained_at_final_frontier", False))
        round_idx = int(state.get("next_round_index", 0))
        latest_full_val_acc = float(state.get("latest_full_val_acc", math.nan))

        if "python_rng_state" in state:
            rng.setstate(state["python_rng_state"])
        if "torch_rng_state" in state:
            torch.random.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and "torch_cuda_rng_state_all" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])

        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary_payload = json.load(handle)
            round_records = list(summary_payload.get("rounds", []))
            growth_events = list(summary_payload.get("growth_events", []))

        print(
            f"[INFO] Resumed from {continuation_state_path}: next_round={round_idx} stage={stage_idx} frontier<={frontier_max_digits}",
            flush=True,
        )
    else:
        initial_train = build_initial_train(
            min_digits=args.initial_min_digits,
            max_digits=args.initial_max_digits,
            per_digit_count=args.initial_train_per_digit,
            rng=rng,
            occupied=occupied,
        )
        component_example_lookup = {ex.key(): ex for ex in initial_train}
        composed_pool, component_map = build_composed_pool(
            component_examples=initial_train,
            min_digits=args.initial_max_digits + 1,
            max_digits=final_max_digits,
            per_digit_count=args.composed_train_per_digit,
            rng=rng,
            occupied=occupied,
            boundary_mode=args.composed_boundary_mode,
        )
        validation = build_validation(
            min_digits=args.initial_min_digits,
            max_digits=final_max_digits,
            per_digit_count=args.validation_per_digit,
            rng=rng,
            occupied=occupied,
        )

        save_examples_jsonl(initial_train_path, initial_train)
        save_examples_jsonl(composed_pool_path, composed_pool)
        save_examples_jsonl(validation_path, validation)
        with component_map_path.open("w", encoding="utf-8") as handle:
            payload = {encode_key(k): [encode_key(c) for c in comps] for k, comps in component_map.items()}
            json.dump(payload, handle, indent=2)
        composed_pool_augments_path.write_text("", encoding="utf-8")
        composed_component_map_augments_path.write_text("", encoding="utf-8")

        model = instantiate_model(tokenizer, stage_cfgs[0], args, device)
        stage_idx = 0
        seed_examples = dedupe_examples(initial_train)
        pseudo_examples = []
        pending_growth_seed = None
        growth_pending = False
        stage_best_frontier_val = -1.0
        stale_rounds = 0
        rounds_in_current_frontier = 0
        stall_bootstrap_events_in_frontier = 0
        expansion_step = 0
        frontier_max_digits = args.initial_max_digits
        initial_mastered = False
        initial_mastery_streak = 0
        trained_at_final_frontier = False

    validation_initial = filter_examples_by_digits(validation, max_digits=args.initial_max_digits)

    def build_summary(current_stop_reason: str) -> Dict[str, object]:
        return {
            "config": vars(args),
            "device": str(device),
            "stage_configs": [asdict(cfg) for cfg in stage_cfgs],
            "rounds": round_records,
            "growth_events": growth_events,
            "planned_expansion_rounds": planned_expansion_rounds,
            "context_window_max_digits": max_context_digits,
            "frontier_targets": frontier_targets,
            "frontier_spans": frontier_spans,
            "final_stage_index": stage_idx,
            "final_seed_examples": len(seed_examples),
            "final_frontier_max_digits": frontier_max_digits,
            "final_expansion_step": expansion_step,
            "initial_mastered": initial_mastered,
            "trained_at_final_frontier": trained_at_final_frontier,
            "stop_reason": current_stop_reason,
        }

    def flush_summary(current_stop_reason: str) -> None:
        summary_tmp_path = output_dir / "summary.json.tmp"
        with summary_tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(build_summary(current_stop_reason), handle, indent=2)
        summary_tmp_path.replace(summary_path)

    continuation_save_enabled = not args.disable_continuation_save

    def save_continuation_state(next_round_index: int) -> None:
        if not continuation_save_enabled:
            return
        payload = {
            "next_round_index": next_round_index,
            "stage_idx": stage_idx,
            "model_state_dict": model.state_dict(),
            "seed_examples": [serialize_example(ex) for ex in seed_examples],
            "pseudo_examples": [serialize_example(ex) for ex in pseudo_examples],
            "pending_growth_seed_examples": (
                [serialize_example(ex) for ex in pending_growth_seed]
                if pending_growth_seed is not None
                else None
            ),
            "growth_pending": growth_pending,
            "stage_best_frontier_val": stage_best_frontier_val,
            "stale_rounds": stale_rounds,
            "rounds_in_current_frontier": rounds_in_current_frontier,
            "stall_bootstrap_events_in_frontier": stall_bootstrap_events_in_frontier,
            "expansion_step": expansion_step,
            "frontier_max_digits": frontier_max_digits,
            "frontier_span_digits": frontier_span_digits,
            "initial_mastered": initial_mastered,
            "initial_mastery_streak": initial_mastery_streak,
            "trained_at_final_frontier": trained_at_final_frontier,
            "latest_full_val_acc": latest_full_val_acc,
            "python_rng_state": rng.getstate(),
            "torch_rng_state": torch.random.get_rng_state(),
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        tmp_path = continuation_state_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(continuation_state_path)

    def handle_stop_signal(signum: int, _frame: object) -> None:
        nonlocal stop_reason, stop_requested, signal_stop_reason
        stop_requested = True
        signal_name = signal.Signals(signum).name.lower()
        signal_stop_reason = f"signal_{signal_name}"
        stop_reason = signal_stop_reason
        print(
            f"[WARN] Received {signal.Signals(signum).name}; will stop after current round boundary.",
            flush=True,
        )
        try:
            flush_summary(stop_reason)
            print(f"[INFO] Saved interim summary to {summary_path}", flush=True)
        except Exception as exc:
            print(f"[WARN] Failed to flush interim summary after signal: {exc}", flush=True)

    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, handle_stop_signal)
    signal.signal(signal.SIGINT, handle_stop_signal)

    print(f"[INFO] Output dir: {output_dir}", flush=True)
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Stage configs: {[asdict(cfg) for cfg in stage_cfgs]}", flush=True)
    print(
        "[INFO] Dataset sizes: initial_train={} composed_pool={} validation={} initial_validation={}".format(
            len(initial_train),
            len(composed_pool),
            len(validation),
            len(validation_initial),
        ),
        flush=True,
    )
    print(f"[INFO] Composed boundary mode: {args.composed_boundary_mode}", flush=True)
    print(
        "[INFO] Full validation interval: {}".format(
            args.full_eval_interval if args.full_eval_interval > 0 else "disabled (frontier-only)"
        ),
        flush=True,
    )
    if continuation_save_enabled:
        print(
            f"[INFO] Continuation state: {continuation_state_path} (interval={args.continuation_save_interval} rounds)",
            flush=True,
        )
    elif args.resume_from_latest_state:
        print("[WARN] Resumed run with continuation-save disabled.", flush=True)
    if args.log_round_breakdown:
        print(
            "[INFO] Round timing breakdown enabled (sync_cuda={}).".format(
                args.log_round_breakdown_sync_cuda
            ),
            flush=True,
        )
    if args.expand_num_digits > 0:
        expand_mode = f"fixed({args.expand_num_digits})"
    else:
        expand_mode = (
            "adaptive(((2*d)-d)/"
            f"{args.expand_num_digits_division_factor:g})"
        )
    print(f"[INFO] Context-window max digits: {max_context_digits}", flush=True)
    print(
        "[INFO] Final target max digits: {} (requested_num_rounds={} planned_expansions={} mode={})".format(
            final_max_digits,
            args.num_rounds,
            planned_expansion_rounds,
            expand_mode,
        ),
        flush=True,
    )
    if planned_expansion_rounds < args.num_rounds:
        print(
            "[WARN] Expansion rounds truncated by context-window cap: "
            f"requested={args.num_rounds} planned={planned_expansion_rounds}.",
            flush=True,
        )

    def start_timing() -> Optional[float]:
        if not args.log_round_breakdown:
            return None
        if args.log_round_breakdown_sync_cuda and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def stop_timing(label: str, start: Optional[float], timings: Dict[str, float]) -> None:
        if start is None:
            return
        if args.log_round_breakdown_sync_cuda and device.type == "cuda":
            torch.cuda.synchronize(device)
        timings[label] = timings.get(label, 0.0) + (time.perf_counter() - start)

    flush_summary(stop_reason)
    while round_idx < args.max_total_rounds:
        if stop_requested:
            if signal_stop_reason is not None:
                stop_reason = signal_stop_reason
            print("[WARN] Stop requested by signal before starting next round; stopping.", flush=True)
            break
        just_grew = False
        if growth_pending:
            teacher_model = model
            stage_idx += 1
            if stage_idx >= len(stage_cfgs):
                raise RuntimeError("Internal error: growth pending but no next stage exists.")
            model = instantiate_model(tokenizer, stage_cfgs[stage_idx], args, device)
            distill_loss = math.nan
            seed_examples = dedupe_examples(pending_growth_seed or seed_examples)
            if args.growth_distill_epochs > 0:
                distill_lr = (
                    args.growth_distill_learning_rate
                    if args.growth_distill_learning_rate > 0
                    else args.learning_rate
                )
                distill_loss = distill_warmstart_round(
                    model,
                    teacher_model,
                    seed_examples,
                    tokenizer,
                    device=device,
                    batch_size=args.batch_size,
                    learning_rate=distill_lr,
                    weight_decay=args.weight_decay,
                    grad_clip=args.grad_clip,
                    num_epochs=args.growth_distill_epochs,
                    alpha=args.growth_distill_alpha,
                    temperature=args.growth_distill_temperature,
                )
                print(
                    "[INFO] Growth distillation -> stage {}: epochs={} loss={:.4f} alpha={} temp={} lr={}".format(
                        stage_idx,
                        args.growth_distill_epochs,
                        distill_loss if not math.isnan(distill_loss) else float("nan"),
                        args.growth_distill_alpha,
                        args.growth_distill_temperature,
                        distill_lr,
                    ),
                    flush=True,
                )
            del teacher_model
            pseudo_examples = []
            stage_best_frontier_val = -1.0
            stale_rounds = 0
            rounds_in_current_frontier = 0
            stall_bootstrap_events_in_frontier = 0
            growth_pending = False
            just_grew = True
            growth_events.append(
                {
                    "round_index": round_idx,
                    "new_stage_index": stage_idx,
                    "new_stage_config": asdict(stage_cfgs[stage_idx]),
                    "seed_examples": len(seed_examples),
                    "frontier_max_digits": frontier_max_digits,
                }
            )
            print(
                f"[INFO] Capacity growth -> stage {stage_idx}: {asdict(stage_cfgs[stage_idx])}, seed={len(seed_examples)} frontier<={frontier_max_digits}",
                flush=True,
            )

        round_start = time.time()
        round_timing: Dict[str, float] = {}
        round_frontier_digits = frontier_max_digits
        pseudo_used_count = len(pseudo_examples)
        seed_retention_pruned_count = 0
        round_frontier_span_digits = frontier_span_digits
        seed_retention_max_digits = round_frontier_digits - round_frontier_span_digits
        bootstrap_added_count = 0
        bootstrap_start = start_timing()
        if not initial_mastered and args.bootstrap_new_examples_per_digit > 0:
            bootstrap_new: List[AdditionExample] = []
            for digits in range(args.initial_min_digits, args.initial_max_digits + 1):
                max_pairs = max_unique_pairs_for_digits(digits)
                occupied_count = count_occupied_for_digits(occupied, digits)
                remaining = max(0, max_pairs - occupied_count)
                sample_count = min(args.bootstrap_new_examples_per_digit, remaining)
                if sample_count <= 0:
                    continue
                try:
                    bootstrap_new.extend(
                        sample_unique_examples(
                            digits=digits,
                            count=sample_count,
                            rng=rng,
                            occupied=occupied,
                            allow_carry=True,
                        )
                    )
                except RuntimeError:
                    continue
            if bootstrap_new:
                seed_examples = dedupe_examples(list(seed_examples) + bootstrap_new)
                bootstrap_added_count = len(bootstrap_new)
        stop_timing("bootstrap_seed", bootstrap_start, round_timing)

        train_examples = merge_seed_and_pseudo(seed_examples, pseudo_examples)

        if not initial_mastered:
            epochs = max(args.num_epochs, args.initial_bootstrapping_epochs)
        elif just_grew:
            epochs = max(args.num_epochs, args.growth_warmup_epochs)
        else:
            epochs = args.num_epochs

        train_start = start_timing()
        train_loss = train_one_round(
            model,
            train_examples,
            tokenizer,
            device=device,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            num_epochs=epochs,
        )
        stop_timing("train", train_start, round_timing)

        frontier_validation = filter_examples_by_digits(validation, max_digits=round_frontier_digits)
        initial_eval_start = start_timing()
        initial_val_acc, initial_val_per_digit, _ = evaluate_exact_autoregressive(
            model,
            validation_initial,
            tokenizer,
            device=device,
            batch_size=args.eval_batch_size,
            use_true_targets=True,
        )
        stop_timing("eval_initial", initial_eval_start, round_timing)
        frontier_eval_start = start_timing()
        frontier_val_acc, frontier_val_per_digit, _ = evaluate_exact_autoregressive(
            model,
            frontier_validation,
            tokenizer,
            device=device,
            batch_size=args.eval_batch_size,
            use_true_targets=True,
        )
        stop_timing("eval_frontier", frontier_eval_start, round_timing)
        full_eval_due = args.full_eval_interval > 0 and ((round_idx + 1) % args.full_eval_interval == 0)
        if full_eval_due:
            full_eval_start = start_timing()
            full_val_acc, _, _ = evaluate_exact_autoregressive(
                model,
                validation,
                tokenizer,
                device=device,
                batch_size=args.eval_batch_size,
                use_true_targets=True,
            )
            stop_timing("eval_full", full_eval_start, round_timing)
            latest_full_val_acc = full_val_acc
        else:
            full_val_acc = latest_full_val_acc
        train_eval_start = start_timing()
        train_true_acc, _, train_success_map = evaluate_exact_autoregressive(
            model,
            train_examples,
            tokenizer,
            device=device,
            batch_size=args.eval_batch_size,
            use_true_targets=True,
        )
        stop_timing("eval_train", train_eval_start, round_timing)
        successful_train_examples = select_successful_examples(train_examples, train_success_map)
        seed_examples = dedupe_examples(list(seed_examples) + successful_train_examples)
        if (
            initial_mastered
            and args.seed_retention_per_digit > 0
            and seed_retention_max_digits >= args.initial_min_digits
        ):
            seed_examples, seed_retention_pruned_count = downsample_seed_for_learned_levels(
                seed_examples,
                max_learned_digits=seed_retention_max_digits,
                retain_per_digit=args.seed_retention_per_digit,
                rng=rng,
            )

        component_true_start = start_timing()
        component_true_acc, _, _ = evaluate_exact_autoregressive(
            model,
            initial_train,
            tokenizer,
            device=device,
            batch_size=args.eval_batch_size,
            use_true_targets=True,
        )
        stop_timing("eval_components_true", component_true_start, round_timing)
        max_component_target_len = max(len(ex.true_target()) for ex in initial_train)
        component_pred_start = start_timing()
        component_prediction_targets = generate_prediction_map(
            model,
            initial_train,
            tokenizer,
            device=device,
            batch_size=args.eval_batch_size,
            max_new_tokens=max_component_target_len + 2,
        )
        stop_timing("predict_components", component_pred_start, round_timing)
        component_pred_coverage = len(component_prediction_targets) / max(len(initial_train), 1)

        if initial_val_acc >= args.initial_mastery_threshold:
            initial_mastery_streak += 1
        else:
            initial_mastery_streak = 0
        if initial_mastery_streak >= args.initial_mastery_patience:
            initial_mastered = True

        improved = frontier_val_acc > (stage_best_frontier_val + args.saturation_delta)
        if improved:
            stage_best_frontier_val = frontier_val_acc
            stale_rounds = 0
            stall_bootstrap_events_in_frontier = 0
        else:
            stale_rounds += 1

        can_expand_frontier = frontier_val_acc >= args.frontier_expand_threshold
        edge_min_digits = max(
            args.initial_max_digits + 1,
            round_frontier_digits - round_frontier_span_digits + 1,
        )
        edge_max_digits = round_frontier_digits
        stall_bootstrap_possible = edge_max_digits >= edge_min_digits
        edge_digit_count = max(0, edge_max_digits - edge_min_digits + 1)
        stall_bootstrap_added_count = 0
        stall_bootstrap_attempted = False
        stall_bootstrap_start = start_timing()
        if (
            initial_mastered
            and not can_expand_frontier
            and args.stall_bootstrap_composed_per_digit > 0
            and stale_rounds >= args.stall_bootstrap_patience
            and stall_bootstrap_possible
            and stall_bootstrap_events_in_frontier < args.stall_bootstrap_max_events_per_edge_digit
        ):
            stall_bootstrap_attempted = True
            new_pool, new_component_map = extend_composed_pool(
                component_examples=initial_train,
                min_digits=edge_min_digits,
                max_digits=edge_max_digits,
                per_digit_count=args.stall_bootstrap_composed_per_digit,
                rng=rng,
                occupied=occupied,
                boundary_mode=args.composed_boundary_mode,
            )
            if new_pool:
                composed_pool.extend(new_pool)
                component_map.update(new_component_map)
                append_examples_jsonl(composed_pool_augments_path, new_pool)
                append_component_map_jsonl(composed_component_map_augments_path, new_component_map)
                stall_bootstrap_added_count = len(new_pool)
            stall_bootstrap_events_in_frontier += 1
            stale_rounds = 0
        stop_timing("stall_bootstrap", stall_bootstrap_start, round_timing)

        allow_growth = args.allow_growth_before_initial_mastery or initial_mastered
        growth_triggered = False
        rounds_this_frontier = rounds_in_current_frontier + 1
        stall_bootstrap_exhausted = (
            stall_bootstrap_events_in_frontier >= args.stall_bootstrap_max_events_per_edge_digit
        )
        bootstrap_escalation_ready = (
            args.stall_bootstrap_composed_per_digit <= 0
            or can_expand_frontier
            or not stall_bootstrap_possible
            or stall_bootstrap_exhausted
        )
        if (
            stage_idx + 1 < len(stage_cfgs)
            and allow_growth
            and not can_expand_frontier
            and bootstrap_escalation_ready
            and (round_idx + 1) >= args.min_rounds_before_growth
            and rounds_this_frontier >= args.min_rounds_per_frontier_before_growth
            and stale_rounds >= args.saturation_patience
            and frontier_val_acc < args.frontier_mastery_threshold
        ):
            growth_seed = successful_train_examples
            if len(growth_seed) < args.min_growth_seed_examples:
                growth_seed = seed_examples
            pending_growth_seed = dedupe_examples(growth_seed)
            growth_pending = True
            growth_triggered = True
            stale_rounds = 0

        next_frontier_max_digits = frontier_max_digits
        next_frontier_span_digits = frontier_span_digits
        applied_expand_span = 0
        if (
            not growth_triggered
            and initial_mastered
            and expansion_step < planned_expansion_rounds
            and can_expand_frontier
        ):
            next_frontier_max_digits = frontier_targets[expansion_step]
            applied_expand_span = frontier_spans[expansion_step]
            next_frontier_span_digits = applied_expand_span
            expansion_step += 1
        frontier_expanded = next_frontier_max_digits > round_frontier_digits
        active_pseudo_digit_count = max(0, next_frontier_max_digits - args.initial_max_digits)
        min_unique_candidate_required_total = args.min_unique_candidate_per_digit * active_pseudo_digit_count

        unique_candidate_component_gate_passed = (
            component_pred_coverage >= args.min_unique_candidate_min_component_acc
        )
        unique_candidate_enforced = (
            min_unique_candidate_required_total > 0 and unique_candidate_component_gate_passed
        )
        unique_candidate_augment_events = 0
        unique_candidate_augment_added = 0
        unique_candidate_target_met = (
            min_unique_candidate_required_total <= 0 or not unique_candidate_enforced
        )
        pseudo_build_start = start_timing()
        if next_frontier_max_digits > args.initial_max_digits:
            next_pseudo_candidates, pseudo_stats = build_composed_pseudo_dataset(
                composed_pool,
                component_map,
                component_prediction_targets,
                component_example_lookup,
                target_max_digits=next_frontier_max_digits,
                max_pseudo_per_round_per_digit=args.max_pseudo_per_round_per_digit,
                rng=rng,
                boundary_mode=args.composed_boundary_mode,
            )
            next_pseudo, pseudo_selection_stats = select_next_round_pseudo(
                next_pseudo_candidates,
                seed_examples=seed_examples,
                new_unique_quota_per_digit=args.new_pseudo_unique_quota_per_digit,
                rng=rng,
            )
            if unique_candidate_enforced:
                augment_per_digit = args.unique_candidate_augment_per_digit
                if augment_per_digit <= 0:
                    if args.stall_bootstrap_composed_per_digit > 0:
                        augment_per_digit = args.stall_bootstrap_composed_per_digit
                    else:
                        augment_per_digit = args.composed_train_per_digit
                augment_min_digits = max(
                    args.initial_max_digits + 1,
                    next_frontier_max_digits - next_frontier_span_digits + 1,
                )
                augment_max_digits = next_frontier_max_digits
                successful_components = [
                    ex for ex in initial_train if ex.key() in component_prediction_targets
                ]
                seed_key_set = {ex.key() for ex in seed_examples}
                while (
                    pseudo_selection_stats["unique_candidate_total"] < min_unique_candidate_required_total
                    and unique_candidate_augment_events < args.unique_candidate_augment_max_events_per_edge_digit
                ):
                    remaining_unique_needed = (
                        min_unique_candidate_required_total - pseudo_selection_stats["unique_candidate_total"]
                    )
                    target_unique = max(
                        remaining_unique_needed,
                        augment_per_digit * max(1, augment_max_digits - augment_min_digits + 1),
                    )
                    if (
                        augment_per_digit <= 0
                        or augment_max_digits < augment_min_digits
                        or len(successful_components) < 2
                    ):
                        break
                    extra_pool, extra_component_map = sample_unique_composed_against_keys(
                        component_examples=successful_components,
                        min_digits=augment_min_digits,
                        max_digits=augment_max_digits,
                        target_unique_count=target_unique,
                        rng=rng,
                        occupied=occupied,
                        excluded_keys=seed_key_set,
                        boundary_mode=args.composed_boundary_mode,
                    )
                    unique_candidate_augment_events += 1
                    if not extra_pool:
                        break
                    composed_pool.extend(extra_pool)
                    component_map.update(extra_component_map)
                    append_examples_jsonl(composed_pool_augments_path, extra_pool)
                    append_component_map_jsonl(composed_component_map_augments_path, extra_component_map)
                    unique_candidate_augment_added += len(extra_pool)
                    next_pseudo_candidates, pseudo_stats = build_composed_pseudo_dataset(
                        composed_pool,
                        component_map,
                        component_prediction_targets,
                        component_example_lookup,
                        target_max_digits=next_frontier_max_digits,
                        max_pseudo_per_round_per_digit=args.max_pseudo_per_round_per_digit,
                        rng=rng,
                        boundary_mode=args.composed_boundary_mode,
                    )
                    next_pseudo, pseudo_selection_stats = select_next_round_pseudo(
                        next_pseudo_candidates,
                        seed_examples=seed_examples,
                        new_unique_quota_per_digit=args.new_pseudo_unique_quota_per_digit,
                        rng=rng,
                    )
                unique_candidate_target_met = (
                    pseudo_selection_stats["unique_candidate_total"] >= min_unique_candidate_required_total
                )
                if not unique_candidate_target_met and args.min_unique_candidate_hard_fail:
                    raise RuntimeError(
                        "Unable to satisfy min_unique_candidate_required_total="
                        f"{min_unique_candidate_required_total} "
                        f"(got {pseudo_selection_stats['unique_candidate_total']})."
                    )
        else:
            next_pseudo = []
            pseudo_stats = {"candidate_total": 0, "retained_total": 0, "missing_total": 0, "invalid_total": 0}
            pseudo_selection_stats = {
                "new_unique_quota_per_digit": args.new_pseudo_unique_quota_per_digit,
                "candidate_dedup_total": 0,
                "unique_candidate_total": 0,
                "overlap_with_seed_total": 0,
                "selected_total": 0,
                "selected_per_digit": {},
            }
        stop_timing("pseudo_build", pseudo_build_start, round_timing)
        pseudo_stats = {
            **pseudo_stats,
            **pseudo_selection_stats,
            "min_unique_candidate_per_digit": args.min_unique_candidate_per_digit,
            "min_unique_candidate_required_total": min_unique_candidate_required_total,
            "min_unique_candidate_min_component_acc": args.min_unique_candidate_min_component_acc,
            "unique_candidate_component_gate_passed": unique_candidate_component_gate_passed,
            "unique_candidate_enforced": unique_candidate_enforced,
            "unique_candidate_target_met": unique_candidate_target_met,
            "unique_candidate_augment_events": unique_candidate_augment_events,
            "unique_candidate_augment_added_examples": unique_candidate_augment_added,
        }

        pseudo_examples = next_pseudo
        frontier_max_digits = next_frontier_max_digits
        if frontier_expanded:
            frontier_span_digits = next_frontier_span_digits
            # Frontier changed; reset saturation tracking for the new regime.
            stage_best_frontier_val = -1.0
            stale_rounds = 0
            rounds_in_current_frontier = 0
            stall_bootstrap_events_in_frontier = 0
        else:
            rounds_in_current_frontier = rounds_this_frontier
        if round_frontier_digits >= final_max_digits:
            trained_at_final_frontier = True

        elapsed = time.time() - round_start
        timing_accounted = sum(round_timing.values())
        timing_other = max(0.0, elapsed - timing_accounted)
        record: Dict[str, object] = {
            "round_index": round_idx,
            "stage_index": stage_idx,
            "stage_config": asdict(stage_cfgs[stage_idx]),
            "train_example_count": len(train_examples),
            "seed_example_count": len(seed_examples),
            "bootstrap_added_examples": bootstrap_added_count,
            "pseudo_used_count": pseudo_used_count,
            "pseudo_next_count": len(pseudo_examples),
            "pseudo_unique_candidate_total": pseudo_selection_stats["unique_candidate_total"],
            "pseudo_overlap_with_seed_total": pseudo_selection_stats["overlap_with_seed_total"],
            "new_pseudo_unique_quota_per_digit": args.new_pseudo_unique_quota_per_digit,
            "min_unique_candidate_per_digit": args.min_unique_candidate_per_digit,
            "min_unique_candidate_required_total": min_unique_candidate_required_total,
            "min_unique_candidate_min_component_acc": args.min_unique_candidate_min_component_acc,
            "unique_candidate_component_gate_passed": unique_candidate_component_gate_passed,
            "unique_candidate_enforced": unique_candidate_enforced,
            "unique_candidate_target_met": unique_candidate_target_met,
            "unique_candidate_augment_events": unique_candidate_augment_events,
            "unique_candidate_augment_added_examples": unique_candidate_augment_added,
            "seed_retention_per_digit": args.seed_retention_per_digit,
            "seed_retention_max_digits": seed_retention_max_digits,
            "seed_retention_pruned_examples": seed_retention_pruned_count,
            "train_loss": train_loss,
            "initial_validation_accuracy": initial_val_acc,
            "initial_validation_per_digit_accuracy": {str(k): v for k, v in sorted(initial_val_per_digit.items())},
            "frontier_validation_accuracy": frontier_val_acc,
            "frontier_validation_per_digit_accuracy": {str(k): v for k, v in sorted(frontier_val_per_digit.items())},
            "full_validation_accuracy": full_val_acc,
            "full_validation_evaluated_this_round": full_eval_due,
            "frontier_max_digits": round_frontier_digits,
            "frontier_span_digits": round_frontier_span_digits,
            "edge_digit_count": edge_digit_count,
            "next_frontier_max_digits": frontier_max_digits,
            "next_frontier_span_digits": frontier_span_digits,
            "frontier_expanded": frontier_expanded,
            "applied_expand_span": applied_expand_span,
            "frontier_expand_threshold": args.frontier_expand_threshold,
            "frontier_expand_gate_passed": can_expand_frontier,
            "expansion_step": expansion_step,
            "planned_expansion_rounds": planned_expansion_rounds,
            "rounds_in_current_frontier": rounds_in_current_frontier,
            "stall_bootstrap_attempted": stall_bootstrap_attempted,
            "stall_bootstrap_possible": stall_bootstrap_possible,
            "stall_bootstrap_added_examples": stall_bootstrap_added_count,
            "stall_bootstrap_events_in_frontier": stall_bootstrap_events_in_frontier,
            "stall_bootstrap_exhausted": stall_bootstrap_exhausted,
            "composed_pool_size": len(composed_pool),
            "initial_mastered": initial_mastered,
            "initial_mastery_streak": initial_mastery_streak,
            "train_true_accuracy": train_true_acc,
            "successful_train_examples": len(successful_train_examples),
            "component_true_accuracy": component_true_acc,
            "component_prediction_coverage": component_pred_coverage,
            "component_prediction_count": len(component_prediction_targets),
            "pseudo_generation_stats": pseudo_stats,
            "growth_triggered": growth_triggered,
            "stale_rounds_in_stage": stale_rounds,
            "elapsed_seconds": elapsed,
        }
        if args.log_round_breakdown:
            record["timing_breakdown_seconds"] = {
                key: value for key, value in sorted(round_timing.items())
            }
            record["timing_accounted_seconds"] = timing_accounted
            record["timing_other_seconds"] = timing_other
        round_records.append(record)

        print(
            "[ROUND {idx}] stage={stage} frontier<={frontier} train={train} seed={seed} pseudo_used={pseudo_used} pseudo_next={pseudo} unique_pseudo={unique_pseudo} "
            "bootstrap_added={bootstrap} stall_bootstrap_added={stall_bootstrap} "
            "seed_pruned={seed_pruned} "
            "init_val={init_val:.4f} frontier_val={front_val:.4f} full_val={full_val:.4f} "
            "train_true={train_true:.4f} comp_true={comp_true:.4f} comp_pred_cov={comp_pred_cov:.4f} init_mastered={init_mastered} "
            "growth={growth} elapsed={elapsed:.1f}s".format(
                idx=round_idx,
                stage=stage_idx,
                frontier=round_frontier_digits,
                train=len(train_examples),
                seed=len(seed_examples),
                pseudo_used=pseudo_used_count,
                pseudo=len(pseudo_examples),
                unique_pseudo=pseudo_selection_stats["unique_candidate_total"],
                bootstrap=bootstrap_added_count,
                stall_bootstrap=stall_bootstrap_added_count,
                seed_pruned=seed_retention_pruned_count,
                init_val=initial_val_acc,
                front_val=frontier_val_acc,
                full_val=full_val_acc,
                train_true=train_true_acc,
                comp_true=component_true_acc,
                comp_pred_cov=component_pred_coverage,
                init_mastered=initial_mastered,
                growth=growth_triggered,
                elapsed=elapsed,
            ),
            flush=True,
        )
        if args.log_round_breakdown:
            ordered_timing_keys = [
                "bootstrap_seed",
                "train",
                "eval_initial",
                "eval_frontier",
                "eval_full",
                "eval_train",
                "eval_components_true",
                "predict_components",
                "stall_bootstrap",
                "pseudo_build",
            ]
            timing_summary = " ".join(
                f"{key}={round_timing.get(key, 0.0):.2f}s" for key in ordered_timing_keys
            )
            print(
                f"[TIMING {round_idx}] {timing_summary} other={timing_other:.2f}s total={elapsed:.2f}s",
                flush=True,
            )

        if args.save_models:
            stage_dir = output_dir / "models" / f"stage_{stage_idx:02d}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = stage_dir / f"round_{round_idx:03d}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "stage_index": stage_idx,
                    "stage_config": asdict(stage_cfgs[stage_idx]),
                    "round_index": round_idx,
                    "frontier_max_digits": frontier_max_digits,
                    "expansion_step": expansion_step,
                    "initial_mastered": initial_mastered,
                },
                checkpoint_path,
            )

        round_idx += 1
        if continuation_save_enabled and (round_idx % args.continuation_save_interval == 0):
            save_continuation_state(round_idx)
        flush_summary(stop_reason)
        if stop_requested:
            if signal_stop_reason is not None:
                stop_reason = signal_stop_reason
            print("[WARN] Stop requested by signal; exiting after round boundary.", flush=True)
            save_continuation_state(round_idx)
            flush_summary(stop_reason)
            break
        if (
            trained_at_final_frontier
            and initial_mastered
            and expansion_step >= planned_expansion_rounds
            and not growth_pending
        ):
            stop_reason = "reached_final_frontier"
            print("[INFO] Reached final frontier with initial mastery satisfied; stopping.", flush=True)
            save_continuation_state(round_idx)
            flush_summary(stop_reason)
            break

    signal.signal(signal.SIGTERM, previous_sigterm_handler)
    signal.signal(signal.SIGINT, previous_sigint_handler)

    if stop_reason == "max_total_rounds_reached":
        print(
            "[WARN] Stopped due to max_total_rounds before satisfying final-frontier stop condition. "
            f"initial_mastered={initial_mastered} expansion_step={expansion_step}/{planned_expansion_rounds}",
            flush=True,
        )
    elif stop_reason.startswith("signal_"):
        print(f"[WARN] Stopped due to {stop_reason}.", flush=True)

    save_continuation_state(round_idx)
    flush_summary(stop_reason)
    print(f"[INFO] Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
