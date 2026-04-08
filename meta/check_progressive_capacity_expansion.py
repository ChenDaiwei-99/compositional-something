#!/usr/bin/env python3
"""Smoke-check progressive depth and depth+width capacity expansion."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from meta.train_meta_self_improvement_rope import (
    AdditionExample,
    AdditionTokenizer,
    DEFAULT_LEGACY_STAGE_CONFIGS,
    DEFAULT_PROGRESSIVE_DEPTH_STAGE_CONFIGS,
    ModelStageConfig,
    build_replication_assignment,
    expand_model_for_capacity_growth,
    generate_prediction_map,
    instantiate_model,
    normalize_capacity_growth_scheme,
    parse_stage_configs,
    sample_unique_examples,
    validate_capacity_growth_transition,
)

ExampleKey = Tuple[int, int, int]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify progressive depth / depth+width stage expansion behavior."
    )
    parser.add_argument(
        "--capacity-growth-scheme",
        type=str,
        choices=("progressive", "progressive_depth", "progressive_depth_width"),
        default="progressive_depth",
    )
    parser.add_argument("--stage-configs", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--eval-min-digits", type=int, default=2)
    parser.add_argument("--eval-max-digits", type=int, default=8)
    parser.add_argument("--eval-per-digit", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=160)
    parser.add_argument(
        "--max-width-mismatch-rate",
        type=float,
        default=0.05,
        help="Maximum tolerated decoded-prediction mismatch rate for progressive_depth_width smoke-checks.",
    )
    parser.add_argument("--ffn-mult", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rope-base", type=float, default=10_000.0)
    parser.add_argument("--context-window", type=int, default=512)
    return parser.parse_args(argv)


def default_stage_configs_for_scheme(scheme: str) -> str:
    if scheme == "progressive_depth":
        return DEFAULT_PROGRESSIVE_DEPTH_STAGE_CONFIGS
    if scheme == "progressive_depth_width":
        return DEFAULT_LEGACY_STAGE_CONFIGS
    raise ValueError(f"Unsupported scheme: {scheme!r}")


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def build_eval_examples(args: argparse.Namespace, rng: random.Random) -> List[AdditionExample]:
    if args.eval_min_digits < 1:
        raise ValueError("eval-min-digits must be >= 1.")
    if args.eval_max_digits < args.eval_min_digits:
        raise ValueError("eval-max-digits must be >= eval-min-digits.")
    if args.eval_per_digit <= 0:
        raise ValueError("eval-per-digit must be positive.")

    occupied: set[ExampleKey] = set()
    examples: List[AdditionExample] = []
    for digits in range(args.eval_min_digits, args.eval_max_digits + 1):
        examples.extend(
            sample_unique_examples(
                digits=digits,
                count=args.eval_per_digit,
                rng=rng,
                occupied=occupied,
                allow_carry=True,
            )
        )
    rng.shuffle(examples)
    return examples


def compare_prediction_maps(
    examples: Sequence[AdditionExample],
    before: dict[ExampleKey, str],
    after: dict[ExampleKey, str],
) -> List[Tuple[AdditionExample, str | None, str | None]]:
    mismatches: List[Tuple[AdditionExample, str | None, str | None]] = []
    for ex in examples:
        key = ex.key()
        before_pred = before.get(key)
        after_pred = after.get(key)
        if before_pred != after_pred:
            mismatches.append((ex, before_pred, after_pred))
    return mismatches


def run_backward(model: torch.nn.Module, tokenizer: AdditionTokenizer, device: torch.device) -> None:
    model.zero_grad(set_to_none=True)
    model.train()
    input_ids = torch.randint(0, tokenizer.vocab_size, (4, 16), device=device)
    labels = torch.randint(0, tokenizer.vocab_size, (4, 16), device=device)
    _, loss = model(input_ids, labels)
    if loss is None:
        raise RuntimeError("Expected language-model loss during gradient check.")
    loss.backward()


def check_new_block_gradients(model: torch.nn.Module) -> dict[str, float]:
    block = model.blocks[-1]
    attn_grad = float(block.attn.out_proj.weight.grad.abs().sum().item())
    mlp_out = block.mlp[2]
    if not isinstance(mlp_out, torch.nn.Linear):
        raise TypeError("Expected the MLP output projection to be an nn.Linear.")
    mlp_grad = float(mlp_out.weight.grad.abs().sum().item())
    return {
        "new_block_attn_out_proj_grad_sum": attn_grad,
        "new_block_mlp_out_grad_sum": mlp_grad,
    }


def check_widened_gradients(model: torch.nn.Module, prev_cfg: ModelStageConfig) -> dict[str, float]:
    block = model.blocks[min(prev_cfg.n_layers - 1, len(model.blocks) - 1)]
    new_d_model = model.tok_emb.embedding_dim
    old_n_heads = prev_cfg.n_heads
    new_n_heads = block.attn.n_heads
    old_head_dim = prev_cfg.d_model // prev_cfg.n_heads
    new_head_dim = block.attn.head_dim

    qkv_grad = block.attn.qkv.weight.grad.view(3, new_n_heads, new_head_dim, new_d_model)
    out_proj_grad = block.attn.out_proj.weight.grad.view(new_d_model, new_n_heads, new_head_dim)
    fc1 = block.mlp[0]
    fc2 = block.mlp[2]
    if not isinstance(fc1, torch.nn.Linear) or not isinstance(fc2, torch.nn.Linear):
        raise TypeError("Expected MLP projections to be nn.Linear.")
    hidden_old = int(fc1.out_features * (prev_cfg.d_model / new_d_model))
    hidden_old = max(1, hidden_old)

    extra_qkv_heads = float(qkv_grad[:, old_n_heads:, :, : prev_cfg.d_model].abs().sum().item()) if new_n_heads > old_n_heads else 0.0
    extra_qkv_head_dims = (
        float(qkv_grad[:, :old_n_heads, old_head_dim:, : prev_cfg.d_model].abs().sum().item())
        if new_head_dim > old_head_dim
        else 0.0
    )
    extra_out_proj_heads = (
        float(out_proj_grad[: prev_cfg.d_model, old_n_heads:, :].abs().sum().item())
        if new_n_heads > old_n_heads
        else 0.0
    )
    extra_out_proj_head_dims = (
        float(out_proj_grad[: prev_cfg.d_model, :old_n_heads, old_head_dim:].abs().sum().item())
        if new_head_dim > old_head_dim
        else 0.0
    )
    extra_fc1_rows = (
        float(fc1.weight.grad[hidden_old:, : prev_cfg.d_model].abs().sum().item())
        if fc1.out_features > hidden_old
        else 0.0
    )
    extra_fc2_cols = (
        float(fc2.weight.grad[: prev_cfg.d_model, hidden_old:].abs().sum().item())
        if fc2.in_features > hidden_old
        else 0.0
    )
    return {
        "extra_qkv_heads_grad_sum": extra_qkv_heads,
        "extra_qkv_head_dims_grad_sum": extra_qkv_head_dims,
        "extra_out_proj_heads_grad_sum": extra_out_proj_heads,
        "extra_out_proj_head_dims_grad_sum": extra_out_proj_head_dims,
        "extra_mlp_fc1_rows_grad_sum": extra_fc1_rows,
        "extra_mlp_fc2_cols_grad_sum": extra_fc2_cols,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.capacity_growth_scheme = normalize_capacity_growth_scheme(args.capacity_growth_scheme)
    if not args.stage_configs:
        args.stage_configs = default_stage_configs_for_scheme(args.capacity_growth_scheme)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    device = resolve_device(args.device)

    tokenizer = AdditionTokenizer()
    stage_cfgs = parse_stage_configs(args.stage_configs)
    if len(stage_cfgs) < 2:
        raise ValueError("Need at least two stage configs to test progressive expansion transitions.")

    for idx in range(len(stage_cfgs) - 1):
        try:
            validate_capacity_growth_transition(args.capacity_growth_scheme, stage_cfgs[idx], stage_cfgs[idx + 1])
        except ValueError as exc:
            print(
                f"[FAIL] Incompatible transition {idx}->{idx + 1}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return 1

    eval_examples = build_eval_examples(args, rng)
    max_new_tokens = max(len(ex.true_target()) for ex in eval_examples) + 2
    model = instantiate_model(tokenizer, stage_cfgs[0], args, device)
    print(
        "[INFO] Running {} expansion smoke-check: transitions={} eval_examples={} device={}".format(
            args.capacity_growth_scheme,
            len(stage_cfgs) - 1,
            len(eval_examples),
            device,
        ),
        flush=True,
    )

    for idx in range(len(stage_cfgs) - 1):
        prev_cfg = stage_cfgs[idx]
        next_cfg = stage_cfgs[idx + 1]
        before = generate_prediction_map(
            model,
            eval_examples,
            tokenizer,
            device=device,
            batch_size=args.eval_batch_size,
            max_new_tokens=max_new_tokens,
        )
        model = expand_model_for_capacity_growth(
            model,
            next_cfg,
            args,
            tokenizer,
            device,
        )
        after = generate_prediction_map(
            model,
            eval_examples,
            tokenizer,
            device=device,
            batch_size=args.eval_batch_size,
            max_new_tokens=max_new_tokens,
        )

        mismatches = compare_prediction_maps(eval_examples, before, after)
        run_backward(model, tokenizer, device)

        gradient_stats: dict[str, float] = {}
        if args.capacity_growth_scheme == "progressive_depth":
            if mismatches:
                print(
                    "[FAIL] Transition {}->{} mismatches: {}/{}".format(
                        idx,
                        idx + 1,
                        len(mismatches),
                        len(eval_examples),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                for ex, before_pred, after_pred in mismatches[:10]:
                    print(
                        f"  digits={ex.digits} prompt={ex.prompt()} before={before_pred!r} after={after_pred!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                return 1
            if next_cfg.n_layers > prev_cfg.n_layers:
                gradient_stats.update(check_new_block_gradients(model))
                if (
                    gradient_stats["new_block_attn_out_proj_grad_sum"] <= 0.0
                    or gradient_stats["new_block_mlp_out_grad_sum"] <= 0.0
                ):
                    print(
                        f"[FAIL] Transition {idx}->{idx + 1} produced a non-trainable identity block: {gradient_stats}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 1
            print(
                "[PASS] Transition {}->{} preserved decoded predictions{}.".format(
                    idx,
                    idx + 1,
                    f" and new-block gradients {gradient_stats}" if gradient_stats else "",
                ),
                flush=True,
            )
            continue

        gradient_stats.update(check_widened_gradients(model, prev_cfg))
        if next_cfg.n_layers > prev_cfg.n_layers:
            gradient_stats.update(check_new_block_gradients(model))
        mismatch_rate = len(mismatches) / max(len(eval_examples), 1)
        if mismatch_rate > args.max_width_mismatch_rate:
            print(
                "[FAIL] Transition {}->{} widened too destructively: mismatches={}/{} rate={:.3f} threshold={:.3f}".format(
                    idx,
                    idx + 1,
                    len(mismatches),
                    len(eval_examples),
                    mismatch_rate,
                    args.max_width_mismatch_rate,
                ),
                file=sys.stderr,
                flush=True,
            )
            for ex, before_pred, after_pred in mismatches[:10]:
                print(
                    f"  digits={ex.digits} prompt={ex.prompt()} before={before_pred!r} after={after_pred!r}",
                    file=sys.stderr,
                    flush=True,
                )
            return 1
        active_widened_grad = (
            gradient_stats["extra_qkv_heads_grad_sum"]
            + gradient_stats["extra_qkv_head_dims_grad_sum"]
            + gradient_stats["extra_out_proj_heads_grad_sum"]
            + gradient_stats["extra_out_proj_head_dims_grad_sum"]
            + gradient_stats["extra_mlp_fc1_rows_grad_sum"]
            + gradient_stats["extra_mlp_fc2_cols_grad_sum"]
        )
        if active_widened_grad <= 0.0:
            print(
                f"[FAIL] Transition {idx}->{idx + 1} produced inactive widened parameters: {gradient_stats}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        print(
            "[PASS] Transition {}->{} widened successfully: mismatches={}/{} rate={:.3f} gradients={}.".format(
                idx,
                idx + 1,
                len(mismatches),
                len(eval_examples),
                mismatch_rate,
                gradient_stats,
            ),
            flush=True,
        )

    print("[PASS] Progressive capacity expansion smoke-check succeeded.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
