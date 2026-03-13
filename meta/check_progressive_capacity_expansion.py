#!/usr/bin/env python3
"""Smoke-check that progressive expansion preserves decoded predictions."""
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
    DEFAULT_PROGRESSIVE_STAGE_CONFIGS,
    expand_model_progressive,
    generate_prediction_map,
    instantiate_model,
    parse_stage_configs,
    sample_unique_examples,
    validate_progressive_transition,
)

ExampleKey = Tuple[int, int, int]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify exact decoded-prediction equivalence across progressive stage expansions."
    )
    parser.add_argument("--stage-configs", type=str, default=DEFAULT_PROGRESSIVE_STAGE_CONFIGS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--eval-min-digits", type=int, default=2)
    parser.add_argument("--eval-max-digits", type=int, default=8)
    parser.add_argument("--eval-per-digit", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=160)
    parser.add_argument("--ffn-mult", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rope-base", type=float, default=10_000.0)
    parser.add_argument("--context-window", type=int, default=512)
    return parser.parse_args(argv)


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


def check_new_block_gradients(model: torch.nn.Module, tokenizer: AdditionTokenizer, device: torch.device) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    model.train()
    input_ids = torch.randint(0, tokenizer.vocab_size, (4, 16), device=device)
    labels = torch.randint(0, tokenizer.vocab_size, (4, 16), device=device)
    _, loss = model(input_ids, labels)
    if loss is None:
        raise RuntimeError("Expected language-model loss during gradient check.")
    loss.backward()

    block = model.blocks[-1]
    attn_grad = float(block.attn.out_proj.weight.grad.abs().sum().item())
    mlp_out = block.mlp[2]
    if not isinstance(mlp_out, torch.nn.Linear):
        raise TypeError("Expected the MLP output projection to be an nn.Linear.")
    mlp_grad = float(mlp_out.weight.grad.abs().sum().item())
    return {
        "attn_out_proj_grad_sum": attn_grad,
        "mlp_out_grad_sum": mlp_grad,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
            validate_progressive_transition(stage_cfgs[idx], stage_cfgs[idx + 1])
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
        "[INFO] Running progressive expansion equivalence check: transitions={} eval_examples={} device={}".format(
            len(stage_cfgs) - 1,
            len(eval_examples),
            device,
        ),
        flush=True,
    )

    for idx in range(len(stage_cfgs) - 1):
        before = generate_prediction_map(
            model,
            eval_examples,
            tokenizer,
            device=device,
            batch_size=args.eval_batch_size,
            max_new_tokens=max_new_tokens,
        )
        model = expand_model_progressive(
            model,
            stage_cfgs[idx + 1],
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
        gradient_stats = check_new_block_gradients(model, tokenizer, device)
        if gradient_stats["attn_out_proj_grad_sum"] <= 0.0 or gradient_stats["mlp_out_grad_sum"] <= 0.0:
            print(
                "[FAIL] Transition {}->{} produced a non-trainable identity block: {}".format(
                    idx,
                    idx + 1,
                    gradient_stats,
                ),
                file=sys.stderr,
                flush=True,
            )
            return 1
        print(
            "[PASS] Transition {}->{} preserved decoded predictions and new-block gradients {}.".format(
                idx,
                idx + 1,
                gradient_stats,
            ),
            flush=True,
        )

    print("[PASS] Progressive capacity expansion smoke-check succeeded.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
