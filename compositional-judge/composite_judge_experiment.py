#!/usr/bin/env python3
"""
Prototype experiment: compare Qwen3-0.6B judge accuracy on individual arithmetic
subtasks versus a composed multi-step reasoning task.
"""
from __future__ import annotations

import argparse
import importlib.machinery
import math
import random
import sys
import types
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from vllm import LLM, SamplingParams

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **kwargs: x  # type: ignore


# Avoid optional deepspeed dependency trying to probe CUDA binaries when using HF fallbacks.
if "deepspeed" not in sys.modules:
    ds_stub = types.ModuleType("deepspeed")
    ds_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
    sys.modules["deepspeed"] = ds_stub


ADDITION_TEMPLATE = """You are evaluating whether the assistant's answer is correct. About half of the answers are correct. Respond with exactly one word: Correct or Incorrect.

Examples:
Problem: Compute 3 + 5.
Assistant answer: 8.
Verdict: Correct

Problem: Compute 6 + 2.
Assistant answer: 11.
Verdict: Incorrect

Now evaluate:
Problem: Compute {a} + {b}.
Assistant answer: {answer}.
Respond Correct only if the assistant answer equals {a} + {b}; otherwise respond Incorrect.
"""

SUBTRACTION_TEMPLATE = """You are evaluating whether the assistant's answer is correct. About half of the answers are correct. Respond with exactly one word: Correct or Incorrect.

Examples:
Problem: Compute 9 - 4.
Assistant answer: 5.
Verdict: Correct

Problem: Compute 12 - 7.
Assistant answer: 6.
Verdict: Incorrect

Now evaluate:
Problem: Compute {c} - {d}.
Assistant answer: {answer}.
Respond Correct only if the assistant answer equals {c} - {d}; otherwise respond Incorrect.
"""

COMPOSITE_TEMPLATE = """You are evaluating a multi-step arithmetic solution. Approximately half of the solutions are correct. Respond with exactly one word: Correct or Incorrect. Output Correct only if every stated computation and the final answer are accurate.

Examples:
Problem steps:
  1. Add 10 and 5.
  2. Subtract 3 from 8.
  3. Subtract the Step 2 result from the Step 1 result.

Assistant's reasoning:
Step 1: 10 + 5 = 15.
Step 2: 8 - 3 = 5.
Final: 15 - 5 = 10.
Verdict: Correct

Problem steps:
  1. Add 7 and 9.
  2. Subtract 6 from 4.
  3. Subtract the Step 2 result from the Step 1 result.

Assistant's reasoning:
Step 1: 7 + 9 = 17.
Step 2: 4 - 6 = -1.
Final: 17 - (-1) = 17.
Verdict: Incorrect

Now evaluate:
Problem steps:
  1. Add {a} and {b}.
  2. Subtract {d} from {c}.
  3. Subtract the Step 2 result from the Step 1 result.

Assistant's reasoning:
{assistant_text}
Return Correct only if every computation and the final answer in the reasoning are accurate; otherwise return Incorrect.
"""


@dataclass
class EvalItem:
    prompt: str
    label: bool  # True if assistant answer is correct
    metadata: Dict[str, int]


def append_verdict_label(text: str) -> str:
    return text.rstrip() + "\nVerdict:"


def sample_nonzero_delta(rng: random.Random, low: int = -4, high: int = 4) -> int:
    while True:
        delta = rng.randint(low, high)
        if delta != 0:
            return delta


def build_prompts(num_examples: int, seed: int) -> Tuple[List[EvalItem], List[EvalItem], List[EvalItem]]:
    rng = random.Random(seed)
    addition_items: List[EvalItem] = []
    subtraction_items: List[EvalItem] = []
    composite_items: List[EvalItem] = []

    for idx in range(num_examples):
        target_correct = (idx % 2 == 0)  # enforce balanced composite labels

        while True:
            a = rng.randint(1, 9)
            b = rng.randint(1, 9)
            c = rng.randint(1, 9)
            d = rng.randint(1, 9)

            true_add = a + b
            true_sub = c - d
            true_final = true_add - true_sub

            add_answer = true_add
            sub_answer = true_sub
            add_correct = True
            sub_correct = True
            final_answer = true_final
            final_correct = True
            failure_modes: List[str] = []

            if target_correct:
                failure_modes.append("none")
            else:
                add_error = rng.random() < 0.8
                sub_error = rng.random() < 0.8
                final_error = rng.random() < 0.5

                if not (add_error or sub_error or final_error):
                    choice = rng.choice(["addition", "subtraction", "final"])
                    if choice == "addition":
                        add_error = True
                    elif choice == "subtraction":
                        sub_error = True
                    else:
                        final_error = True

                if add_error:
                    delta = sample_nonzero_delta(rng, -6, 6)
                    add_answer = true_add + delta
                    add_correct = False
                    failure_modes.append("addition")
                if sub_error:
                    delta = sample_nonzero_delta(rng, -6, 6)
                    sub_answer = true_sub + delta
                    sub_correct = False
                    failure_modes.append("subtraction")

                final_answer = add_answer - sub_answer
                if final_error:
                    delta = sample_nonzero_delta(rng, -6, 6)
                    final_answer = final_answer + delta
                    failure_modes.append("final")

                final_correct = final_answer == true_final

            composite_label = bool(add_correct and sub_correct and final_correct)
            if composite_label == target_correct:
                addition_items.append(
                    EvalItem(
                        prompt=append_verdict_label(ADDITION_TEMPLATE.format(
                            a=a,
                            b=b,
                            answer=add_answer,
                        )),
                        label=add_correct,
                        metadata={
                            "id": idx,
                            "a": a,
                            "b": b,
                            "true": true_add,
                            "answer": add_answer,
                        },
                    )
                )

                subtraction_items.append(
                    EvalItem(
                        prompt=append_verdict_label(SUBTRACTION_TEMPLATE.format(
                            c=c,
                            d=d,
                            answer=sub_answer,
                        )),
                        label=sub_correct,
                        metadata={
                            "id": idx,
                            "c": c,
                            "d": d,
                            "true": true_sub,
                            "answer": sub_answer,
                        },
                    )
                )

                assistant_text = (
                    f"Step 1: {a} + {b} = {add_answer}.\n"
                    f"Step 2: {c} - {d} = {sub_answer}.\n"
                    f"Final: {add_answer} - {sub_answer} = {final_answer}."
                )
                composite_items.append(
                    EvalItem(
                        prompt=append_verdict_label(COMPOSITE_TEMPLATE.format(
                            a=a,
                            b=b,
                            c=c,
                            d=d,
                            assistant_text=assistant_text,
                        )),
                        label=composite_label,
                        metadata={
                            "id": idx,
                            "a": a,
                            "b": b,
                            "c": c,
                            "d": d,
                            "true_add": true_add,
                            "true_sub": true_sub,
                            "true_final": true_final,
                            "assistant_text": assistant_text,
                            "final_answer": final_answer,
                            "failure_modes": failure_modes,
                        },
                    )
                )
                break

    return addition_items, subtraction_items, composite_items


class JudgeEngine:
    def __init__(self, model_name: str, max_new_tokens: int, batch_size: int = 16):
        self.llm = LLM(model=model_name, trust_remote_code=True)
        self.batch_size = max(1, batch_size)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_new_tokens,
            stop=["\n", "<|im_end|>"],
        )

    def generate_batch(self, prompts: Sequence[str], desc: str) -> List[str]:
        outputs: List[str] = []
        with tqdm(total=len(prompts), desc=desc) as pbar:
            for start in range(0, len(prompts), self.batch_size):
                chunk = prompts[start : start + self.batch_size]
                results = self.llm.generate(chunk, self.sampling_params)
                for result in results:
                    text = result.outputs[0].text if result.outputs else ""
                    outputs.append(text.strip())
                pbar.update(len(chunk))
        return outputs


def parse_verdict(text: str) -> bool | None:
    lowered = text.lower()
    if not lowered:
        return None
    # Prefer the earliest verdict token to break ties gracefully.
    idx_incorrect = lowered.find("incorrect")
    idx_correct = lowered.find("correct")
    if idx_incorrect == -1 and idx_correct == -1:
        return None
    if idx_incorrect != -1 and (idx_correct == -1 or idx_incorrect < idx_correct):
        return False
    if idx_correct != -1 and (idx_incorrect == -1 or idx_correct < idx_incorrect):
        return True
    return None


def format_example_summary(name: str, item: EvalItem, completion: str, verdict: bool | None) -> str:
    meta = item.metadata
    gold = "Correct" if item.label else "Incorrect"
    pred = "Correct" if verdict else "Incorrect"
    header = f"[LOG][{name}] id={meta['id']} gold={gold} pred={pred}"

    if name == "addition":
        a, b, answer = meta["a"], meta["b"], meta["answer"]
        detail = f"Problem: {a} + {b}; assistant answer: {answer}"
    elif name == "subtraction":
        c, d, answer = meta["c"], meta["d"], meta["answer"]
        detail = f"Problem: {c} - {d}; assistant answer: {answer}"
    else:
        detail = (
            f"Steps: {meta['assistant_text']} | true final: {meta['true_final']} | "
            f"assistant final: {meta['final_answer']}"
        )
    return f"{header}\n    {detail}\n    judge output: {completion}\n"


def evaluate_split(
    name: str,
    items: Sequence[EvalItem],
    engine: JudgeEngine,
    log_examples: int = 0,
) -> Dict[str, object]:
    preds: List[bool] = []
    raw_outputs: List[str] = []
    unresolved: List[int] = []

    completions = engine.generate_batch([it.prompt for it in items], desc=f"Scoring {name}")

    for idx_local, (item, completion) in enumerate(zip(items, completions)):
        verdict = parse_verdict(completion)
        raw_outputs.append(completion)
        if verdict is None:
            unresolved.append(item.metadata["id"])
            verdict = False  # pessimistic fallback
        preds.append(verdict)
        if idx_local < log_examples:
            summary = format_example_summary(name, item, completion, verdict)
            print(summary)

    labels = [item.label for item in items]
    correctness = np.array([int(p == gt) for p, gt in zip(preds, labels)], dtype=np.float64)
    accuracy = float(correctness.mean())
    std = float(correctness.std(ddof=1)) if len(correctness) > 1 else 0.0

    return {
        "name": name,
        "accuracy": accuracy,
        "std": std,
        "num_examples": len(items),
        "num_unresolved": len(unresolved),
        "ids_unresolved": unresolved,
        "predictions": preds,
        "labels": labels,
        "raw_outputs": raw_outputs,
    }


def two_proportion_z_test(success_a: int, total_a: int, success_b: int, total_b: int) -> Tuple[float, float]:
    if total_a == 0 or total_b == 0:
        return float("nan"), float("nan")
    p1 = success_a / total_a
    p2 = success_b / total_b
    pooled = (success_a + success_b) / (total_a + total_b)
    denom = pooled * (1 - pooled) * ((1 / total_a) + (1 / total_b))
    if denom <= 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / math.sqrt(denom)
    # Two-tailed p-value using the error function.
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p_value


def summarize(results: Sequence[Dict[str, object]]) -> None:
    print("\n=== Accuracy Summary ===")
    for entry in results:
        print(
            f"{entry['name']:>12}: "
            f"acc={entry['accuracy']:.3f} ± {entry['std']:.3f} "
            f"(n={entry['num_examples']}, unresolved={entry['num_unresolved']})"
        )

    composite_entry = next(r for r in results if r["name"] == "composite")
    subtask_entries = [r for r in results if r["name"] != "composite"]
    min_subtask = min(subtask_entries, key=lambda r: r["accuracy"])
    comp_success = int(sum(
        int(pred == label) for pred, label in zip(composite_entry["predictions"], composite_entry["labels"])
    ))
    min_success = int(sum(
        int(pred == label) for pred, label in zip(min_subtask["predictions"], min_subtask["labels"])
    ))
    z, p_value = two_proportion_z_test(
        comp_success,
        composite_entry["num_examples"],
        min_success,
        min_subtask["num_examples"],
    )
    print(
        f"\nComposite vs worst subtask ({min_subtask['name']}): "
        f"Δacc={composite_entry['accuracy'] - min_subtask['accuracy']:.3f}, "
        f"z={z:.3f}, p={p_value:.3g}"
    )


def save_report(path: str, results: Sequence[Dict[str, object]], seed: int) -> None:
    import json

    serializable = {
        "seed": seed,
        "splits": [],
    }
    for entry in results:
        serializable["splits"].append(
            {
                "name": entry["name"],
                "accuracy": entry["accuracy"],
                "std": entry["std"],
                "num_examples": entry["num_examples"],
                "num_unresolved": entry["num_unresolved"],
                "ids_unresolved": entry["ids_unresolved"],
                "predictions": entry["predictions"],
                "labels": entry["labels"],
                "raw_outputs": entry["raw_outputs"],
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved detailed report to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate compositional accuracy gaps with Qwen3-0.6B judge.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="HF model id for the judge.")
    parser.add_argument("--num-examples", type=int, default=200, help="Number of base problem contexts per split.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for synthetic data.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Tokens to sample for each judgement.")
    parser.add_argument("--report-path", type=str, default="", help="Optional path to dump JSON report.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for vLLM batched generation.")
    parser.add_argument("--log-examples", type=int, default=3, help="How many examples per split to log with prompts and judgements (0 to disable).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model {args.model} ...")
    engine = JudgeEngine(args.model, args.max_new_tokens, batch_size=args.batch_size)

    print(f"Generating synthetic workloads with {args.num_examples} contexts (seed={args.seed}) ...")
    addition_items, subtraction_items, composite_items = build_prompts(args.num_examples, args.seed)

    splits = [
        ("addition", addition_items),
        ("subtraction", subtraction_items),
        ("composite", composite_items),
    ]

    results = []
    for name, items in splits:
        results.append(
            evaluate_split(name, items, engine, log_examples=args.log_examples)
        )

    summarize(results)

    if args.report_path:
        save_report(args.report_path, results, args.seed)


if __name__ == "__main__":
    main()
