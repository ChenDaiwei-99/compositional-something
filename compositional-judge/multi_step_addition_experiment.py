#!/usr/bin/env python3
"""
Evaluate LLM-judge accuracy on progressively longer multi-step addition chains.

Generates synthetic reasoning traces that perform a sequence of additions and
asks the judge (e.g., Qwen3-8B) to verify whether the reasoning is correct.
Steps range from 1 (single addition) up to a configurable maximum (default 6).
"""
from __future__ import annotations

import argparse
import importlib.machinery
import math
import random
import sys
import types
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from vllm import LLM, SamplingParams

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **kwargs: x  # type: ignore


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

MULTISTEP_TEMPLATE = """You are evaluating a multi-step addition solution. Approximately half of the solutions are correct. Respond with exactly one word: Correct or Incorrect. Output Correct only if every stated computation and the final answer are accurate.

Now evaluate:
Problem steps:
{step_bullets}

Assistant's reasoning:
{assistant_text}

Respond Correct only if every computation and the final answer match what the steps require; otherwise respond Incorrect.
"""


@dataclass
class EvalItem:
    prompt: str
    label: bool
    metadata: Dict[str, object]


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
    idx_incorrect = lowered.find("incorrect")
    idx_correct = lowered.find("correct")
    if idx_incorrect == -1 and idx_correct == -1:
        return None
    if idx_incorrect != -1 and (idx_correct == -1 or idx_incorrect < idx_correct):
        return False
    if idx_correct != -1 and (idx_incorrect == -1 or idx_correct < idx_incorrect):
        return True
    return None


def sample_nonzero_delta(rng: random.Random, low: int = -3, high: int = 3) -> int:
    while True:
        delta = rng.randint(low, high)
        if delta != 0:
            return delta


def apply_error(value: int, rng: random.Random) -> int:
    for _ in range(5):
        new_val = value + sample_nonzero_delta(rng)
        if new_val >= 0:
            return new_val
    return max(0, value + 1)


def build_single_step_items(num_examples: int, seed: int) -> List[EvalItem]:
    rng = random.Random(seed)
    items: List[EvalItem] = []
    for idx in range(num_examples):
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        true_sum = a + b
        target_correct = (idx % 2 == 0)
        if target_correct:
            answer = true_sum
        else:
            answer = apply_error(true_sum, rng)
            if answer == true_sum:
                answer = true_sum + 1
        items.append(
            EvalItem(
                prompt=ADDITION_TEMPLATE.format(a=a, b=b, answer=answer).rstrip() + "\nVerdict:",
                label=target_correct,
                metadata={
                    "id": idx,
                    "numbers": [a, b],
                    "answer": answer,
                    "true_sum": true_sum,
                },
            )
        )
    return items


def build_multistep_items(num_examples: int, steps: int, seed: int) -> List[EvalItem]:
    rng = random.Random(seed)
    items: List[EvalItem] = []
    step_error_prob = 0.4
    final_error_prob = 0.4

    for idx in range(num_examples):
        numbers = [rng.randint(1, 9) for _ in range(steps + 1)]
        true_values: List[int] = []
        running_true = numbers[0] + numbers[1]
        true_values.append(running_true)
        for j in range(2, steps + 1):
            running_true += numbers[j]
            true_values.append(running_true)
        true_final = true_values[-1]

        target_correct = (idx % 2 == 0)
        if target_correct:
            error_flags = [False] * steps
            final_error = False
        else:
            while True:
                error_flags = [rng.random() < step_error_prob for _ in range(steps)]
                final_error = rng.random() < final_error_prob
                if any(error_flags) or final_error:
                    break

        assistant_values: List[int] = []
        assistant_text_lines: List[str] = []

        # Step 1
        step_value = numbers[0] + numbers[1]
        if error_flags[0]:
            step_value = apply_error(step_value, rng)
        assistant_values.append(step_value)
        assistant_text_lines.append(
            f"Step 1: {numbers[0]} + {numbers[1]} = {step_value}."
        )

        # Steps 2+
        for step_idx in range(1, steps):
            addend = numbers[step_idx + 1]
            prev_value = assistant_values[step_idx - 1]
            step_value = prev_value + addend
            if error_flags[step_idx]:
                step_value = apply_error(step_value, rng)
            assistant_values.append(step_value)
            assistant_text_lines.append(
                f"Step {step_idx + 1}: {prev_value} + {addend} = {step_value}."
            )

        assistant_final = assistant_values[-1]
        if final_error:
            assistant_final = apply_error(assistant_final, rng)
        assistant_text_lines.append(f"Final: result = {assistant_final}.")

        label = not (any(error_flags) or final_error)

        step_bullets = "\n".join(
            f"  {i + 1}. Add {numbers[i]} and {numbers[i + 1]}"
            if i == 0
            else f"  {i + 1}. Add Step {i} result and {numbers[i + 1]}"
            for i in range(steps)
        )

        prompt = MULTISTEP_TEMPLATE.format(
            step_bullets=step_bullets,
            assistant_text="\n".join(assistant_text_lines),
        ).rstrip() + "\nVerdict:"

        items.append(
            EvalItem(
                prompt=prompt,
                label=label,
                metadata={
                    "id": idx,
                    "numbers": numbers,
                    "assistant_text": assistant_text_lines,
                    "true_values": true_values,
                    "assistant_values": assistant_values,
                    "assistant_final": assistant_final,
                    "true_final": true_final,
                },
            )
        )
    return items


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
            verdict = False
        preds.append(verdict)
        if idx_local < log_examples:
            meta = item.metadata
            gold = "Correct" if item.label else "Incorrect"
            pred = "Correct" if verdict else "Incorrect"
            print(
                f"[LOG][{name}] id={meta['id']} gold={gold} pred={pred}\n"
                f"    numbers={meta['numbers']} true_final={meta.get('true_final')}\n"
                f"    judge output: {completion}\n"
            )

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


def two_proportion_z_test(success_a: int, total_a: int, success_b: int, total_b: int) -> tuple[float, float]:
    if total_a == 0 or total_b == 0:
        return float("nan"), float("nan")
    p1 = success_a / total_a
    p2 = success_b / total_b
    pooled = (success_a + success_b) / (total_a + total_b)
    denom = pooled * (1 - pooled) * ((1 / total_a) + (1 / total_b))
    if denom <= 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / math.sqrt(denom)
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p_value


def summarize(results: Sequence[Dict[str, object]], baseline_name: str) -> None:
    print("\n=== Accuracy Summary ===")
    for entry in results:
        print(
            f"{entry['name']:>10}: "
            f"acc={entry['accuracy']:.3f} ± {entry['std']:.3f} "
            f"(n={entry['num_examples']}, unresolved={entry['num_unresolved']})"
        )

    baseline = next(r for r in results if r["name"] == baseline_name)
    base_success = sum(int(p == l) for p, l in zip(baseline["predictions"], baseline["labels"]))

    for entry in results:
        if entry["name"] == baseline_name:
            continue
        success = sum(int(p == l) for p, l in zip(entry["predictions"], entry["labels"]))
        z, p_value = two_proportion_z_test(
            success,
            entry["num_examples"],
            base_success,
            baseline["num_examples"],
        )
        diff = entry["accuracy"] - baseline["accuracy"]
        print(
            f"Δ vs {baseline_name} ({entry['name']} - {baseline_name}): "
            f"{diff:.3f}, z={z:.3f}, p={p_value:.3g}"
        )


def save_report(path: str, results: Sequence[Dict[str, object]], seed: int, steps: Sequence[int]) -> None:
    import json

    serializable = {
        "seed": seed,
        "steps": list(steps),
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
    parser = argparse.ArgumentParser(description="Evaluate addition composition accuracy gaps with LLM judges.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="HF model id for the judge.")
    parser.add_argument("--num-examples", type=int, default=200, help="Number of examples per step count.")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum number of addition steps.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for synthetic data.")
    parser.add_argument("--max-new-tokens", type=int, default=6, help="Tokens to sample for each judgement.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for vLLM generation.")
    parser.add_argument("--log-examples", type=int, default=0, help="How many judged examples to log per split.")
    parser.add_argument("--report-path", type=str, default="", help="Optional path to dump JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps_range = list(range(1, args.max_steps + 1))
    random_seed = args.seed

    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Loading model {args.model} ...")
    engine = JudgeEngine(args.model, args.max_new_tokens, batch_size=args.batch_size)

    print(f"Generating synthetic workloads for steps {steps_range} with {args.num_examples} examples each (seed={random_seed}) ...")

    results: List[Dict[str, object]] = []

    single_items = build_single_step_items(args.num_examples, random_seed)
    results.append(
        evaluate_split(
            name="steps=1",
            items=single_items,
            engine=engine,
            log_examples=args.log_examples,
        )
    )

    for step in steps_range[1:]:
        items = build_multistep_items(args.num_examples, steps=step, seed=random_seed + step)
        results.append(
            evaluate_split(
                name=f"steps={step}",
                items=items,
                engine=engine,
                log_examples=args.log_examples,
            )
        )

    summarize(results, baseline_name="steps=1")

    if args.report_path:
        save_report(args.report_path, results, random_seed, steps_range)


if __name__ == "__main__":
    main()

