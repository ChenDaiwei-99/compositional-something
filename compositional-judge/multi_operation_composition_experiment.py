#!/usr/bin/env python3
"""
Evaluate LLM-judge accuracy on compositions of heterogeneous arithmetic operations.

Provides single-operation evaluation (addition, subtraction, multiply-by-k, divide-by-k)
and multi-step reasoning traces that mix these operations for step counts 2..N.
The goal is to test how judge accuracy degrades as more heterogeneous steps are composed.
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


OPERATIONS = [
    {
        "name": "add",
        "description": "add(x, y) = x + y",
        "template": "add({x}, {y}) = {value}",
        "prompt": """You are evaluating whether the assistant's answer is correct. Respond with exactly one word: Correct or Incorrect.

Operation: add(x, y) = x + y

Examples:
Problem: add(3, 5)
Assistant answer: 8
Verdict: Correct

Problem: add(6, 2)
Assistant answer: 11
Verdict: Incorrect

Now evaluate:
Problem: add({x}, {y})
Assistant answer: {answer}
Respond Correct only if the assistant answer equals {x} + {y}; otherwise respond Incorrect.
""",
        "fn": lambda x, y: x + y,
        "operand_sampler": lambda rng: rng.randint(1, 9),
    },
    {
        "name": "sub",
        "description": "sub(x, y) = x - y",
        "template": "sub({x}, {y}) = {value}",
        "prompt": """You are evaluating whether the assistant's answer is correct. Respond with exactly one word: Correct or Incorrect.

Operation: sub(x, y) = x - y

Examples:
Problem: sub(9, 4)
Assistant answer: 5
Verdict: Correct

Problem: sub(12, 7)
Assistant answer: 6
Verdict: Incorrect

Now evaluate:
Problem: sub({x}, {y})
Assistant answer: {answer}
Respond Correct only if the assistant answer equals {x} - {y}; otherwise respond Incorrect.
""",
        "fn": lambda x, y: x - y,
        "operand_sampler": lambda rng: rng.randint(1, 9),
    },
    {
        "name": "mul",
        "description": "mul(x, y) = x × y (y ranges from 2 to 5)",
        "template": "mul({x}, {y}) = {value}",
        "prompt": """You are evaluating whether the assistant's answer is correct. Respond with exactly one word: Correct or Incorrect.

Operation: mul(x, y) = x × y

Examples:
Problem: mul(4, 3)
Assistant answer: 12
Verdict: Correct

Problem: mul(5, 2)
Assistant answer: 11
Verdict: Incorrect

Now evaluate:
Problem: mul({x}, {y})
Assistant answer: {answer}
Respond Correct only if the assistant answer equals {x} × {y}; otherwise respond Incorrect.
""",
        "fn": lambda x, y: x * y,
        "operand_sampler": lambda rng: rng.randint(2, 5),
    },
    {
        "name": "div2",
        "description": "div2(x) = x ÷ 2 (integer division if needed)",
        "template": "div2({x}) = {value}",
        "prompt": """You are evaluating whether the assistant's answer is correct. Respond with exactly one word: Correct or Incorrect.

Operation: div2(x) = x ÷ 2 (use exact division; inputs guarantee even x)

Examples:
Problem: div2(8)
Assistant answer: 4
Verdict: Correct

Problem: div2(10)
Assistant answer: 6
Verdict: Incorrect

Now evaluate:
Problem: div2({x})
Assistant answer: {answer}
Respond Correct only if the assistant answer equals {x} ÷ 2; otherwise respond Incorrect.
""",
        "fn": lambda x, _: x // 2,
        "operand_sampler": lambda rng: 0,  # unused
    },
]


MULTISTEP_TEMPLATE = """You are evaluating a multi-step arithmetic solution. Operations are defined as follows:
- add(x, y) = x + y
- sub(x, y) = x - y
- mul(x, y) = x × y (y ranges from 2 to 5)
- div2(x) = x ÷ 2 (inputs to div2 are always even)

Respond with exactly one word: Correct or Incorrect. Output Correct only if every stated computation and the final answer are accurate.

Problem steps:
{step_bullets}

Assistant's reasoning:
{assistant_text}

Respond Correct only if every computation and the final answer match the problem steps; otherwise respond Incorrect.
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


def sample_even_operand(rng: random.Random) -> int:
    return rng.choice([2, 4, 6, 8, 10, 12, 14, 16])


def sample_start_value(rng: random.Random) -> int:
    return rng.randint(4, 16)


def apply_error(value: int, rng: random.Random) -> int:
    delta = rng.randint(-5, 5)
    if delta == 0:
        delta = 1
    new_val = value + delta
    return max(-50, new_val)


def build_single_operation_items(num_examples: int, seed: int) -> List[EvalItem]:
    rng = random.Random(seed)
    items: List[EvalItem] = []
    for op in OPERATIONS:
        for idx in range(num_examples // len(OPERATIONS)):
            if op["name"] == "div2":
                x = sample_even_operand(rng)
                y = 0
            else:
                x = rng.randint(1, 20)
                y = op["operand_sampler"](rng)
            true_val = op["fn"](x, y)
            target_correct = (idx % 2 == 0)
            if target_correct:
                answer = true_val
            else:
                answer = apply_error(true_val, rng)
                if answer == true_val:
                    answer += 1

            prompt = op["prompt"].format(x=x, y=y, answer=answer).rstrip() + "\nVerdict:"
            items.append(
                EvalItem(
                    prompt=prompt,
                    label=target_correct,
                    metadata={
                        "operation": op["name"],
                        "x": x,
                        "y": y,
                        "answer": answer,
                        "true_value": true_val,
                    },
                )
            )
    return items


def build_multistep_items(num_examples: int, steps: int, seed: int) -> List[EvalItem]:
    rng = random.Random(seed)
    items: List[EvalItem] = []
    step_error_prob = 0.35
    final_error_prob = 0.4

    for idx in range(num_examples):
        target_correct = (idx % 2 == 0)
        ops_sequence = [rng.choice(OPERATIONS) for _ in range(steps)]

        while True:
            numbers: List[int] = []
            assistant_lines: List[str] = []
            values: List[int] = []

            current = sample_start_value(rng)
            if ops_sequence[0]["name"] == "div2":
                current = sample_even_operand(rng)

            numbers.append(current)

            step_values_true: List[int] = []
            assistant_values: List[int] = []

            for step_idx, op in enumerate(ops_sequence, start=1):
                if op["name"] == "div2":
                    operand = 0
                    input_val = current if current % 2 == 0 else current + (current % 2)
                    current = input_val
                    true_val = op["fn"](current, operand)
                    numbers.append(input_val)
                else:
                    operand = op["operand_sampler"](rng)
                    true_val = op["fn"](current, operand)
                    numbers.append(operand)

                step_values_true.append(true_val)
                current = true_val

            true_final = step_values_true[-1]

            if abs(true_final) > 200:
                continue

            if target_correct:
                error_flags = [False] * steps
                final_error = False
            else:
                while True:
                    error_flags = [rng.random() < step_error_prob for _ in range(steps)]
                    final_error = rng.random() < final_error_prob
                    if any(error_flags) or final_error:
                        break

            current = numbers[0]
            for step_idx, op in enumerate(ops_sequence, start=1):
                operand = 0 if op["name"] == "div2" else numbers[step_idx]
                if op["name"] == "div2" and current % 2 != 0:
                    current += 1
                correct_val = step_values_true[step_idx - 1]
                value = correct_val if not error_flags[step_idx - 1] else apply_error(correct_val, rng)
                assistant_values.append(value)
                if op["name"] == "div2":
                    assistant_lines.append(f"Step {step_idx}: div2({current}) = {value}.")
                else:
                    assistant_lines.append(
                        f"Step {step_idx}: {op['name']}({current}, {operand}) = {value}."
                    )
                current = value

            assistant_final = assistant_values[-1]
            if final_error:
                assistant_final = apply_error(assistant_final, rng)
            assistant_lines.append(f"Final: result = {assistant_final}.")

            label = not (any(error_flags) or final_error)

            step_bullets = []
            current = numbers[0]
            for step_idx, op in enumerate(ops_sequence, start=1):
                if op["name"] == "div2":
                    step_bullets.append(f"  {step_idx}. Apply div2 to Step {step_idx - 1 if step_idx > 1 else 0} result.")
                    current = current if current % 2 == 0 else current + (current % 2)
                    current = op["fn"](current, 0)
                else:
                    operand = numbers[step_idx]
                    if step_idx == 1:
                        step_bullets.append(f"  {step_idx}. Compute {op['name']}({numbers[0]}, {operand}).")
                    else:
                        step_bullets.append(f"  {step_idx}. Compute {op['name']}(Step {step_idx - 1} result, {operand}).")
                    current = op["fn"](current, operand)

            prompt = MULTISTEP_TEMPLATE.format(
                step_bullets="\n".join(step_bullets),
                assistant_text="\n".join(assistant_lines),
            ).rstrip() + "\nVerdict:"

            items.append(
                EvalItem(
                    prompt=prompt,
                    label=label,
                    metadata={
                        "id": idx,
                        "operations": [op["name"] for op in ops_sequence],
                        "numbers": numbers,
                        "assistant_values": assistant_values,
                        "assistant_final": assistant_final,
                        "true_values": step_values_true,
                        "true_final": true_final,
                    },
                )
            )
            break

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

    completions = engine.generate_batch([item.prompt for item in items], desc=f"Scoring {name}")

    for idx_local, (item, completion) in enumerate(zip(items, completions)):
        verdict = parse_verdict(completion)
        raw_outputs.append(completion)
        if verdict is None:
            unresolved.append(item.metadata["id"])
            verdict = False
        preds.append(verdict)
        if idx_local < log_examples:
            gold = "Correct" if item.label else "Incorrect"
            pred = "Correct" if verdict else "Incorrect"
            print(
                f"[LOG][{name}] id={item.metadata['id']} gold={gold} pred={pred}\n"
                f"    ops={item.metadata.get('operations')}, true_final={item.metadata.get('true_final')}\n"
                f"    judge output: {completion}\n"
            )

    labels = [item.label for item in items]
    correctness = np.array([int(p == l) for p, l in zip(preds, labels)], dtype=np.float64)
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


def summarize(results: Sequence[Dict[str, object]], baseline_names: Sequence[str]) -> None:
    print("\n=== Accuracy Summary ===")
    for entry in results:
        print(
            f"{entry['name']:>12}: acc={entry['accuracy']:.3f} ± {entry['std']:.3f} "
            f"(n={entry['num_examples']}, unresolved={entry['num_unresolved']})"
        )

    baselines = {name: next(r for r in results if r["name"] == name) for name in baseline_names}
    for entry in results:
        if entry["name"] in baseline_names:
            continue
        for base_name, base_entry in baselines.items():
            success_entry = sum(int(p == l) for p, l in zip(entry["predictions"], entry["labels"]))
            success_base = sum(int(p == l) for p, l in zip(base_entry["predictions"], base_entry["labels"]))
            z, p_value = two_proportion_z_test(
                success_entry,
                entry["num_examples"],
                success_base,
                base_entry["num_examples"],
            )
            diff = entry["accuracy"] - base_entry["accuracy"]
            print(
                f"Δ vs {base_name} ({entry['name']} - {base_name}): {diff:.3f}, z={z:.3f}, p={p_value:.3g}"
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
    parser = argparse.ArgumentParser(description="Evaluate mixed-operation compositional accuracy gaps with LLM judges.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="HF model id for the judge.")
    parser.add_argument("--num-examples", type=int, default=200, help="Number of examples per split.")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum number of operations for composite evaluations.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--max-new-tokens", type=int, default=6, help="Tokens to generate for judge verdict.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for vLLM generation.")
    parser.add_argument("--log-examples", type=int, default=0, help="How many judged examples to print per split.")
    parser.add_argument("--report-path", type=str, default="", help="Optional path to save JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    steps_range = list(range(2, args.max_steps + 1))
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Loading model {args.model} ...")
    engine = JudgeEngine(args.model, args.max_new_tokens, batch_size=args.batch_size)

    print(f"Generating datasets (seed={args.seed}) ...")

    results: List[Dict[str, object]] = []

    single_items = build_single_operation_items(args.num_examples, args.seed)
    results.append(
        evaluate_split(
            name="add",
            items=[item for item in single_items if item.metadata["operation"] == "add"],
            engine=engine,
            log_examples=args.log_examples,
        )
    )
    results.append(
        evaluate_split(
            name="sub",
            items=[item for item in single_items if item.metadata["operation"] == "sub"],
            engine=engine,
            log_examples=args.log_examples,
        )
    )
    results.append(
        evaluate_split(
            name="mul",
            items=[item for item in single_items if item.metadata["operation"] == "mul"],
            engine=engine,
            log_examples=args.log_examples,
        )
    )
    results.append(
        evaluate_split(
            name="div2",
            items=[item for item in single_items if item.metadata["operation"] == "div2"],
            engine=engine,
            log_examples=args.log_examples,
        )
    )

    for step in steps_range:
        items = build_multistep_items(args.num_examples, steps=step, seed=args.seed + step)
        results.append(
            evaluate_split(
                name=f"steps={step}",
                items=items,
                engine=engine,
                log_examples=args.log_examples,
            )
        )

    summarize(results, baseline_names=["add", "sub", "mul", "div2"])

    if args.report_path:
        save_report(args.report_path, results, args.seed, steps=steps_range)


if __name__ == "__main__":
    main()

