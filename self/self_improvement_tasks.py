#!/usr/bin/env python3
"""Task adapters for compositional self-improvement experiments."""

from __future__ import annotations

import json
import itertools
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.addition_pipeline import (
    ADDITION_SAMPLING_MODES,
    ADDITION_SAMPLING_NATURAL,
    ADDITION_WIDTH_EXACT_DIGITS,
    ADDITION_WIDTH_FIXED_MIXED_PROMPT,
    ADDITION_WIDTH_MODES,
    COMPOSITION_PATH_MODES,
    COMPOSITION_PATH_RANDOM,
    AdditionExample,
    build_composed_datasets,
    build_composed_pseudo_map,
    build_length_bucket_dataset,
    clone_with_override,
    decode_key,
    encode_key,
    example_key,
    has_component_boundary_carry,
)
from self.self_improvement_core import (
    JsonDict,
    SelfImprovementTask,
    extract_numeric_answer,
    generate_prediction_map,
)


INTEGER_PATTERN = re.compile(r"[-+]?\d+")
MAJORITY_FORMATS = {"legacy", "symbolic_v1"}
RUN_LENGTH_FORMATS = {"legacy", "symbolic_v1"}
MULTIPLICATION_FORMATS = {"legacy", "symbolic_v1"}
BIT_TARGET_MODES = {"default", "plain_output", "symbol_run_pair"}
BIT_COMPOSE_ARITIES = {"at_least2", "exact2"}
BIT_GUARDED_COMPOSE_RULES = {"none", "majority_agree_pair", "run_length_no_boundary_continue"}
RUN_LENGTH_ALPHABET_SYMBOLS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SplitName = str


def choose_component_sizes(
    target_size: int,
    sizes: Sequence[int],
    rng: random.Random,
    *,
    min_parts: int = 2,
    compose_arity: str = "at_least2",
) -> Optional[List[int]]:
    unique_sizes = sorted({size for size in sizes if size > 0 and size <= target_size})
    if not unique_sizes:
        return None

    if compose_arity == "exact2":
        pairs = [
            [left, right]
            for left in unique_sizes
            for right in unique_sizes
            if left + right == target_size
        ]
        if not pairs:
            return None
        return rng.choice(pairs)

    memo: Dict[Tuple[int, int], Optional[List[int]]] = {}

    def helper(remaining: int, parts_needed: int) -> Optional[List[int]]:
        key = (remaining, parts_needed)
        if key in memo:
            return memo[key]
        candidates = list(unique_sizes)
        rng.shuffle(candidates)
        for size in candidates:
            if size > remaining:
                continue
            next_remaining = remaining - size
            next_parts_needed = max(0, parts_needed - 1)
            if next_remaining == 0:
                if next_parts_needed == 0:
                    memo[key] = [size]
                    return memo[key]
                continue
            tail = helper(next_remaining, next_parts_needed)
            if tail is not None:
                memo[key] = [size, *tail]
                return memo[key]
        memo[key] = None
        return None

    return helper(target_size, min_parts)


def build_direct_pseudo_examples(
    candidate_examples: Sequence[Any],
    *,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    decode_max_new_tokens: int,
    key_getter: Callable[[Any], Any],
    prediction_parser: Callable[[str], Optional[str]],
    clone_builder: Callable[[Any, Optional[str]], Any],
    mode: str,
) -> Tuple[List[Any], int, JsonDict]:
    prediction_map = generate_prediction_map(
        model=model,
        tokenizer=tokenizer,
        examples=candidate_examples,
        batch_size=batch_size,
        max_new_tokens=decode_max_new_tokens,
        key_getter=key_getter,
        prediction_parser=prediction_parser,
    )
    pseudo_examples: List[Any] = []
    missing_total = 0
    for example in candidate_examples:
        override = prediction_map.get(key_getter(example))
        if override is None:
            missing_total += 1
            continue
        pseudo_examples.append(clone_builder(example, override))
    diagnostics: JsonDict = {
        "mode": mode,
        "candidate_total": len(candidate_examples),
        "retained_total": len(pseudo_examples),
        "missing_total": missing_total,
        "retained_fraction": len(pseudo_examples) / len(candidate_examples) if candidate_examples else math.nan,
    }
    return pseudo_examples, missing_total, diagnostics


def corrupt_numeric_target(value: str) -> str:
    return str(int(value) + 1)


def normalize_task_format_version(args: Any, default: str = "legacy") -> str:
    return str(getattr(args, "format_version", default))


def normalize_bit_target_mode(args: Any, default: str = "default") -> str:
    return str(getattr(args, "target_mode", default))


def normalize_compose_arity(args: Any, default: str = "at_least2") -> str:
    return str(getattr(args, "compose_arity", default))


def normalize_guarded_compose_rule(args: Any, default: str = "none") -> str:
    return str(getattr(args, "guarded_compose_rule", default))


def normalize_symbol_alphabet_size(args: Any, default: int = 2) -> int:
    return int(getattr(args, "symbol_alphabet_size", default))


def format_multiplication_target(value: int, digits: int, format_version: str) -> str:
    if format_version == "symbolic_v1":
        return f"{value:0{digits * 2}d}"
    return str(value)


def parse_multiplication_prediction(text: str, example: Optional[Any] = None) -> Optional[str]:
    value = extract_numeric_answer(text)
    if value is None:
        return None
    if example is None or getattr(example, "format_version", "legacy") != "symbolic_v1":
        return value
    return format_multiplication_target(int(value), int(example.digits), str(example.format_version))


def parse_majority_prediction(text: str, example: Optional[Any] = None) -> Optional[str]:
    matches = INTEGER_PATTERN.findall(text)
    target_mode = getattr(example, "target_mode", "default") if example is not None else "default"
    if target_mode == "plain_output":
        if not matches:
            return None
        value = int(matches[-1])
        if value not in (0, 1):
            return None
        return str(value)
    if len(matches) < 2:
        return None
    ones = int(matches[0])
    majority = int(matches[1])
    return f"{ones}|{majority}"


def parse_run_length_prediction(text: str, example: Optional[Any] = None) -> Optional[str]:
    matches = INTEGER_PATTERN.findall(text)
    target_mode = getattr(example, "target_mode", "default") if example is not None else "default"
    if target_mode == "plain_output":
        if not matches:
            return None
        value = int(matches[-1])
        if value < 0:
            return None
        return str(value)
    if target_mode == "symbol_run_pair":
        return parse_run_length_symbol_pair_prediction(text, example)
    if len(matches) < 3:
        return None
    max_run = int(matches[0])
    prefix = int(matches[1])
    suffix = int(matches[2])
    return f"{max_run}|{prefix}|{suffix}"


SYMBOL_RUN_PAIR_PATTERN = re.compile(
    rf"([{re.escape(RUN_LENGTH_ALPHABET_SYMBOLS)}])\s*(?:\||,|:|\s+)\s*([-+]?\d+)"
)


def parse_run_length_symbol_pair_prediction(text: str, example: Optional[Any] = None) -> Optional[str]:
    allowed_symbols = set(RUN_LENGTH_ALPHABET_SYMBOLS)
    if example is not None and getattr(example, "bitstring", None):
        allowed_symbols = set(str(example.bitstring))
    for match in SYMBOL_RUN_PAIR_PATTERN.finditer(text):
        symbol = match.group(1)
        if symbol not in allowed_symbols:
            continue
        value = int(match.group(2))
        if value < 0:
            continue
        return f"{symbol}|{value}"
    return None


def prepare_addition_initial_splits(
    rng: random.Random,
    min_digits: int,
    max_digits: int,
    train_per_digit: int,
    eval_per_digit: int,
    addition_width_mode: str = ADDITION_WIDTH_EXACT_DIGITS,
    addition_sampling_mode: str = ADDITION_SAMPLING_NATURAL,
) -> Tuple[Dict[SplitName, List[AdditionExample]], Dict[SplitName, set[Tuple[int, int, int]]]]:
    splits = {name: [] for name in ("train", "validation", "test")}
    records: Dict[SplitName, set[Tuple[int, int, int]]] = {name: set() for name in splits}
    generated = build_length_bucket_dataset(
        min_digits=min_digits,
        max_digits=max_digits,
        per_digit_counts={
            "train": train_per_digit,
            "validation": eval_per_digit,
            "test": eval_per_digit,
        },
        allow_carry=True,
        rng=rng,
        record_pairs=records,
        progress_name="initial",
        addition_width_mode=addition_width_mode,
        addition_sampling_mode=addition_sampling_mode,
    )
    for split in splits:
        splits[split] = generated.get(split, [])
    return splits, records


def prepare_addition_composed_train(
    rng: random.Random,
    base_splits: Dict[SplitName, List[AdditionExample]],
    base_records: Dict[SplitName, set[Tuple[int, int, int]]],
    min_digits: int,
    max_digits: int,
    per_digit_count: int,
    allow_carry: bool,
    boundary_carry_policy: str = "any",
    additional_exclude: Optional[set[Tuple[int, int, int]]] = None,
    addition_width_mode: str = ADDITION_WIDTH_EXACT_DIGITS,
    composition_path_mode: str = COMPOSITION_PATH_RANDOM,
) -> Tuple[List[AdditionExample], Dict[Tuple[int, int, int], List[Tuple[int, int, int]]], set[Tuple[int, int, int]]]:
    if max_digits < min_digits or per_digit_count <= 0:
        return [], {}, set()
    composed_records: Dict[SplitName, set[Tuple[int, int, int]]] = {"train": set(), "validation": set(), "test": set()}
    component_records: Dict[SplitName, Dict[Tuple[int, int, int], List[Tuple[int, int, int]]]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    base_used = set().union(*base_records.values())
    if additional_exclude:
        base_used.update(additional_exclude)
    composed_splits = build_composed_datasets(
        base_splits=base_splits,
        min_digits=min_digits,
        max_digits=max_digits,
        per_digit_counts={"train": per_digit_count, "validation": 0, "test": 0},
        rng=rng,
        exclude_pairs=base_used,
        record_pairs=composed_records,
        progress_name="composed",
        record_components=component_records,
        allow_carry=allow_carry,
        allow_nocarry=True,
        boundary_carry_policy=boundary_carry_policy,
        addition_width_mode=addition_width_mode,
        composition_path_mode=composition_path_mode,
    )
    return composed_splits.get("train", []), component_records.get("train", {}), composed_records.get("train", set())


def prepare_addition_composed_eval(
    rng: random.Random,
    base_splits: Dict[SplitName, List[AdditionExample]],
    base_records: Dict[SplitName, set[Tuple[int, int, int]]],
    min_digits: int,
    max_digits: int,
    per_digit_count: int,
    additional_exclude: Optional[set[Tuple[int, int, int]]] = None,
    addition_width_mode: str = ADDITION_WIDTH_EXACT_DIGITS,
    composition_path_mode: str = COMPOSITION_PATH_RANDOM,
) -> Tuple[List[AdditionExample], Dict[Tuple[int, int, int], List[Tuple[int, int, int]]], set[Tuple[int, int, int]]]:
    if max_digits < min_digits or per_digit_count <= 0:
        return [], {}, set()
    composed_records: Dict[SplitName, set[Tuple[int, int, int]]] = {"train": set(), "validation": set(), "test": set()}
    component_records: Dict[SplitName, Dict[Tuple[int, int, int], List[Tuple[int, int, int]]]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    base_used = set().union(*base_records.values())
    if additional_exclude:
        base_used.update(additional_exclude)
    stitched_base_splits = {
        "train": list(base_splits.get("train", [])),
        "validation": list(base_splits.get("train", [])),
        "test": list(base_splits.get("train", [])),
    }
    composed_splits = build_composed_datasets(
        base_splits=stitched_base_splits,
        min_digits=min_digits,
        max_digits=max_digits,
        per_digit_counts={"train": 0, "validation": 0, "test": per_digit_count},
        rng=rng,
        exclude_pairs=base_used,
        record_pairs=composed_records,
        progress_name="composed-eval",
        record_components=component_records,
        allow_carry=True,
        allow_nocarry=True,
        addition_width_mode=addition_width_mode,
        composition_path_mode=composition_path_mode,
    )
    return composed_splits.get("test", []), component_records.get("test", {}), composed_records.get("test", set())


def prepare_addition_eval_examples(
    rng: random.Random,
    min_digits: int,
    max_digits: int,
    per_digit: int,
    exclude: set[Tuple[int, int, int]],
    addition_width_mode: str = ADDITION_WIDTH_EXACT_DIGITS,
    addition_sampling_mode: str = ADDITION_SAMPLING_NATURAL,
) -> List[AdditionExample]:
    generated = build_length_bucket_dataset(
        min_digits=min_digits,
        max_digits=max_digits,
        per_digit_counts={"train": 0, "validation": 0, "test": per_digit},
        allow_carry=True,
        rng=rng,
        exclude_pairs=exclude,
        record_pairs={split: set() for split in ("train", "validation", "test")},
        progress_name="evaluation",
        addition_width_mode=addition_width_mode,
        addition_sampling_mode=addition_sampling_mode,
    )
    return list(generated.get("test", []))


def get_boundary_carry_status(
    example: AdditionExample,
    component_map: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]],
) -> Optional[bool]:
    component_keys = component_map.get(example_key(example))
    if not component_keys:
        return None
    component_digits = [key[0] for key in component_keys]
    if len(component_digits) <= 1:
        return None
    if sum(component_digits) != example.digits:
        return None
    return has_component_boundary_carry(example, component_digits)


def split_addition_examples_by_boundary_status(
    examples: Sequence[AdditionExample],
    component_map: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]],
) -> Dict[str, List[AdditionExample]]:
    slices: Dict[str, List[AdditionExample]] = {
        "boundary_carry": [],
        "no_boundary_carry": [],
        "unknown": [],
    }
    for example in examples:
        status = get_boundary_carry_status(example, component_map)
        if status is True:
            slices["boundary_carry"].append(example)
        elif status is False:
            slices["no_boundary_carry"].append(example)
        else:
            slices["unknown"].append(example)
    return slices


def count_examples_by_size(examples: Sequence[Any], size_getter: Callable[[Any], int]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for example in examples:
        counts[size_getter(example)] += 1
    return dict(counts)


def format_size_count_map(values: Dict[int, int]) -> str:
    return ", ".join(f"{size}:{count}" for size, count in sorted(values.items()))


def majority_label_from_bitstring(bitstring: str) -> int:
    ones = bitstring.count("1")
    return 1 if ones * 2 >= len(bitstring) else 0


def majority_guard_accepts_true_components(component_keys: Sequence[Tuple[int, str]]) -> Optional[bool]:
    if len(component_keys) != 2:
        return None
    left_label = majority_label_from_bitstring(component_keys[0][1])
    right_label = majority_label_from_bitstring(component_keys[1][1])
    return left_label == right_label


def run_length_guard_accepts_true_components(
    component_keys: Sequence[Tuple[int, str]],
) -> Optional[bool]:
    if len(component_keys) != 2:
        return None
    left_bitstring = component_keys[0][1]
    right_bitstring = component_keys[1][1]
    if not left_bitstring or not right_bitstring:
        return None
    return left_bitstring[-1] != right_bitstring[0]


def guard_slice_partition(
    examples: Sequence[Any],
    component_map: Dict[Any, List[Any]],
    *,
    key_getter: Callable[[Any], Any],
    guard_fn: Callable[[Sequence[Any]], Optional[bool]],
) -> Dict[str, List[Any]]:
    accepted: List[Any] = []
    rejected: List[Any] = []
    for example in examples:
        component_keys = component_map.get(key_getter(example), [])
        status = guard_fn(component_keys)
        if status is True:
            accepted.append(example)
        elif status is False:
            rejected.append(example)
    return {
        "accepted_by_guard": accepted,
        "rejected_by_guard": rejected,
        "all": list(examples),
    }


def format_majority_target(ones: int, bits: int, format_version: str, target_mode: str = "default") -> str:
    majority = 1 if ones * 2 >= bits else 0
    if target_mode == "plain_output":
        return str(majority)
    return f"{ones}|{majority}"


@dataclass(frozen=True)
class MajorityExample:
    bitstring: str
    bits: int
    ones: int
    majority: int
    format_version: str = "legacy"
    target_mode: str = "default"
    target_override: Optional[str] = None

    def prompt(self) -> str:
        if self.format_version == "symbolic_v1":
            return f"majority({self.bitstring})="
        return f"Q: majority({self.bitstring}) = ?\nA:"

    def target(self) -> str:
        if self.target_override is not None:
            return self.target_override
        return format_majority_target(self.ones, self.bits, self.format_version, self.target_mode)

    def target_prefix(self) -> str:
        return "" if self.format_version == "symbolic_v1" else " "


def majority_key(example: MajorityExample) -> Tuple[int, str]:
    return example.bits, example.bitstring


def encode_majority_key(key: Tuple[int, str]) -> str:
    return f"{key[0]}|{key[1]}"


def decode_majority_key(value: str) -> Tuple[int, str]:
    bits, bitstring = value.split("|", 1)
    return int(bits), bitstring


def generate_majority_example(
    num_bits: int,
    rng: random.Random,
    format_version: str = "legacy",
    target_mode: str = "default",
) -> MajorityExample:
    bitstring = "".join(rng.choice("01") for _ in range(num_bits))
    ones = sum(int(bit) for bit in bitstring)
    majority = 1 if ones * 2 >= num_bits else 0
    return MajorityExample(
        bitstring=bitstring,
        bits=num_bits,
        ones=ones,
        majority=majority,
        format_version=format_version,
        target_mode=target_mode,
    )


def compose_majority_examples(*examples: MajorityExample) -> MajorityExample:
    if len(examples) < 2:
        raise ValueError("Need at least two majority examples to compose a longer instance.")
    bitstring = "".join(example.bitstring for example in examples)
    ones = sum(example.ones for example in examples)
    bits = sum(example.bits for example in examples)
    majority = 1 if ones * 2 >= bits else 0
    return MajorityExample(
        bitstring=bitstring,
        bits=bits,
        ones=ones,
        majority=majority,
        format_version=examples[0].format_version,
        target_mode=examples[0].target_mode,
    )


def compose_majority_to_length(
    buckets: Dict[int, List[MajorityExample]],
    target_bits: int,
    rng: random.Random,
    *,
    compose_arity: str = "at_least2",
) -> Tuple[MajorityExample, List[MajorityExample]]:
    sizes = list(buckets.keys())
    chosen_sizes = choose_component_sizes(target_bits, sizes, rng, compose_arity=compose_arity)
    if not chosen_sizes:
        raise ValueError(
            f"Unable to compose a majority example of {target_bits} bits from base bucket sizes {sorted(sizes)}."
        )
    chosen = [rng.choice(buckets[size]) for size in chosen_sizes]
    return compose_majority_examples(*chosen), chosen


def sample_unique_bitstrings(
    bits: int,
    count: int,
    rng: random.Random,
    occupied: set[Tuple[int, str]],
    *,
    alphabet: str = "01",
    max_attempts: int = 10_000,
    ) -> List[str]:
    if count <= 0:
        return []
    available = (len(alphabet) ** bits) - sum(1 for key in occupied if key[0] == bits)
    if count > available:
        raise ValueError(f"Requested {count} unique bitstrings for bits={bits}, but only {available} remain.")

    if available <= 131_072 and count * 4 >= available:
        pool = [
            "".join(symbols)
            for symbols in itertools.product(alphabet, repeat=bits)
            if (bits, "".join(symbols)) not in occupied
        ]
        rng.shuffle(pool)
        return pool[:count]

    sampled: List[str] = []
    attempts = 0
    while len(sampled) < count:
        attempts += 1
        bitstring = "".join(rng.choice(alphabet) for _ in range(bits))
        key = (bits, bitstring)
        if key in occupied or bitstring in sampled:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"Unable to sample {count} unique bitstrings for bits={bits} after {max_attempts} attempts."
                )
            continue
        sampled.append(bitstring)
        attempts = 0
    return sampled


def build_majority_length_bucket_dataset(
    min_bits: int,
    max_bits: int,
    per_bit_counts: Dict[str, int],
    rng: random.Random,
    *,
    exclude_keys: Optional[set[Tuple[int, str]]] = None,
    record_keys: Optional[Dict[str, set[Tuple[int, str]]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 10_000,
    format_version: str = "legacy",
    target_mode: str = "default",
    split_order: Sequence[str] = ("train", "validation", "test"),
) -> Dict[str, List[MajorityExample]]:
    splits = {key: [] for key in ("train", "validation", "test")}
    occupied = set(exclude_keys) if exclude_keys else set()
    used_counts: Dict[int, int] = defaultdict(int)
    for key in occupied:
        used_counts[key[0]] += 1

    for bits in range(min_bits, max_bits + 1):
        per_bit_per_split = {split: per_bit_counts.get(split, 0) for split in split_order}
        total_requested = sum(per_bit_per_split.values())
        if total_requested == 0:
            continue
        available_unique = max(0, (2**bits) - used_counts.get(bits, 0))
        if available_unique < total_requested:
            print(
                f"[WARN] Requested {total_requested} examples for bits={bits} exceeds available unique strings ({available_unique}); capping counts.",
                flush=True,
            )
            remaining = available_unique
            for split in split_order:
                requested = per_bit_per_split[split]
                if requested > remaining:
                    per_bit_per_split[split] = remaining
                    remaining = 0
                else:
                    remaining -= requested
            total_requested = sum(per_bit_per_split.values())
            if total_requested == 0:
                continue

        generated: List[Tuple[MajorityExample, Tuple[int, str], bool]] = []
        for bitstring in sample_unique_bitstrings(bits, total_requested, rng, occupied, max_attempts=max_attempts):
            ones = bitstring.count("1")
            example = MajorityExample(
                bitstring=bitstring,
                bits=bits,
                ones=ones,
                majority=1 if ones * 2 >= bits else 0,
                format_version=format_version,
                target_mode=target_mode,
            )
            key = majority_key(example)
            occupied.add(key)
            used_counts[bits] += 1
            generated.append((example, key, False))

        index = 0
        for split in split_order:
            count = per_bit_per_split.get(split, 0)
            if count <= 0:
                continue
            chunk = generated[index : index + count]
            index += count
            splits[split].extend(example for example, _, _ in chunk)
            if record_keys and split in record_keys:
                for _, key, is_duplicate in chunk:
                    if not is_duplicate:
                        record_keys[split].add(key)
            if progress_name:
                print(
                    f"[INFO] Generated {len(chunk)}/{count} {progress_name} examples for split='{split}' bits={bits}",
                    flush=True,
                )

    for split in splits:
        rng.shuffle(splits[split])
    return splits


def bucket_majority_by_bits(examples: Sequence[MajorityExample]) -> Dict[int, List[MajorityExample]]:
    buckets: Dict[int, List[MajorityExample]] = defaultdict(list)
    for example in examples:
        buckets[example.bits].append(example)
    return buckets


def build_majority_composed_dataset(
    base_splits: Dict[str, List[MajorityExample]],
    min_bits: int,
    max_bits: int,
    per_bit_counts: Dict[str, int],
    rng: random.Random,
    *,
    exclude_keys: Optional[set[Tuple[int, str]]] = None,
    record_keys: Optional[Dict[str, set[Tuple[int, str]]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 10_000,
    record_components: Optional[Dict[str, Dict[Tuple[int, str], List[Tuple[int, str]]]]] = None,
    compose_arity: str = "at_least2",
) -> Dict[str, List[MajorityExample]]:
    splits = {key: [] for key in ("train", "validation", "test")}
    occupied = set(exclude_keys) if exclude_keys else set()
    used_counts: Dict[int, int] = defaultdict(int)
    for key in occupied:
        used_counts[key[0]] += 1

    for split in ("train", "validation", "test"):
        requested_per_bit = per_bit_counts.get(split, 0)
        if requested_per_bit <= 0:
            continue
        buckets = bucket_majority_by_bits(base_splits.get(split, []))
        component_map = None
        if record_components is not None:
            component_map = record_components.setdefault(split, {})
        for bits in range(min_bits, max_bits + 1):
            available_unique = max(0, (2**bits) - used_counts.get(bits, 0))
            effective_target = min(requested_per_bit, available_unique)
            if effective_target < requested_per_bit:
                print(
                    f"[WARN] Requested {requested_per_bit} composed examples for bits={bits} split='{split}' exceeds available unique strings ({available_unique}); capping.",
                    flush=True,
                )
            if effective_target <= 0:
                continue
            generated: List[Tuple[MajorityExample, Tuple[int, str], bool, List[Tuple[int, str]]]] = []
            attempts = 0
            duplicates_allowed = False
            while len(generated) < effective_target:
                attempts += 1
                example, components = compose_majority_to_length(
                    buckets,
                    bits,
                    rng,
                    compose_arity=compose_arity,
                )
                component_keys = [majority_key(component) for component in components]
                key = majority_key(example)
                if key in occupied:
                    if attempts >= max_attempts:
                        if not duplicates_allowed:
                            print(
                                f"[WARN] Exhausted unique composed sampling for bits={bits} split='{split}' (progress={progress_name}); allowing duplicates.",
                                flush=True,
                            )
                            duplicates_allowed = True
                        generated.append((example, key, True, component_keys))
                        attempts = 0
                    continue
                occupied.add(key)
                used_counts[bits] += 1
                generated.append((example, key, False, component_keys))
                attempts = 0
            splits[split].extend(example for example, _, _, _ in generated)
            if record_keys and split in record_keys:
                for _, key, is_duplicate, _ in generated:
                    if not is_duplicate:
                        record_keys[split].add(key)
            if component_map is not None:
                for _, key, _, component_keys in generated:
                    component_map[key] = component_keys
            if progress_name:
                print(
                    f"[INFO] Generated {len(generated)}/{effective_target} {progress_name} examples for split='{split}' bits={bits}",
                    flush=True,
                )
        rng.shuffle(splits[split])
    return splits


def clone_majority_with_override(example: MajorityExample, override: Optional[str]) -> MajorityExample:
    if override is None:
        return example
    return MajorityExample(
        bitstring=example.bitstring,
        bits=example.bits,
        ones=example.ones,
        majority=example.majority,
        format_version=example.format_version,
        target_mode=example.target_mode,
        target_override=override,
    )


def format_run_length_target(
    max_run: int,
    prefix: int,
    suffix: int,
    format_version: str,
    target_mode: str = "default",
    *,
    bitstring: Optional[str] = None,
) -> str:
    if target_mode == "plain_output":
        return str(max_run)
    if target_mode == "symbol_run_pair":
        if bitstring is None:
            raise ValueError("symbol_run_pair run-length targets require bitstring context.")
        symbol, run_length = leftmost_max_run_pair(bitstring)
        return f"{symbol}|{run_length}"
    return f"{max_run}|{prefix}|{suffix}"


@dataclass(frozen=True)
class RunLengthExample:
    bitstring: str
    bits: int
    max_run: int
    prefix_run: int
    suffix_run: int
    format_version: str = "legacy"
    target_mode: str = "default"
    target_override: Optional[str] = None

    def prompt(self) -> str:
        if self.format_version == "symbolic_v1":
            return f"runlen({self.bitstring})="
        return f"Q: runlen({self.bitstring}) = ?\nA:"

    def target(self) -> str:
        if self.target_override is not None:
            return self.target_override
        return format_run_length_target(
            self.max_run,
            self.prefix_run,
            self.suffix_run,
            self.format_version,
            self.target_mode,
            bitstring=self.bitstring,
        )

    def target_prefix(self) -> str:
        return "" if self.format_version == "symbolic_v1" else " "


def run_length_key(example: RunLengthExample) -> Tuple[int, str]:
    return example.bits, example.bitstring


def encode_run_length_key(key: Tuple[int, str]) -> str:
    return f"{key[0]}|{key[1]}"


def decode_run_length_key(value: str) -> Tuple[int, str]:
    bits, bitstring = value.split("|", 1)
    return int(bits), bitstring


def compute_run_stats(bitstring: str) -> Tuple[int, int, int]:
    if not bitstring:
        return 0, 0, 0
    max_run = 1
    current = 1
    for previous, current_symbol in zip(bitstring, bitstring[1:]):
        if current_symbol == previous:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    prefix_symbol = bitstring[0]
    prefix = 0
    for ch in bitstring:
        if ch == prefix_symbol:
            prefix += 1
        else:
            break
    suffix_symbol = bitstring[-1]
    suffix = 0
    for ch in reversed(bitstring):
        if ch == suffix_symbol:
            suffix += 1
        else:
            break
    return max_run, prefix, suffix


def leftmost_max_run_pair(bitstring: str) -> Tuple[str, int]:
    if not bitstring:
        return "", 0
    best_symbol = bitstring[0]
    best_length = 1
    current_symbol = bitstring[0]
    current_length = 1
    for ch in bitstring[1:]:
        if ch == current_symbol:
            current_length += 1
        else:
            current_symbol = ch
            current_length = 1
        if current_length > best_length:
            best_symbol = current_symbol
            best_length = current_length
    return best_symbol, best_length


def generate_run_length_example(
    num_bits: int,
    rng: random.Random,
    format_version: str = "legacy",
    target_mode: str = "default",
    alphabet: str = "01",
) -> RunLengthExample:
    bitstring = "".join(rng.choice(alphabet) for _ in range(num_bits))
    max_run, prefix, suffix = compute_run_stats(bitstring)
    return RunLengthExample(
        bitstring=bitstring,
        bits=num_bits,
        max_run=max_run,
        prefix_run=prefix,
        suffix_run=suffix,
        format_version=format_version,
        target_mode=target_mode,
    )


def merge_run_length(left: RunLengthExample, right: RunLengthExample) -> RunLengthExample:
    bitstring = left.bitstring + right.bitstring
    bits = left.bits + right.bits
    max_run, prefix, suffix = compute_run_stats(bitstring)
    return RunLengthExample(
        bitstring=bitstring,
        bits=bits,
        max_run=max_run,
        prefix_run=prefix,
        suffix_run=suffix,
        format_version=left.format_version,
        target_mode=left.target_mode,
    )


def compose_run_length_examples(*examples: RunLengthExample) -> RunLengthExample:
    if len(examples) < 2:
        raise ValueError("Need at least two run-length examples to compose a longer instance.")
    merged = examples[0]
    for nxt in examples[1:]:
        merged = merge_run_length(merged, nxt)
    return merged


def compose_run_length_to_length(
    buckets: Dict[int, List[RunLengthExample]],
    target_bits: int,
    rng: random.Random,
    *,
    compose_arity: str = "at_least2",
) -> Tuple[RunLengthExample, List[RunLengthExample]]:
    sizes = list(buckets.keys())
    chosen_sizes = choose_component_sizes(target_bits, sizes, rng, compose_arity=compose_arity)
    if not chosen_sizes:
        raise ValueError(
            f"Unable to compose a run-length example of {target_bits} bits from base bucket sizes {sorted(sizes)}."
        )
    chosen = [rng.choice(buckets[size]) for size in chosen_sizes]
    return compose_run_length_examples(*chosen), chosen


def build_run_length_length_bucket_dataset(
    min_bits: int,
    max_bits: int,
    per_bit_counts: Dict[str, int],
    rng: random.Random,
    *,
    exclude_keys: Optional[set[Tuple[int, str]]] = None,
    record_keys: Optional[Dict[str, set[Tuple[int, str]]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 10_000,
    format_version: str = "legacy",
    target_mode: str = "default",
    alphabet: str = "01",
    split_order: Sequence[str] = ("train", "validation", "test"),
) -> Dict[str, List[RunLengthExample]]:
    splits = {key: [] for key in ("train", "validation", "test")}
    occupied = set(exclude_keys) if exclude_keys else set()
    used_counts: Dict[int, int] = defaultdict(int)
    for key in occupied:
        used_counts[key[0]] += 1

    for bits in range(min_bits, max_bits + 1):
        per_bit_per_split = {split: per_bit_counts.get(split, 0) for split in split_order}
        total_requested = sum(per_bit_per_split.values())
        if total_requested == 0:
            continue
        available_unique = max(0, (len(alphabet) ** bits) - used_counts.get(bits, 0))
        if available_unique < total_requested:
            print(
                f"[WARN] Requested {total_requested} examples for bits={bits} exceeds available unique strings ({available_unique}); capping counts.",
                flush=True,
            )
            remaining = available_unique
            for split in split_order:
                requested = per_bit_per_split[split]
                if requested > remaining:
                    per_bit_per_split[split] = remaining
                    remaining = 0
                else:
                    remaining -= requested
            total_requested = sum(per_bit_per_split.values())
            if total_requested == 0:
                continue

        generated: List[Tuple[RunLengthExample, Tuple[int, str], bool]] = []
        for bitstring in sample_unique_bitstrings(
            bits,
            total_requested,
            rng,
            occupied,
            alphabet=alphabet,
            max_attempts=max_attempts,
        ):
            max_run, prefix, suffix = compute_run_stats(bitstring)
            example = RunLengthExample(
                bitstring=bitstring,
                bits=bits,
                max_run=max_run,
                prefix_run=prefix,
                suffix_run=suffix,
                format_version=format_version,
                target_mode=target_mode,
            )
            key = run_length_key(example)
            occupied.add(key)
            used_counts[bits] += 1
            generated.append((example, key, False))

        index = 0
        for split in split_order:
            count = per_bit_per_split.get(split, 0)
            if count <= 0:
                continue
            chunk = generated[index : index + count]
            index += count
            splits[split].extend(example for example, _, _ in chunk)
            if record_keys and split in record_keys:
                for _, key, is_duplicate in chunk:
                    if not is_duplicate:
                        record_keys[split].add(key)
            if progress_name:
                print(
                    f"[INFO] Generated {len(chunk)}/{count} {progress_name} examples for split='{split}' bits={bits}",
                    flush=True,
                )

    for split in splits:
        rng.shuffle(splits[split])
    return splits


def bucket_run_length_by_bits(examples: Sequence[RunLengthExample]) -> Dict[int, List[RunLengthExample]]:
    buckets: Dict[int, List[RunLengthExample]] = defaultdict(list)
    for example in examples:
        buckets[example.bits].append(example)
    return buckets


def build_run_length_composed_dataset(
    base_splits: Dict[str, List[RunLengthExample]],
    min_bits: int,
    max_bits: int,
    per_bit_counts: Dict[str, int],
    rng: random.Random,
    *,
    exclude_keys: Optional[set[Tuple[int, str]]] = None,
    record_keys: Optional[Dict[str, set[Tuple[int, str]]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 10_000,
    record_components: Optional[Dict[str, Dict[Tuple[int, str], List[Tuple[int, str]]]]] = None,
    compose_arity: str = "at_least2",
) -> Dict[str, List[RunLengthExample]]:
    splits = {key: [] for key in ("train", "validation", "test")}
    occupied = set(exclude_keys) if exclude_keys else set()
    used_counts: Dict[int, int] = defaultdict(int)
    for key in occupied:
        used_counts[key[0]] += 1

    for split in ("train", "validation", "test"):
        requested_per_bit = per_bit_counts.get(split, 0)
        if requested_per_bit <= 0:
            continue
        buckets = bucket_run_length_by_bits(base_splits.get(split, []))
        component_map = None
        if record_components is not None:
            component_map = record_components.setdefault(split, {})
        for bits in range(min_bits, max_bits + 1):
            available_unique = max(0, (2**bits) - used_counts.get(bits, 0))
            effective_target = min(requested_per_bit, available_unique)
            if effective_target < requested_per_bit:
                print(
                    f"[WARN] Requested {requested_per_bit} composed examples for bits={bits} split='{split}' exceeds available unique strings ({available_unique}); capping.",
                    flush=True,
                )
            if effective_target <= 0:
                continue
            generated: List[Tuple[RunLengthExample, Tuple[int, str], bool, List[Tuple[int, str]]]] = []
            attempts = 0
            duplicates_allowed = False
            while len(generated) < effective_target:
                attempts += 1
                example, components = compose_run_length_to_length(
                    buckets,
                    bits,
                    rng,
                    compose_arity=compose_arity,
                )
                component_keys = [run_length_key(component) for component in components]
                key = run_length_key(example)
                if key in occupied:
                    if attempts >= max_attempts:
                        if not duplicates_allowed:
                            print(
                                f"[WARN] Exhausted unique composed sampling for bits={bits} split='{split}' (progress={progress_name}); allowing duplicates.",
                                flush=True,
                            )
                            duplicates_allowed = True
                        generated.append((example, key, True, component_keys))
                        attempts = 0
                    continue
                occupied.add(key)
                used_counts[bits] += 1
                generated.append((example, key, False, component_keys))
                attempts = 0
            splits[split].extend(example for example, _, _, _ in generated)
            if record_keys and split in record_keys:
                for _, key, is_duplicate, _ in generated:
                    if not is_duplicate:
                        record_keys[split].add(key)
            if component_map is not None:
                for _, key, _, component_keys in generated:
                    component_map[key] = component_keys
            if progress_name:
                print(
                    f"[INFO] Generated {len(generated)}/{effective_target} {progress_name} examples for split='{split}' bits={bits}",
                    flush=True,
                )
        rng.shuffle(splits[split])
    return splits


def clone_run_length_with_override(example: RunLengthExample, override: Optional[str]) -> RunLengthExample:
    if override is None:
        return example
    return RunLengthExample(
        bitstring=example.bitstring,
        bits=example.bits,
        max_run=example.max_run,
        prefix_run=example.prefix_run,
        suffix_run=example.suffix_run,
        format_version=example.format_version,
        target_mode=example.target_mode,
        target_override=override,
    )


def build_guarded_bit_pseudo_examples(
    candidate_examples: Sequence[Any],
    initial_component_map: Dict[Any, List[Any]],
    *,
    target_max_size: int,
    requested_per_size: int,
    size_getter: Callable[[Any], int],
    key_getter: Callable[[Any], Any],
    clone_builder: Callable[[Any, Optional[str]], Any],
    evaluate_candidate: Callable[[Any, Optional[Sequence[Any]]], Tuple[str, Optional[str]]],
    refill_builder: Callable[[int, int, set[Any]], Tuple[List[Any], Dict[Any, List[Any]]]],
    mode: str,
    max_refill_rounds: int = 32,
) -> Tuple[List[Any], int, JsonDict]:
    active_candidates = [example for example in candidate_examples if size_getter(example) <= target_max_size]
    target_sizes = sorted({size_getter(example) for example in active_candidates})
    requested_counts = {size: requested_per_size for size in target_sizes}
    requested_total = sum(requested_counts.values())

    candidate_total_by_size: Dict[int, int] = defaultdict(int)
    retained_total_by_size: Dict[int, int] = defaultdict(int)
    missing_total_by_size: Dict[int, int] = defaultdict(int)
    rejected_total_by_size: Dict[int, int] = defaultdict(int)
    pseudo_examples: List[Any] = []
    missing_total = 0
    rejected_total = 0
    occupied_keys = {key_getter(example) for example in active_candidates}

    def process_batch(
        examples: Sequence[Any],
        component_map: Dict[Any, List[Any]],
    ) -> None:
        nonlocal missing_total, rejected_total
        for example in examples:
            key = key_getter(example)
            size = size_getter(example)
            candidate_total_by_size[size] += 1
            status, override = evaluate_candidate(example, component_map.get(key))
            if status == "accepted" and override is not None:
                pseudo_examples.append(clone_builder(example, override))
                retained_total_by_size[size] += 1
            elif status == "missing":
                missing_total += 1
                missing_total_by_size[size] += 1
            else:
                rejected_total += 1
                rejected_total_by_size[size] += 1

    process_batch(active_candidates, initial_component_map)

    refill_rounds = 0
    while True:
        deficits = {
            size: max(0, requested_counts[size] - retained_total_by_size.get(size, 0))
            for size in requested_counts
        }
        deficits = {size: count for size, count in deficits.items() if count > 0}
        if not deficits:
            break
        if refill_rounds >= max_refill_rounds:
            raise RuntimeError(
                "Unable to retain the requested guarded pseudo examples after refill attempts. "
                f"Missing per-size counts: {format_size_count_map(deficits)}"
            )
        refill_rounds += 1
        progress_made = False
        for size, need in sorted(deficits.items()):
            refill_examples, refill_component_map = refill_builder(size, need, occupied_keys)
            if not refill_examples:
                continue
            progress_made = True
            occupied_keys.update(key_getter(example) for example in refill_examples)
            process_batch(refill_examples, refill_component_map)
        if not progress_made:
            raise RuntimeError(
                "Unable to retain the requested guarded pseudo examples after refill attempts. "
                f"Missing per-size counts: {format_size_count_map(deficits)}"
            )

    diagnostics: JsonDict = {
        "mode": mode,
        "target_max_bits": int(target_max_size),
        "requested_per_size": requested_per_size,
        "requested_total": requested_total,
        "candidate_total": sum(candidate_total_by_size.values()),
        "retained_total": len(pseudo_examples),
        "missing_total": missing_total,
        "rejected_total": rejected_total,
        "retained_fraction": len(pseudo_examples) / sum(candidate_total_by_size.values())
        if candidate_total_by_size
        else math.nan,
        "per_size_candidate_total": dict(sorted(candidate_total_by_size.items())),
        "per_size_retained_total": dict(sorted(retained_total_by_size.items())),
        "per_size_missing_total": dict(sorted(missing_total_by_size.items())),
        "per_size_rejected_total": dict(sorted(rejected_total_by_size.items())),
        "refill_rounds": refill_rounds,
    }
    return pseudo_examples, missing_total, diagnostics


def exact2_reachable_sizes_from_examples(
    examples: Sequence[Any],
    *,
    size_getter: Callable[[Any], int],
    min_size: int,
    max_size: int,
) -> List[int]:
    available_sizes = sorted({size_getter(example) for example in examples})
    if not available_sizes:
        return []
    reachable = {
        left + right
        for left in available_sizes
        for right in available_sizes
        if min_size <= left + right <= max_size
    }
    return sorted(reachable)


@dataclass(frozen=True)
class MultiplicationExample:
    a: int
    b: int
    digits: int
    result: int
    operand_width: int
    format_version: str = "legacy"
    target_override: Optional[str] = None

    def prompt(self) -> str:
        if self.format_version == "symbolic_v1":
            return f"{self.a:0{self.operand_width}d}×{self.b:0{self.operand_width}d}="
        return f"Q: {self.a:0{self.operand_width}d} * {self.b:0{self.operand_width}d} = ?\nA:"

    def target(self) -> str:
        if self.target_override is not None:
            return self.target_override
        return format_multiplication_target(self.result, self.digits, self.format_version)

    def target_prefix(self) -> str:
        return "" if self.format_version == "symbolic_v1" else " "


def multiplication_key(example: MultiplicationExample) -> Tuple[int, int, int]:
    return example.digits, example.a, example.b


def encode_multiplication_key(key: Tuple[int, int, int]) -> str:
    return f"{key[0]}|{key[1]}|{key[2]}"


def decode_multiplication_key(value: str) -> Tuple[int, int, int]:
    digits, a, b = value.split("|", 2)
    return int(digits), int(a), int(b)


def clone_multiplication_with_override(
    example: MultiplicationExample,
    override: Optional[str],
) -> MultiplicationExample:
    if override is None:
        return example
    return MultiplicationExample(
        a=example.a,
        b=example.b,
        digits=example.digits,
        result=example.result,
        operand_width=example.operand_width,
        format_version=example.format_version,
        target_override=override,
    )


def random_int_with_exact_digits(num_digits: int, rng: random.Random) -> int:
    if num_digits <= 0:
        raise ValueError("num_digits must be positive.")
    if num_digits == 1:
        return rng.randint(0, 9)
    low = 10 ** (num_digits - 1)
    high = (10**num_digits) - 1
    return rng.randint(low, high)


def generate_multiplication_seed_example(
    block_size: int,
    rng: random.Random,
    format_version: str = "legacy",
) -> MultiplicationExample:
    upper = (10**block_size) - 1
    a = rng.randint(0, upper)
    b = rng.randint(0, upper)
    return MultiplicationExample(
        a=a,
        b=b,
        digits=block_size,
        result=a * b,
        operand_width=block_size,
        format_version=format_version,
    )


def generate_long_multiplication_example(
    digits: int,
    rng: random.Random,
    format_version: str = "legacy",
) -> MultiplicationExample:
    a = random_int_with_exact_digits(digits, rng)
    b = random_int_with_exact_digits(digits, rng)
    return MultiplicationExample(
        a=a,
        b=b,
        digits=digits,
        result=a * b,
        operand_width=digits,
        format_version=format_version,
    )


def iter_multiplication_sizes(min_digits: int, max_digits: int, block_size: int) -> List[int]:
    if max_digits < min_digits:
        return []
    return list(range(min_digits, max_digits + 1))


def split_value_into_blocks(value: int, total_digits: int, block_size: int) -> List[int]:
    text = f"{value:0{total_digits}d}"
    blocks: List[int] = []
    for end in range(len(text), 0, -block_size):
        start = max(0, end - block_size)
        blocks.append(int(text[start:end]))
    return blocks


def analyze_partial_products(partials: Sequence[Dict[str, int]]) -> Tuple[int, int]:
    digit_fan_in: Dict[int, int] = defaultdict(int)
    column_sums: Dict[int, int] = defaultdict(int)
    max_position = 0
    for partial in partials:
        product = partial["a"] * partial["b"]
        shift = partial["shift"]
        text = str(product)
        for offset, digit_char in enumerate(reversed(text)):
            position = shift + offset
            digit_fan_in[position] += 1
            column_sums[position] += int(digit_char)
            max_position = max(max_position, position)

    max_overlap = max(digit_fan_in.values()) if digit_fan_in else 0

    carry_count = 0
    carry = 0
    position = 0
    while position <= max_position or carry > 0:
        total = column_sums.get(position, 0) + carry
        if total >= 10:
            carry_count += 1
        carry = total // 10
        position += 1
    return max_overlap, carry_count


def build_multiplication_component_payload(
    example: MultiplicationExample,
    block_size: int,
) -> Dict[str, Any]:
    a_blocks = split_value_into_blocks(example.a, example.digits, block_size)
    b_blocks = split_value_into_blocks(example.b, example.digits, block_size)
    partials: List[Dict[str, int]] = []
    for i, block_a in enumerate(a_blocks):
        for j, block_b in enumerate(b_blocks):
            partials.append(
                {
                    "a": block_a,
                    "b": block_b,
                    "shift": (i + j) * block_size,
                }
            )
    max_overlap, carry_count = analyze_partial_products(partials)
    return {
        "partials": partials,
        "max_overlap": int(max_overlap),
        "carry_count": int(carry_count),
        "block_size": int(block_size),
    }


def build_multiplication_seed_dataset(
    *,
    block_size: int,
    per_split_counts: Dict[str, int],
    rng: random.Random,
    exclude_keys: Optional[set[Tuple[int, int, int]]] = None,
    record_keys: Optional[Dict[str, set[Tuple[int, int, int]]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 10_000,
    format_version: str = "legacy",
) -> Dict[str, List[MultiplicationExample]]:
    splits = {key: [] for key in ("train", "validation", "test")}
    occupied = set(exclude_keys) if exclude_keys else set()
    requested_total = sum(per_split_counts.get(split, 0) for split in ("train", "validation", "test"))
    universe = 10 ** (2 * block_size)
    available_unique = max(0, universe - len(occupied))
    if requested_total > available_unique:
        print(
            f"[WARN] Requested {requested_total} multiplication seed examples exceeds available unique pairs ({available_unique}); capping.",
            flush=True,
        )
    remaining_total = available_unique
    adjusted_counts = {}
    for split in ("train", "validation", "test"):
        requested = per_split_counts.get(split, 0)
        adjusted = min(requested, remaining_total)
        adjusted_counts[split] = adjusted
        remaining_total -= adjusted

    generated: List[Tuple[MultiplicationExample, Tuple[int, int, int], bool]] = []
    attempts = 0
    duplicates_allowed = False
    total_needed = sum(adjusted_counts.values())
    while len(generated) < total_needed:
        attempts += 1
        example = generate_multiplication_seed_example(block_size, rng, format_version=format_version)
        key = multiplication_key(example)
        if key in occupied:
            if attempts >= max_attempts:
                if not duplicates_allowed:
                    print(
                        f"[WARN] Exhausted unique multiplication seed sampling (progress={progress_name}); allowing duplicates.",
                        flush=True,
                    )
                    duplicates_allowed = True
                generated.append((example, key, True))
                attempts = 0
            continue
        occupied.add(key)
        generated.append((example, key, False))
        attempts = 0

    index = 0
    for split in ("train", "validation", "test"):
        count = adjusted_counts[split]
        if count <= 0:
            continue
        chunk = generated[index : index + count]
        index += count
        splits[split].extend(example for example, _, _ in chunk)
        if record_keys and split in record_keys:
            for _, key, is_duplicate in chunk:
                if not is_duplicate:
                    record_keys[split].add(key)
        if progress_name:
            print(
                f"[INFO] Generated {len(chunk)}/{count} {progress_name} examples for split='{split}' digits={block_size}",
                flush=True,
            )
    for split in splits:
        rng.shuffle(splits[split])
    return splits


def build_multiplication_long_dataset(
    *,
    min_digits: int,
    max_digits: int,
    per_digit_counts: Dict[str, int],
    rng: random.Random,
    block_size: int,
    exclude_keys: Optional[set[Tuple[int, int, int]]] = None,
    record_keys: Optional[Dict[str, set[Tuple[int, int, int]]]] = None,
    progress_name: Optional[str] = None,
    record_components: Optional[Dict[str, Dict[Tuple[int, int, int], Dict[str, Any]]]] = None,
    max_attempts: int = 50_000,
    format_version: str = "legacy",
) -> Dict[str, List[MultiplicationExample]]:
    splits = {key: [] for key in ("train", "validation", "test")}
    occupied = set(exclude_keys) if exclude_keys else set()
    sizes = iter_multiplication_sizes(min_digits, max_digits, block_size)
    per_split_counts = {key: int(per_digit_counts.get(key, 0)) for key in ("train", "validation", "test")}

    for digits in sizes:
        count_per_size = {split: per_split_counts.get(split, 0) for split in ("train", "validation", "test")}
        requested_total = sum(count_per_size.values())
        if requested_total <= 0:
            continue

        value_count = 10 if digits == 1 else 9 * (10 ** (digits - 1))
        total_unique = value_count * value_count
        already_used = sum(1 for key in occupied if key[0] == digits)
        available_unique = max(0, total_unique - already_used)
        if requested_total > available_unique:
            print(
                f"[WARN] Requested {requested_total} multiplication examples exceeds available unique pairs "
                f"({available_unique}) for digits={digits}; capping.",
                flush=True,
            )
        remaining_total = available_unique
        adjusted_counts: Dict[str, int] = {}
        for split in ("validation", "test", "train"):
            requested = count_per_size.get(split, 0)
            adjusted = min(requested, remaining_total)
            adjusted_counts[split] = adjusted
            remaining_total -= adjusted

        generated: List[Tuple[MultiplicationExample, Tuple[int, int, int], Dict[str, Any], bool]] = []
        attempts = 0
        duplicates_allowed = False
        total_needed = sum(adjusted_counts.values())
        while len(generated) < total_needed:
            attempts += 1
            example = generate_long_multiplication_example(digits, rng, format_version=format_version)
            key = multiplication_key(example)
            if key in occupied:
                if attempts >= max_attempts:
                    if not duplicates_allowed:
                        print(
                            f"[WARN] Exhausted unique multiplication sampling (digits={digits}); allowing duplicates.",
                            flush=True,
                        )
                        duplicates_allowed = True
                    generated.append((example, key, {}, True))
                    attempts = 0
                continue
            payload = build_multiplication_component_payload(example, block_size)
            occupied.add(key)
            generated.append((example, key, payload, False))
            attempts = 0

        index = 0
        for split in ("train", "validation", "test"):
            count = adjusted_counts.get(split, 0)
            if count <= 0:
                continue
            chunk = generated[index : index + count]
            index += count
            splits[split].extend(example for example, _, _, _ in chunk)
            if record_keys and split in record_keys:
                for _, key, _, is_duplicate in chunk:
                    if not is_duplicate:
                        record_keys[split].add(key)
            if record_components is not None:
                component_map = record_components.setdefault(split, {})
                for _, key, payload, is_duplicate in chunk:
                    if is_duplicate:
                        continue
                    component_map[key] = payload
            if progress_name:
                print(
                    f"[INFO] Generated {len(chunk)}/{count} {progress_name} examples for split='{split}' digits={digits}",
                    flush=True,
                )

    for split in splits:
        rng.shuffle(splits[split])
    return splits


def get_multiplication_slice_name(payload: Dict[str, Any], block_size: int) -> str:
    max_overlap = int(payload.get("max_overlap", 0))
    carry_count = int(payload.get("carry_count", 0))
    overlap_tag = "low_overlap" if max_overlap <= 2 else "high_overlap"
    carry_tag = "low_carry" if carry_count <= block_size else "high_carry"
    return f"{overlap_tag}_{carry_tag}"


class AdditionTask(SelfImprovementTask):
    name = "addition"
    size_label = "digits"
    size_alias_singular = "digit"
    size_alias_plural = "digits"

    def validate_args(self, args: Any) -> None:
        if args.composition_error_percent < 0.0 or args.composition_error_percent > 100.0:
            raise ValueError("composition_error_percent must be between 0 and 100.")
        if args.corruption_rate < 0.0 or args.corruption_rate > 1.0:
            raise ValueError("corruption_rate must be between 0 and 1.")
        if getattr(args, "addition_width_mode", ADDITION_WIDTH_EXACT_DIGITS) not in ADDITION_WIDTH_MODES:
            raise ValueError(f"Unsupported addition_width_mode={args.addition_width_mode!r}.")
        if getattr(args, "addition_sampling_mode", ADDITION_SAMPLING_NATURAL) not in ADDITION_SAMPLING_MODES:
            raise ValueError(f"Unsupported addition_sampling_mode={args.addition_sampling_mode!r}.")
        if (
            getattr(args, "addition_sampling_mode", ADDITION_SAMPLING_NATURAL) != ADDITION_SAMPLING_NATURAL
            and getattr(args, "addition_width_mode", ADDITION_WIDTH_EXACT_DIGITS) != ADDITION_WIDTH_FIXED_MIXED_PROMPT
        ):
            raise ValueError("Balanced addition sampling requires --addition-width-mode fixed_width_mixed_prompt.")
        if getattr(args, "addition_composition_path_mode", COMPOSITION_PATH_RANDOM) not in COMPOSITION_PATH_MODES:
            raise ValueError(
                f"Unsupported addition_composition_path_mode={args.addition_composition_path_mode!r}."
            )

    def serialize_example(self, example: AdditionExample) -> JsonDict:
        return {
            "a": example.a,
            "b": example.b,
            "result": example.result,
            "digits": example.digits,
            "operand_width": example.block_width,
            "has_carry": example.has_carry,
            "target_override": example.target_override,
        }

    def deserialize_example(self, payload: JsonDict) -> AdditionExample:
        return AdditionExample(
            a=int(payload["a"]),
            b=int(payload["b"]),
            result=int(payload["result"]),
            digits=int(payload["digits"]),
            has_carry=bool(payload["has_carry"]),
            target_override=payload.get("target_override"),
            operand_width=int(payload.get("operand_width", payload["digits"])),
        )

    def save_component_map(
        self,
        path: Path,
        component_map: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {encode_key(key): [encode_key(child) for child in children] for key, children in component_map.items()}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load_component_map(self, path: Path) -> Dict[Tuple[int, int, int], List[Tuple[int, int, int]]]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return {decode_key(key): [decode_key(child) for child in children] for key, children in raw.items()}

    def _allow_carry_for_composed(self, args: Any) -> bool:
        return args.composed_strategy in ("with_carry", "with_carry_filtered")

    def _boundary_carry_policy_for_composed(self, args: Any) -> str:
        if args.composed_strategy == "with_carry_filtered":
            return "no_boundary_carry"
        return "any"

    def prepare_initial_splits(
        self,
        rng: random.Random,
        args: Any,
    ) -> Tuple[Dict[SplitName, List[AdditionExample]], Dict[SplitName, set[Tuple[int, int, int]]]]:
        return prepare_addition_initial_splits(
            rng,
            args.initial_min_size,
            args.initial_max_size,
            args.initial_train_per_size,
            args.initial_eval_per_size,
            addition_width_mode=getattr(args, "addition_width_mode", ADDITION_WIDTH_EXACT_DIGITS),
            addition_sampling_mode=getattr(args, "addition_sampling_mode", ADDITION_SAMPLING_NATURAL),
        )

    def prepare_composed_train(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[AdditionExample]],
        base_records: Dict[SplitName, set[Tuple[int, int, int]]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Tuple[int, int, int]]] = None,
    ) -> Tuple[List[AdditionExample], Dict[Tuple[int, int, int], List[Tuple[int, int, int]]], set[Tuple[int, int, int]]]:
        return prepare_addition_composed_train(
            rng,
            base_splits,
            base_records,
            min_size,
            max_size,
            args.expand_train_per_size,
            allow_carry=self._allow_carry_for_composed(args),
            boundary_carry_policy=self._boundary_carry_policy_for_composed(args),
            additional_exclude=additional_exclude,
            addition_width_mode=getattr(args, "addition_width_mode", ADDITION_WIDTH_EXACT_DIGITS),
            composition_path_mode=getattr(args, "addition_composition_path_mode", COMPOSITION_PATH_RANDOM),
        )

    def prepare_composed_eval(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[AdditionExample]],
        base_records: Dict[SplitName, set[Tuple[int, int, int]]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Tuple[int, int, int]]] = None,
    ) -> Tuple[List[AdditionExample], Dict[Tuple[int, int, int], List[Tuple[int, int, int]]], set[Tuple[int, int, int]]]:
        return prepare_addition_composed_eval(
            rng,
            base_splits,
            base_records,
            min_size,
            max_size,
            args.composed_eval_per_size,
            additional_exclude=additional_exclude,
            addition_width_mode=getattr(args, "addition_width_mode", ADDITION_WIDTH_EXACT_DIGITS),
            composition_path_mode=getattr(args, "addition_composition_path_mode", COMPOSITION_PATH_RANDOM),
        )

    def prepare_eval_examples(
        self,
        rng: random.Random,
        args: Any,
        min_size: int,
        max_size: int,
        exclude: set[Tuple[int, int, int]],
    ) -> List[AdditionExample]:
        return prepare_addition_eval_examples(
            rng,
            min_size,
            max_size,
            args.eval_per_size,
            exclude,
            addition_width_mode=getattr(args, "addition_width_mode", ADDITION_WIDTH_EXACT_DIGITS),
            addition_sampling_mode=getattr(args, "addition_sampling_mode", ADDITION_SAMPLING_NATURAL),
        )

    def split_composed_eval_slices(
        self,
        examples: Sequence[AdditionExample],
        component_map: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]],
    ) -> Dict[str, List[AdditionExample]]:
        return split_addition_examples_by_boundary_status(examples, component_map)

    def keys_for_examples(self, examples: Sequence[AdditionExample]) -> set[Tuple[int, int, int]]:
        return {example_key(example) for example in examples}

    def rebuild_records(
        self,
        splits: Dict[SplitName, List[AdditionExample]],
    ) -> Dict[SplitName, set[Tuple[int, int, int]]]:
        return {split: {example_key(example) for example in splits.get(split, [])} for split in ("train", "validation", "test")}

    def key_for_example(self, example: AdditionExample) -> Tuple[int, int, int]:
        return example_key(example)

    def clone_with_override(self, example: AdditionExample, override: Optional[str]) -> AdditionExample:
        return clone_with_override(example, override)

    def size_of(self, example: AdditionExample) -> int:
        return example.digits

    def prediction_parser(self, text: str) -> Optional[str]:
        return extract_numeric_answer(text)

    def derive_round_targets(
        self,
        model: Any,
        tokenizer: Any,
        composed_examples: Sequence[AdditionExample],
        component_map: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]],
        target_max_size: int,
        base_examples: Sequence[AdditionExample],
        *,
        batch_size: int,
        decode_max_new_tokens: int,
        args: Any,
        rng: random.Random,
    ) -> Tuple[List[AdditionExample], int, JsonDict]:
        candidate_examples = [example for example in composed_examples if example.digits <= target_max_size]
        if args.pseudo_label_mode == "direct":
            return build_direct_pseudo_examples(
                candidate_examples,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                decode_max_new_tokens=decode_max_new_tokens,
                key_getter=self.key_for_example,
                prediction_parser=self.prediction_parser,
                clone_builder=self.clone_with_override,
                mode="direct",
            )
        if args.pseudo_label_mode not in {"compose", "compose_corrupt"}:
            return [], 0, {
                "mode": args.pseudo_label_mode,
                "candidate_total": len(candidate_examples),
                "retained_total": 0,
                "missing_total": 0,
            }

        filter_component_carries = args.composed_strategy == "with_carry_filtered"
        carry_error_fraction = args.composition_error_percent / 100.0
        candidate_keys = {example_key(example) for example in candidate_examples}
        base_predictions = generate_prediction_map(
            model=model,
            tokenizer=tokenizer,
            examples=base_examples,
            batch_size=batch_size,
            max_new_tokens=decode_max_new_tokens,
            key_getter=example_key,
            prediction_parser=self.prediction_parser,
        )
        base_map = {
            key: base_predictions[key]
            for key in (example_key(example) for example in base_examples)
            if key in base_predictions
        }
        component_subset = {key: component_map[key] for key in component_map if key in candidate_keys}
        pseudo_map = build_composed_pseudo_map(
            base_map,
            candidate_examples,
            component_subset,
            base_predictions,
            filter_component_carries=filter_component_carries,
            carry_error_fraction=carry_error_fraction if filter_component_carries else 0.0,
            rng=rng,
        )

        candidate_boundary = 0
        candidate_no_boundary = 0
        candidate_unknown = 0
        kept_boundary = 0
        kept_no_boundary = 0
        kept_unknown = 0
        missing_boundary = 0
        missing_no_boundary = 0
        missing_unknown = 0
        corrupted_total = 0

        pseudo_examples: List[AdditionExample] = []
        missing_labels = 0
        for example in candidate_examples:
            status = get_boundary_carry_status(example, component_subset)
            if status is True:
                candidate_boundary += 1
            elif status is False:
                candidate_no_boundary += 1
            else:
                candidate_unknown += 1

            override = pseudo_map.get(example_key(example))
            if override is None:
                missing_labels += 1
                if status is True:
                    missing_boundary += 1
                elif status is False:
                    missing_no_boundary += 1
                else:
                    missing_unknown += 1
                continue

            if args.pseudo_label_mode == "compose_corrupt" and rng.random() < args.corruption_rate:
                override = corrupt_numeric_target(override)
                corrupted_total += 1

            pseudo_examples.append(clone_with_override(example, override))
            if status is True:
                kept_boundary += 1
            elif status is False:
                kept_no_boundary += 1
            else:
                kept_unknown += 1

        diagnostics: JsonDict = {
            "mode": args.pseudo_label_mode,
            "target_max_digits": int(target_max_size),
            "candidate_total": len(candidate_examples),
            "candidate_boundary_carry": candidate_boundary,
            "candidate_no_boundary_carry": candidate_no_boundary,
            "candidate_unknown_boundary": candidate_unknown,
            "retained_total": len(pseudo_examples),
            "retained_boundary_carry": kept_boundary,
            "retained_no_boundary_carry": kept_no_boundary,
            "retained_unknown_boundary": kept_unknown,
            "missing_total": missing_labels,
            "missing_boundary_carry": missing_boundary,
            "missing_no_boundary_carry": missing_no_boundary,
            "missing_unknown_boundary": missing_unknown,
            "retained_boundary_fraction": kept_boundary / candidate_boundary if candidate_boundary > 0 else math.nan,
            "retained_no_boundary_fraction": kept_no_boundary / candidate_no_boundary if candidate_no_boundary > 0 else math.nan,
            "retained_unknown_fraction": kept_unknown / candidate_unknown if candidate_unknown > 0 else math.nan,
            "filter_component_carries": bool(filter_component_carries),
            "carry_error_fraction": carry_error_fraction if filter_component_carries else 0.0,
            "corruption_rate": args.corruption_rate if args.pseudo_label_mode == "compose_corrupt" else 0.0,
            "corrupted_total": corrupted_total,
        }
        return pseudo_examples, missing_labels, diagnostics

    def build_task_metadata(self, args: Any, final_max_size: int) -> JsonDict:
        return {
            "composed_strategy": args.composed_strategy,
            "filter_component_carries": args.composed_strategy == "with_carry_filtered",
            "composed_boundary_carry_policy": self._boundary_carry_policy_for_composed(args),
            "composition_error_percent": args.composition_error_percent,
            "pseudo_label_mode": args.pseudo_label_mode,
            "corruption_rate": args.corruption_rate,
        }

    def metadata_aliases(self, args: Any, final_max_size: int) -> JsonDict:
        return {
            "initial_min_digits": args.initial_min_size,
            "initial_max_digits": args.initial_max_size,
            "expand_num_digits": args.expand_num_size,
            "expand_train_per_digit": args.expand_train_per_size,
            "eval_per_digit": args.eval_per_size,
            "composed_eval_per_digit": args.composed_eval_per_size,
            "composed_max_digits": final_max_size,
            "composed_strategy": args.composed_strategy,
            "composed_without_carry": args.composed_strategy == "without_carry",
            "filter_component_carries": args.composed_strategy == "with_carry_filtered",
            "composed_boundary_carry_policy": self._boundary_carry_policy_for_composed(args),
            "composition_error_percent": args.composition_error_percent,
            "pseudo_label_mode": args.pseudo_label_mode,
            "corruption_rate": args.corruption_rate,
        }

    def validate_loaded_metadata(
        self,
        args: Any,
        metadata: JsonDict,
        final_max_size: int,
        dynamic_composed: bool,
    ) -> None:
        task_config = metadata.get("task_config", {}) if isinstance(metadata.get("task_config"), dict) else {}
        stored_strategy = task_config.get("composed_strategy", metadata.get("composed_strategy"))
        if stored_strategy is None:
            stored_strategy = "without_carry" if metadata.get("composed_without_carry", False) else "with_carry"
        stored_allow_carry = stored_strategy in ("with_carry", "with_carry_filtered")
        if stored_allow_carry != self._allow_carry_for_composed(args):
            raise ValueError(
                "Stored composed dataset carry configuration does not match current --composed-strategy. "
                "Please regenerate datasets or choose a compatible strategy."
            )
        stored_filter_flag = bool(task_config.get("filter_component_carries", metadata.get("filter_component_carries", False)))
        stored_boundary_policy = task_config.get(
            "composed_boundary_carry_policy",
            metadata.get("composed_boundary_carry_policy"),
        )
        expected_boundary_policy = self._boundary_carry_policy_for_composed(args)
        if stored_boundary_policy is None:
            stored_boundary_policy = "any"
        if stored_boundary_policy != expected_boundary_policy:
            if (
                args.composed_strategy == "with_carry_filtered"
                and stored_boundary_policy == "any"
                and stored_filter_flag
            ):
                print(
                    "[INFO] Stored metadata predates explicit boundary-carry buckets; "
                    "reusing the broad composed pool and filtering pseudo labels on-the-fly.",
                    flush=True,
                )
            else:
                raise ValueError(
                    "Stored composed dataset boundary-carry bucket does not match current --composed-strategy. "
                    "Please regenerate datasets or choose a compatible strategy."
                )
        if args.composed_strategy == "with_carry_filtered" and not stored_filter_flag:
            print(
                "[INFO] Stored metadata indicates composed dataset was generated without filtering carries; "
                "pseudo labels will be filtered on-the-fly.",
                flush=True,
            )
        stored_error_percent = float(task_config.get("composition_error_percent", metadata.get("composition_error_percent", 0.0)))
        if abs(stored_error_percent - args.composition_error_percent) > 1e-6:
            raise ValueError(
                "Stored dataset was created with a different composition_error_percent; please regenerate datasets or "
                "specify a matching value."
            )

    def summary_payload_aliases(self, summary: Any) -> JsonDict:
        boundary = summary.composed_eval_slices.get("boundary_carry")
        no_boundary = summary.composed_eval_slices.get("no_boundary_carry")
        unknown = summary.composed_eval_slices.get("unknown")
        return {
            "max_digits": summary.max_size,
            "per_digit_accuracy": {str(size): score for size, score in summary.per_size_accuracy.items()},
            "max_digits_at_90_accuracy": max(
                [size for size, score in summary.per_size_accuracy.items() if score >= 0.90],
                default=None,
            ),
            "stitched_boundary_carry_accuracy": boundary.accuracy if boundary else None,
            "stitched_no_boundary_carry_accuracy": no_boundary.accuracy if no_boundary else None,
            "stitched_unknown_accuracy": unknown.accuracy if unknown else None,
            "stitched_boundary_carry_count": boundary.count if boundary else 0,
            "stitched_no_boundary_carry_count": no_boundary.count if no_boundary else 0,
            "stitched_unknown_count": unknown.count if unknown else 0,
        }


class MajorityTask(SelfImprovementTask):
    name = "majority"
    size_label = "bits"
    size_alias_singular = "bit"
    size_alias_plural = "bits"

    def validate_args(self, args: Any) -> None:
        corruption_rate = float(getattr(args, "corruption_rate", 0.0))
        pseudo_label_mode = str(getattr(args, "pseudo_label_mode", "none"))
        if corruption_rate < 0.0 or corruption_rate > 1.0:
            raise ValueError("corruption_rate must be between 0 and 1.")
        format_version = normalize_task_format_version(args)
        if format_version not in MAJORITY_FORMATS:
            raise ValueError(f"Unsupported majority format_version={format_version!r}.")
        target_mode = normalize_bit_target_mode(args)
        if target_mode not in BIT_TARGET_MODES:
            raise ValueError(f"Unsupported majority target_mode={target_mode!r}.")
        if target_mode == "symbol_run_pair":
            raise ValueError("symbol_run_pair target mode is only supported for run_length.")
        compose_arity = normalize_compose_arity(args)
        if compose_arity not in BIT_COMPOSE_ARITIES:
            raise ValueError(f"Unsupported majority compose_arity={compose_arity!r}.")
        guarded_rule = normalize_guarded_compose_rule(args)
        if guarded_rule not in BIT_GUARDED_COMPOSE_RULES:
            raise ValueError(f"Unsupported majority guarded_compose_rule={guarded_rule!r}.")
        if target_mode == "plain_output" and pseudo_label_mode in {"compose", "compose_corrupt"}:
            if guarded_rule != "majority_agree_pair":
                raise ValueError(
                    "majority plain_output compose mode requires --guarded-compose-rule majority_agree_pair."
                )
            if compose_arity != "exact2":
                raise ValueError("majority plain_output compose mode requires --compose-arity exact2.")
            if pseudo_label_mode == "compose_corrupt":
                raise ValueError("majority plain_output diagnostics do not support compose_corrupt.")

    def serialize_example(self, example: MajorityExample) -> JsonDict:
        return {
            "bitstring": example.bitstring,
            "bits": example.bits,
            "ones": example.ones,
            "majority": example.majority,
            "format_version": example.format_version,
            "target_mode": example.target_mode,
            "target_override": example.target_override,
        }

    def deserialize_example(self, payload: JsonDict) -> MajorityExample:
        return MajorityExample(
            bitstring=str(payload["bitstring"]),
            bits=int(payload["bits"]),
            ones=int(payload["ones"]),
            majority=int(payload["majority"]),
            format_version=str(payload.get("format_version", "legacy")),
            target_mode=str(payload.get("target_mode", "default")),
            target_override=payload.get("target_override"),
        )

    def save_component_map(self, path: Path, component_map: Dict[Tuple[int, str], List[Tuple[int, str]]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            encode_majority_key(key): [encode_majority_key(child) for child in children]
            for key, children in component_map.items()
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load_component_map(self, path: Path) -> Dict[Tuple[int, str], List[Tuple[int, str]]]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return {decode_majority_key(key): [decode_majority_key(child) for child in children] for key, children in raw.items()}

    def prepare_initial_splits(
        self,
        rng: random.Random,
        args: Any,
    ) -> Tuple[Dict[SplitName, List[MajorityExample]], Dict[SplitName, set[Tuple[int, str]]]]:
        splits = {name: [] for name in ("train", "validation", "test")}
        records: Dict[SplitName, set[Tuple[int, str]]] = {name: set() for name in splits}
        split_order = ("validation", "test", "train") if getattr(args, "reserve_heldout_first", False) else (
            "train",
            "validation",
            "test",
        )
        generated = build_majority_length_bucket_dataset(
            min_bits=args.initial_min_size,
            max_bits=args.initial_max_size,
            per_bit_counts={
                "train": args.initial_train_per_size,
                "validation": args.initial_eval_per_size,
                "test": args.initial_eval_per_size,
            },
            rng=rng,
            exclude_keys=getattr(args, "_initial_exclude_keys", None),
            record_keys=records,
            progress_name="initial",
            format_version=normalize_task_format_version(args),
            target_mode=normalize_bit_target_mode(args),
            split_order=split_order,
        )
        for split in splits:
            splits[split] = generated.get(split, [])
        return splits, records

    def prepare_composed_train(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[MajorityExample]],
        base_records: Dict[SplitName, set[Tuple[int, str]]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Tuple[int, str]]] = None,
    ) -> Tuple[List[MajorityExample], Dict[Tuple[int, str], List[Tuple[int, str]]], set[Tuple[int, str]]]:
        if max_size < min_size or args.expand_train_per_size <= 0:
            return [], {}, set()
        composed_records: Dict[SplitName, set[Tuple[int, str]]] = {"train": set(), "validation": set(), "test": set()}
        component_records: Dict[SplitName, Dict[Tuple[int, str], List[Tuple[int, str]]]] = {
            "train": {},
            "validation": {},
            "test": {},
        }
        base_used = set().union(*base_records.values())
        if additional_exclude:
            base_used.update(additional_exclude)
        compose_arity = normalize_compose_arity(args)
        if compose_arity == "exact2":
            target_sizes = exact2_reachable_sizes_from_examples(
                base_splits.get("train", []),
                size_getter=lambda example: example.bits,
                min_size=min_size,
                max_size=max_size,
            )
            train_examples: List[MajorityExample] = []
            for bits in target_sizes:
                composed_splits = build_majority_composed_dataset(
                    base_splits=base_splits,
                    min_bits=bits,
                    max_bits=bits,
                    per_bit_counts={"train": args.expand_train_per_size, "validation": 0, "test": 0},
                    rng=rng,
                    exclude_keys=base_used,
                    record_keys=composed_records,
                    progress_name="composed",
                    record_components=component_records,
                    compose_arity=compose_arity,
                )
                train_examples.extend(composed_splits.get("train", []))
            return train_examples, component_records.get("train", {}), composed_records.get("train", set())
        composed_splits = build_majority_composed_dataset(
            base_splits=base_splits,
            min_bits=min_size,
            max_bits=max_size,
            per_bit_counts={"train": args.expand_train_per_size, "validation": 0, "test": 0},
            rng=rng,
            exclude_keys=base_used,
            record_keys=composed_records,
            progress_name="composed",
            record_components=component_records,
            compose_arity=compose_arity,
        )
        return composed_splits.get("train", []), component_records.get("train", {}), composed_records.get("train", set())

    def prepare_composed_eval(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[MajorityExample]],
        base_records: Dict[SplitName, set[Tuple[int, str]]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Tuple[int, str]]] = None,
    ) -> Tuple[List[MajorityExample], Dict[Tuple[int, str], List[Tuple[int, str]]], set[Tuple[int, str]]]:
        if max_size < min_size or args.composed_eval_per_size <= 0:
            return [], {}, set()
        composed_records: Dict[SplitName, set[Tuple[int, str]]] = {"train": set(), "validation": set(), "test": set()}
        component_records: Dict[SplitName, Dict[Tuple[int, str], List[Tuple[int, str]]]] = {
            "train": {},
            "validation": {},
            "test": {},
        }
        base_used = set().union(*base_records.values())
        if additional_exclude:
            base_used.update(additional_exclude)
        stitched_base_splits = {
            "train": list(base_splits.get("train", [])),
            "validation": list(base_splits.get("train", [])),
            "test": list(base_splits.get("train", [])),
        }
        compose_arity = normalize_compose_arity(args)
        if compose_arity == "exact2":
            target_sizes = exact2_reachable_sizes_from_examples(
                stitched_base_splits.get("train", []),
                size_getter=lambda example: example.bits,
                min_size=min_size,
                max_size=max_size,
            )
            test_examples: List[MajorityExample] = []
            for bits in target_sizes:
                composed_splits = build_majority_composed_dataset(
                    base_splits=stitched_base_splits,
                    min_bits=bits,
                    max_bits=bits,
                    per_bit_counts={"train": 0, "validation": 0, "test": args.composed_eval_per_size},
                    rng=rng,
                    exclude_keys=base_used,
                    record_keys=composed_records,
                    progress_name="composed-eval",
                    record_components=component_records,
                    compose_arity=compose_arity,
                )
                test_examples.extend(composed_splits.get("test", []))
            return test_examples, component_records.get("test", {}), composed_records.get("test", set())
        composed_splits = build_majority_composed_dataset(
            base_splits=stitched_base_splits,
            min_bits=min_size,
            max_bits=max_size,
            per_bit_counts={"train": 0, "validation": 0, "test": args.composed_eval_per_size},
            rng=rng,
            exclude_keys=base_used,
            record_keys=composed_records,
            progress_name="composed-eval",
            record_components=component_records,
            compose_arity=compose_arity,
        )
        return composed_splits.get("test", []), component_records.get("test", {}), composed_records.get("test", set())

    def prepare_eval_examples(
        self,
        rng: random.Random,
        args: Any,
        min_size: int,
        max_size: int,
        exclude: set[Tuple[int, str]],
    ) -> List[MajorityExample]:
        generated = build_majority_length_bucket_dataset(
            min_bits=min_size,
            max_bits=max_size,
            per_bit_counts={"train": 0, "validation": 0, "test": args.eval_per_size},
            rng=rng,
            exclude_keys=exclude,
            record_keys={split: set() for split in ("train", "validation", "test")},
            progress_name="evaluation",
            format_version=normalize_task_format_version(args),
            target_mode=normalize_bit_target_mode(args),
        )
        return list(generated.get("test", []))

    def split_composed_eval_slices(
        self,
        examples: Sequence[MajorityExample],
        component_map: Dict[Tuple[int, str], List[Tuple[int, str]]],
    ) -> Dict[str, List[MajorityExample]]:
        if examples and examples[0].target_mode == "plain_output":
            return guard_slice_partition(
                examples,
                component_map,
                key_getter=majority_key,
                guard_fn=majority_guard_accepts_true_components,
            )
        return {"all": list(examples)}

    def keys_for_examples(self, examples: Sequence[MajorityExample]) -> set[Tuple[int, str]]:
        return {majority_key(example) for example in examples}

    def rebuild_records(self, splits: Dict[SplitName, List[MajorityExample]]) -> Dict[SplitName, set[Tuple[int, str]]]:
        return {split: {majority_key(example) for example in splits.get(split, [])} for split in ("train", "validation", "test")}

    def key_for_example(self, example: MajorityExample) -> Tuple[int, str]:
        return majority_key(example)

    def clone_with_override(self, example: MajorityExample, override: Optional[str]) -> MajorityExample:
        return clone_majority_with_override(example, override)

    def size_of(self, example: MajorityExample) -> int:
        return example.bits

    def prediction_parser(self, text: str, example: Optional[MajorityExample] = None) -> Optional[str]:
        return parse_majority_prediction(text, example)

    def token_initializers(self, args: Any) -> Dict[str, str]:
        return {}

    def derive_round_targets(
        self,
        model: Any,
        tokenizer: Any,
        composed_examples: Sequence[MajorityExample],
        component_map: Dict[Tuple[int, str], List[Tuple[int, str]]],
        target_max_size: int,
        base_examples: Sequence[MajorityExample],
        *,
        batch_size: int,
        decode_max_new_tokens: int,
        args: Any,
        rng: random.Random,
    ) -> Tuple[List[MajorityExample], int, JsonDict]:
        candidate_examples = [example for example in composed_examples if example.bits <= target_max_size]
        target_mode = normalize_bit_target_mode(args)
        if args.pseudo_label_mode == "direct":
            return build_direct_pseudo_examples(
                candidate_examples,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                decode_max_new_tokens=decode_max_new_tokens,
                key_getter=self.key_for_example,
                prediction_parser=self.prediction_parser,
                clone_builder=self.clone_with_override,
                mode="direct",
            )
        if target_mode == "plain_output":
            base_predictions = generate_prediction_map(
                model=model,
                tokenizer=tokenizer,
                examples=base_examples,
                batch_size=batch_size,
                max_new_tokens=decode_max_new_tokens,
                key_getter=majority_key,
                prediction_parser=self.prediction_parser,
            )

            def evaluate_candidate(
                example: MajorityExample,
                component_keys: Optional[Sequence[Tuple[int, str]]],
            ) -> Tuple[str, Optional[str]]:
                if not component_keys or len(component_keys) != 2:
                    return "missing", None
                predictions: List[int] = []
                for component_key in component_keys:
                    prediction = base_predictions.get(component_key)
                    if prediction is None:
                        return "missing", None
                    try:
                        value = int(prediction)
                    except ValueError:
                        return "missing", None
                    if value not in (0, 1):
                        return "missing", None
                    predictions.append(value)
                if predictions[0] != predictions[1]:
                    return "rejected", None
                return "accepted", str(predictions[0])

            def refill_builder(
                bits: int,
                need: int,
                occupied_keys: set[Tuple[int, str]],
            ) -> Tuple[List[MajorityExample], Dict[Tuple[int, str], List[Tuple[int, str]]]]:
                record_components = {"train": {}, "validation": {}, "test": {}}
                refill_splits = build_majority_composed_dataset(
                    base_splits={"train": list(base_examples), "validation": [], "test": []},
                    min_bits=bits,
                    max_bits=bits,
                    per_bit_counts={"train": need, "validation": 0, "test": 0},
                    rng=rng,
                    exclude_keys=occupied_keys,
                    record_keys={"train": set(), "validation": set(), "test": set()},
                    progress_name="guarded-refill",
                    record_components=record_components,
                    compose_arity=normalize_compose_arity(args),
                )
                return refill_splits.get("train", []), record_components.get("train", {})

            return build_guarded_bit_pseudo_examples(
                candidate_examples,
                component_map,
                target_max_size=target_max_size,
                requested_per_size=args.expand_train_per_size,
                size_getter=self.size_of,
                key_getter=self.key_for_example,
                clone_builder=self.clone_with_override,
                evaluate_candidate=evaluate_candidate,
                refill_builder=refill_builder,
                mode="compose_guarded",
            )
        if args.pseudo_label_mode not in {"compose", "compose_corrupt"}:
            return [], 0, {
                "mode": args.pseudo_label_mode,
                "candidate_total": len(candidate_examples),
                "retained_total": 0,
                "missing_total": 0,
            }

        base_predictions = generate_prediction_map(
            model=model,
            tokenizer=tokenizer,
            examples=base_examples,
            batch_size=batch_size,
            max_new_tokens=decode_max_new_tokens,
            key_getter=majority_key,
            prediction_parser=self.prediction_parser,
        )
        pseudo_examples: List[MajorityExample] = []
        missing_labels = 0
        corrupted_examples = 0
        for example in candidate_examples:
            component_keys = component_map.get(majority_key(example))
            if not component_keys:
                missing_labels += 1
                continue
            components: List[Tuple[int, int]] = []
            missing = False
            for component_key in component_keys:
                prediction = base_predictions.get(component_key)
                if prediction is None:
                    missing = True
                    break
                parsed = INTEGER_PATTERN.findall(prediction)
                if len(parsed) < 2:
                    missing = True
                    break
                ones = int(parsed[0])
                bits = component_key[0]
                if ones < 0 or ones > bits:
                    missing = True
                    break
                components.append((bits, ones))
            if missing or not components:
                missing_labels += 1
                continue
            if args.pseudo_label_mode == "compose_corrupt" and rng.random() < args.corruption_rate:
                idx = rng.randrange(len(components))
                bits, ones = components[idx]
                flipped = bits - ones
                if flipped == ones:
                    flipped = min(bits, ones + 1) if ones < bits else max(0, ones - 1)
                components[idx] = (bits, flipped)
                corrupted_examples += 1
            total_ones = sum(ones for _, ones in components)
            total_bits = sum(bits for bits, _ in components)
            majority = 1 if total_ones * 2 >= total_bits else 0
            override = f"{total_ones}|{majority}"
            pseudo_examples.append(clone_majority_with_override(example, override))
        diagnostics: JsonDict = {
            "mode": args.pseudo_label_mode,
            "target_max_bits": int(target_max_size),
            "candidate_total": len(candidate_examples),
            "retained_total": len(pseudo_examples),
            "missing_total": missing_labels,
            "retained_fraction": len(pseudo_examples) / len(candidate_examples) if candidate_examples else math.nan,
            "corruption_rate": args.corruption_rate if args.pseudo_label_mode == "compose_corrupt" else 0.0,
            "corrupted_examples": corrupted_examples,
        }
        return pseudo_examples, missing_labels, diagnostics

    def build_task_metadata(self, args: Any, final_max_size: int) -> JsonDict:
        return {
            "pseudo_label_mode": args.pseudo_label_mode,
            "corruption_rate": args.corruption_rate,
            "format_version": normalize_task_format_version(args),
            "target_mode": normalize_bit_target_mode(args),
            "compose_arity": normalize_compose_arity(args),
            "guarded_compose_rule": normalize_guarded_compose_rule(args),
        }

    def metadata_aliases(self, args: Any, final_max_size: int) -> JsonDict:
        return {
            "initial_min_bits": args.initial_min_size,
            "initial_max_bits": args.initial_max_size,
            "expand_num_bits": args.expand_num_size,
            "expand_train_per_bit": args.expand_train_per_size,
            "eval_per_bit": args.eval_per_size,
            "composed_eval_per_bit": args.composed_eval_per_size,
            "composed_max_bits": final_max_size,
            "pseudo_label_mode": args.pseudo_label_mode,
            "corruption_rate": args.corruption_rate,
            "format_version": normalize_task_format_version(args),
            "target_mode": normalize_bit_target_mode(args),
            "compose_arity": normalize_compose_arity(args),
            "guarded_compose_rule": normalize_guarded_compose_rule(args),
        }

    def validate_loaded_metadata(
        self,
        args: Any,
        metadata: JsonDict,
        final_max_size: int,
        dynamic_composed: bool,
    ) -> None:
        task_config = metadata.get("task_config", {}) if isinstance(metadata.get("task_config"), dict) else {}
        stored_format = str(task_config.get("format_version", metadata.get("format_version", "legacy")))
        if stored_format != normalize_task_format_version(args):
            raise ValueError("Stored majority dataset uses a different format_version.")
        stored_target_mode = str(task_config.get("target_mode", metadata.get("target_mode", "default")))
        if stored_target_mode != normalize_bit_target_mode(args):
            raise ValueError("Stored majority dataset uses a different target_mode.")
        stored_compose_arity = str(task_config.get("compose_arity", metadata.get("compose_arity", "at_least2")))
        if stored_compose_arity != normalize_compose_arity(args):
            raise ValueError("Stored majority dataset uses a different compose_arity.")
        stored_guarded_rule = str(
            task_config.get("guarded_compose_rule", metadata.get("guarded_compose_rule", "none"))
        )
        if stored_guarded_rule != normalize_guarded_compose_rule(args):
            raise ValueError("Stored majority dataset uses a different guarded_compose_rule.")

    def summary_payload_aliases(self, summary: Any) -> JsonDict:
        return {
            "max_bits": summary.max_size,
            "per_bit_accuracy": {str(size): score for size, score in summary.per_size_accuracy.items()},
            "max_bits_at_90_accuracy": max(
                [size for size, score in summary.per_size_accuracy.items() if score >= 0.90],
                default=None,
            ),
        }


class RunLengthTask(SelfImprovementTask):
    name = "run_length"
    size_label = "bits"
    size_alias_singular = "bit"
    size_alias_plural = "bits"

    def validate_args(self, args: Any) -> None:
        corruption_rate = float(getattr(args, "corruption_rate", 0.0))
        pseudo_label_mode = str(getattr(args, "pseudo_label_mode", "none"))
        if corruption_rate < 0.0 or corruption_rate > 1.0:
            raise ValueError("corruption_rate must be between 0 and 1.")
        symbol_alphabet_size = normalize_symbol_alphabet_size(args)
        if symbol_alphabet_size < 2:
            raise ValueError("run_length symbol_alphabet_size must be at least 2.")
        if symbol_alphabet_size > len(RUN_LENGTH_ALPHABET_SYMBOLS):
            raise ValueError(
                f"run_length symbol_alphabet_size={symbol_alphabet_size} exceeds supported alphabet size "
                f"{len(RUN_LENGTH_ALPHABET_SYMBOLS)}."
            )
        format_version = normalize_task_format_version(args)
        if format_version not in RUN_LENGTH_FORMATS:
            raise ValueError(f"Unsupported run_length format_version={format_version!r}.")
        target_mode = normalize_bit_target_mode(args)
        if target_mode not in BIT_TARGET_MODES:
            raise ValueError(f"Unsupported run_length target_mode={target_mode!r}.")
        compose_arity = normalize_compose_arity(args)
        if compose_arity not in BIT_COMPOSE_ARITIES:
            raise ValueError(f"Unsupported run_length compose_arity={compose_arity!r}.")
        guarded_rule = normalize_guarded_compose_rule(args)
        if guarded_rule not in BIT_GUARDED_COMPOSE_RULES:
            raise ValueError(f"Unsupported run_length guarded_compose_rule={guarded_rule!r}.")
        if target_mode in {"plain_output", "symbol_run_pair"} and pseudo_label_mode in {"compose", "compose_corrupt"}:
            if guarded_rule != "run_length_no_boundary_continue":
                raise ValueError(
                    "run_length guarded output compose mode requires --guarded-compose-rule run_length_no_boundary_continue."
                )
            if compose_arity != "exact2":
                raise ValueError("run_length guarded output compose mode requires --compose-arity exact2.")
            if pseudo_label_mode == "compose_corrupt":
                raise ValueError("run_length guarded output diagnostics do not support compose_corrupt.")

    def serialize_example(self, example: RunLengthExample) -> JsonDict:
        return {
            "bitstring": example.bitstring,
            "bits": example.bits,
            "max_run": example.max_run,
            "prefix_run": example.prefix_run,
            "suffix_run": example.suffix_run,
            "format_version": example.format_version,
            "target_mode": example.target_mode,
            "target_override": example.target_override,
        }

    def deserialize_example(self, payload: JsonDict) -> RunLengthExample:
        return RunLengthExample(
            bitstring=str(payload["bitstring"]),
            bits=int(payload["bits"]),
            max_run=int(payload["max_run"]),
            prefix_run=int(payload["prefix_run"]),
            suffix_run=int(payload["suffix_run"]),
            format_version=str(payload.get("format_version", "legacy")),
            target_mode=str(payload.get("target_mode", "default")),
            target_override=payload.get("target_override"),
        )

    def save_component_map(self, path: Path, component_map: Dict[Tuple[int, str], List[Tuple[int, str]]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            encode_run_length_key(key): [encode_run_length_key(child) for child in children]
            for key, children in component_map.items()
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load_component_map(self, path: Path) -> Dict[Tuple[int, str], List[Tuple[int, str]]]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return {decode_run_length_key(key): [decode_run_length_key(child) for child in children] for key, children in raw.items()}

    def prepare_initial_splits(
        self,
        rng: random.Random,
        args: Any,
    ) -> Tuple[Dict[SplitName, List[RunLengthExample]], Dict[SplitName, set[Tuple[int, str]]]]:
        splits = {name: [] for name in ("train", "validation", "test")}
        records: Dict[SplitName, set[Tuple[int, str]]] = {name: set() for name in splits}
        split_order = ("validation", "test", "train") if getattr(args, "reserve_heldout_first", False) else (
            "train",
            "validation",
            "test",
        )
        generated = build_run_length_length_bucket_dataset(
            min_bits=args.initial_min_size,
            max_bits=args.initial_max_size,
            per_bit_counts={
                "train": args.initial_train_per_size,
                "validation": args.initial_eval_per_size,
                "test": args.initial_eval_per_size,
            },
            rng=rng,
            exclude_keys=getattr(args, "_initial_exclude_keys", None),
            record_keys=records,
            progress_name="initial",
            format_version=normalize_task_format_version(args),
            target_mode=normalize_bit_target_mode(args),
            alphabet=RUN_LENGTH_ALPHABET_SYMBOLS[:normalize_symbol_alphabet_size(args)],
            split_order=split_order,
        )
        for split in splits:
            splits[split] = generated.get(split, [])
        return splits, records

    def prepare_composed_train(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[RunLengthExample]],
        base_records: Dict[SplitName, set[Tuple[int, str]]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Tuple[int, str]]] = None,
    ) -> Tuple[List[RunLengthExample], Dict[Tuple[int, str], List[Tuple[int, str]]], set[Tuple[int, str]]]:
        if max_size < min_size or args.expand_train_per_size <= 0:
            return [], {}, set()
        composed_records: Dict[SplitName, set[Tuple[int, str]]] = {"train": set(), "validation": set(), "test": set()}
        component_records: Dict[SplitName, Dict[Tuple[int, str], List[Tuple[int, str]]]] = {
            "train": {},
            "validation": {},
            "test": {},
        }
        base_used = set().union(*base_records.values())
        if additional_exclude:
            base_used.update(additional_exclude)
        compose_arity = normalize_compose_arity(args)
        if compose_arity == "exact2":
            target_sizes = exact2_reachable_sizes_from_examples(
                base_splits.get("train", []),
                size_getter=lambda example: example.bits,
                min_size=min_size,
                max_size=max_size,
            )
            train_examples: List[RunLengthExample] = []
            for bits in target_sizes:
                composed_splits = build_run_length_composed_dataset(
                    base_splits=base_splits,
                    min_bits=bits,
                    max_bits=bits,
                    per_bit_counts={"train": args.expand_train_per_size, "validation": 0, "test": 0},
                    rng=rng,
                    exclude_keys=base_used,
                    record_keys=composed_records,
                    progress_name="composed",
                    record_components=component_records,
                    compose_arity=compose_arity,
                )
                train_examples.extend(composed_splits.get("train", []))
            return train_examples, component_records.get("train", {}), composed_records.get("train", set())
        composed_splits = build_run_length_composed_dataset(
            base_splits=base_splits,
            min_bits=min_size,
            max_bits=max_size,
            per_bit_counts={"train": args.expand_train_per_size, "validation": 0, "test": 0},
            rng=rng,
            exclude_keys=base_used,
            record_keys=composed_records,
            progress_name="composed",
            record_components=component_records,
            compose_arity=compose_arity,
        )
        return composed_splits.get("train", []), component_records.get("train", {}), composed_records.get("train", set())

    def prepare_composed_eval(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[RunLengthExample]],
        base_records: Dict[SplitName, set[Tuple[int, str]]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Tuple[int, str]]] = None,
    ) -> Tuple[List[RunLengthExample], Dict[Tuple[int, str], List[Tuple[int, str]]], set[Tuple[int, str]]]:
        if max_size < min_size or args.composed_eval_per_size <= 0:
            return [], {}, set()
        composed_records: Dict[SplitName, set[Tuple[int, str]]] = {"train": set(), "validation": set(), "test": set()}
        component_records: Dict[SplitName, Dict[Tuple[int, str], List[Tuple[int, str]]]] = {
            "train": {},
            "validation": {},
            "test": {},
        }
        base_used = set().union(*base_records.values())
        if additional_exclude:
            base_used.update(additional_exclude)
        stitched_base_splits = {
            "train": list(base_splits.get("train", [])),
            "validation": list(base_splits.get("train", [])),
            "test": list(base_splits.get("train", [])),
        }
        compose_arity = normalize_compose_arity(args)
        if compose_arity == "exact2":
            target_sizes = exact2_reachable_sizes_from_examples(
                stitched_base_splits.get("train", []),
                size_getter=lambda example: example.bits,
                min_size=min_size,
                max_size=max_size,
            )
            test_examples: List[RunLengthExample] = []
            for bits in target_sizes:
                composed_splits = build_run_length_composed_dataset(
                    base_splits=stitched_base_splits,
                    min_bits=bits,
                    max_bits=bits,
                    per_bit_counts={"train": 0, "validation": 0, "test": args.composed_eval_per_size},
                    rng=rng,
                    exclude_keys=base_used,
                    record_keys=composed_records,
                    progress_name="composed-eval",
                    record_components=component_records,
                    compose_arity=compose_arity,
                )
                test_examples.extend(composed_splits.get("test", []))
            return test_examples, component_records.get("test", {}), composed_records.get("test", set())
        composed_splits = build_run_length_composed_dataset(
            base_splits=stitched_base_splits,
            min_bits=min_size,
            max_bits=max_size,
            per_bit_counts={"train": 0, "validation": 0, "test": args.composed_eval_per_size},
            rng=rng,
            exclude_keys=base_used,
            record_keys=composed_records,
            progress_name="composed-eval",
            record_components=component_records,
            compose_arity=compose_arity,
        )
        return composed_splits.get("test", []), component_records.get("test", {}), composed_records.get("test", set())

    def prepare_eval_examples(
        self,
        rng: random.Random,
        args: Any,
        min_size: int,
        max_size: int,
        exclude: set[Tuple[int, str]],
    ) -> List[RunLengthExample]:
        generated = build_run_length_length_bucket_dataset(
            min_bits=min_size,
            max_bits=max_size,
            per_bit_counts={"train": 0, "validation": 0, "test": args.eval_per_size},
            rng=rng,
            exclude_keys=exclude,
            record_keys={split: set() for split in ("train", "validation", "test")},
            progress_name="evaluation",
            format_version=normalize_task_format_version(args),
            target_mode=normalize_bit_target_mode(args),
            alphabet=RUN_LENGTH_ALPHABET_SYMBOLS[:normalize_symbol_alphabet_size(args)],
        )
        return list(generated.get("test", []))

    def split_composed_eval_slices(
        self,
        examples: Sequence[RunLengthExample],
        component_map: Dict[Tuple[int, str], List[Tuple[int, str]]],
    ) -> Dict[str, List[RunLengthExample]]:
        if examples and examples[0].target_mode in {"plain_output", "symbol_run_pair"}:
            return guard_slice_partition(
                examples,
                component_map,
                key_getter=run_length_key,
                guard_fn=run_length_guard_accepts_true_components,
            )
        return {"all": list(examples)}

    def keys_for_examples(self, examples: Sequence[RunLengthExample]) -> set[Tuple[int, str]]:
        return {run_length_key(example) for example in examples}

    def rebuild_records(self, splits: Dict[SplitName, List[RunLengthExample]]) -> Dict[SplitName, set[Tuple[int, str]]]:
        return {split: {run_length_key(example) for example in splits.get(split, [])} for split in ("train", "validation", "test")}

    def key_for_example(self, example: RunLengthExample) -> Tuple[int, str]:
        return run_length_key(example)

    def clone_with_override(self, example: RunLengthExample, override: Optional[str]) -> RunLengthExample:
        return clone_run_length_with_override(example, override)

    def size_of(self, example: RunLengthExample) -> int:
        return example.bits

    def prediction_parser(self, text: str, example: Optional[RunLengthExample] = None) -> Optional[str]:
        return parse_run_length_prediction(text, example)

    def token_initializers(self, args: Any) -> Dict[str, str]:
        return {}

    def derive_round_targets(
        self,
        model: Any,
        tokenizer: Any,
        composed_examples: Sequence[RunLengthExample],
        component_map: Dict[Tuple[int, str], List[Tuple[int, str]]],
        target_max_size: int,
        base_examples: Sequence[RunLengthExample],
        *,
        batch_size: int,
        decode_max_new_tokens: int,
        args: Any,
        rng: random.Random,
    ) -> Tuple[List[RunLengthExample], int, JsonDict]:
        candidate_examples = [example for example in composed_examples if example.bits <= target_max_size]
        target_mode = normalize_bit_target_mode(args)
        if args.pseudo_label_mode == "direct":
            return build_direct_pseudo_examples(
                candidate_examples,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                decode_max_new_tokens=decode_max_new_tokens,
                key_getter=self.key_for_example,
                prediction_parser=self.prediction_parser,
                clone_builder=self.clone_with_override,
                mode="direct",
            )
        if target_mode in {"plain_output", "symbol_run_pair"}:
            base_predictions = generate_prediction_map(
                model=model,
                tokenizer=tokenizer,
                examples=base_examples,
                batch_size=batch_size,
                max_new_tokens=decode_max_new_tokens,
                key_getter=run_length_key,
                prediction_parser=self.prediction_parser,
            )

            def evaluate_candidate(
                example: RunLengthExample,
                component_keys: Optional[Sequence[Tuple[int, str]]],
            ) -> Tuple[str, Optional[str]]:
                if not component_keys or len(component_keys) != 2:
                    return "missing", None
                if run_length_guard_accepts_true_components(component_keys) is not True:
                    return "rejected", None
                plain_values: List[int] = []
                pair_values: List[Tuple[str, int]] = []
                for component_key in component_keys:
                    prediction = base_predictions.get(component_key)
                    if prediction is None:
                        return "missing", None
                    if target_mode == "plain_output":
                        try:
                            value = int(prediction)
                        except ValueError:
                            return "missing", None
                        if value < 0 or value > component_key[0]:
                            return "missing", None
                        plain_values.append(value)
                    else:
                        parsed_pair = parse_run_length_symbol_pair_prediction(
                            prediction,
                            RunLengthExample(
                                bitstring=component_key[1],
                                bits=component_key[0],
                                max_run=0,
                                prefix_run=0,
                                suffix_run=0,
                                target_mode="symbol_run_pair",
                            ),
                        )
                        if parsed_pair is None:
                            return "missing", None
                        symbol, value_text = parsed_pair.split("|", 1)
                        value = int(value_text)
                        if value < 0 or value > component_key[0] or symbol not in component_key[1]:
                            return "missing", None
                        pair_values.append((symbol, value))
                if target_mode == "plain_output":
                    return "accepted", str(max(plain_values))
                best_symbol, best_value = pair_values[0]
                for symbol, value in pair_values[1:]:
                    if value > best_value:
                        best_symbol = symbol
                        best_value = value
                return "accepted", f"{best_symbol}|{best_value}"

            def refill_builder(
                bits: int,
                need: int,
                occupied_keys: set[Tuple[int, str]],
            ) -> Tuple[List[RunLengthExample], Dict[Tuple[int, str], List[Tuple[int, str]]]]:
                record_components = {"train": {}, "validation": {}, "test": {}}
                refill_splits = build_run_length_composed_dataset(
                    base_splits={"train": list(base_examples), "validation": [], "test": []},
                    min_bits=bits,
                    max_bits=bits,
                    per_bit_counts={"train": need, "validation": 0, "test": 0},
                    rng=rng,
                    exclude_keys=occupied_keys,
                    record_keys={"train": set(), "validation": set(), "test": set()},
                    progress_name="guarded-refill",
                    record_components=record_components,
                    compose_arity=normalize_compose_arity(args),
                )
                return refill_splits.get("train", []), record_components.get("train", {})

            return build_guarded_bit_pseudo_examples(
                candidate_examples,
                component_map,
                target_max_size=target_max_size,
                requested_per_size=args.expand_train_per_size,
                size_getter=self.size_of,
                key_getter=self.key_for_example,
                clone_builder=self.clone_with_override,
                evaluate_candidate=evaluate_candidate,
                refill_builder=refill_builder,
                mode="compose_guarded",
            )
        if args.pseudo_label_mode not in {"compose", "compose_corrupt"}:
            return [], 0, {
                "mode": args.pseudo_label_mode,
                "candidate_total": len(candidate_examples),
                "retained_total": 0,
                "missing_total": 0,
            }

        base_predictions = generate_prediction_map(
            model=model,
            tokenizer=tokenizer,
            examples=base_examples,
            batch_size=batch_size,
            max_new_tokens=decode_max_new_tokens,
            key_getter=run_length_key,
            prediction_parser=self.prediction_parser,
        )
        pseudo_examples: List[RunLengthExample] = []
        missing_labels = 0
        corrupted_examples = 0

        def merge_stats(
            left: Tuple[int, int, int, int],
            right: Tuple[int, int, int, int],
        ) -> Tuple[int, int, int, int]:
            left_bits, left_max, left_prefix, left_suffix = left
            right_bits, right_max, right_prefix, right_suffix = right
            bits = left_bits + right_bits
            prefix = left_bits + right_prefix if left_prefix == left_bits else left_prefix
            suffix = right_bits + left_suffix if right_suffix == right_bits else right_suffix
            max_run = max(left_max, right_max, left_suffix + right_prefix)
            return bits, max_run, prefix, suffix

        for example in candidate_examples:
            component_keys = component_map.get(run_length_key(example))
            if not component_keys:
                missing_labels += 1
                continue
            components: List[Tuple[int, int, int, int]] = []
            missing = False
            for component_key in component_keys:
                prediction = base_predictions.get(component_key)
                if prediction is None:
                    missing = True
                    break
                parsed = INTEGER_PATTERN.findall(prediction)
                if len(parsed) < 3:
                    missing = True
                    break
                max_run = int(parsed[0])
                prefix = int(parsed[1])
                suffix = int(parsed[2])
                bits = component_key[0]
                if (
                    max_run < 0
                    or prefix < 0
                    or suffix < 0
                    or max_run > bits
                    or prefix > bits
                    or suffix > bits
                ):
                    missing = True
                    break
                components.append((bits, max_run, prefix, suffix))
            if missing or not components:
                missing_labels += 1
                continue
            if args.pseudo_label_mode == "compose_corrupt" and rng.random() < args.corruption_rate:
                idx = rng.randrange(len(components))
                bits, max_run, prefix, suffix = components[idx]
                if max_run < bits:
                    max_run += 1
                elif max_run > 0:
                    max_run -= 1
                elif bits > 0:
                    max_run = 1
                components[idx] = (bits, max_run, prefix, suffix)
                corrupted_examples += 1
            merged = components[0]
            for nxt in components[1:]:
                merged = merge_stats(merged, nxt)
            _, max_run, prefix, suffix = merged
            override = f"{max_run}|{prefix}|{suffix}"
            pseudo_examples.append(clone_run_length_with_override(example, override))
        diagnostics: JsonDict = {
            "mode": args.pseudo_label_mode,
            "target_max_bits": int(target_max_size),
            "candidate_total": len(candidate_examples),
            "retained_total": len(pseudo_examples),
            "missing_total": missing_labels,
            "retained_fraction": len(pseudo_examples) / len(candidate_examples) if candidate_examples else math.nan,
            "corruption_rate": args.corruption_rate if args.pseudo_label_mode == "compose_corrupt" else 0.0,
            "corrupted_examples": corrupted_examples,
        }
        return pseudo_examples, missing_labels, diagnostics

    def build_task_metadata(self, args: Any, final_max_size: int) -> JsonDict:
        return {
            "pseudo_label_mode": args.pseudo_label_mode,
            "corruption_rate": args.corruption_rate,
            "format_version": normalize_task_format_version(args),
            "target_mode": normalize_bit_target_mode(args),
            "compose_arity": normalize_compose_arity(args),
            "guarded_compose_rule": normalize_guarded_compose_rule(args),
            "symbol_alphabet_size": normalize_symbol_alphabet_size(args),
        }

    def metadata_aliases(self, args: Any, final_max_size: int) -> JsonDict:
        return {
            "initial_min_bits": args.initial_min_size,
            "initial_max_bits": args.initial_max_size,
            "expand_num_bits": args.expand_num_size,
            "expand_train_per_bit": args.expand_train_per_size,
            "eval_per_bit": args.eval_per_size,
            "composed_eval_per_bit": args.composed_eval_per_size,
            "composed_max_bits": final_max_size,
            "pseudo_label_mode": args.pseudo_label_mode,
            "corruption_rate": args.corruption_rate,
            "format_version": normalize_task_format_version(args),
            "target_mode": normalize_bit_target_mode(args),
            "compose_arity": normalize_compose_arity(args),
            "guarded_compose_rule": normalize_guarded_compose_rule(args),
            "symbol_alphabet_size": normalize_symbol_alphabet_size(args),
        }

    def validate_loaded_metadata(
        self,
        args: Any,
        metadata: JsonDict,
        final_max_size: int,
        dynamic_composed: bool,
    ) -> None:
        task_config = metadata.get("task_config", {}) if isinstance(metadata.get("task_config"), dict) else {}
        stored_format = str(task_config.get("format_version", metadata.get("format_version", "legacy")))
        if stored_format != normalize_task_format_version(args):
            raise ValueError("Stored run-length dataset uses a different format_version.")
        stored_target_mode = str(task_config.get("target_mode", metadata.get("target_mode", "default")))
        if stored_target_mode != normalize_bit_target_mode(args):
            raise ValueError("Stored run-length dataset uses a different target_mode.")
        stored_compose_arity = str(task_config.get("compose_arity", metadata.get("compose_arity", "at_least2")))
        if stored_compose_arity != normalize_compose_arity(args):
            raise ValueError("Stored run-length dataset uses a different compose_arity.")
        stored_guarded_rule = str(
            task_config.get("guarded_compose_rule", metadata.get("guarded_compose_rule", "none"))
        )
        if stored_guarded_rule != normalize_guarded_compose_rule(args):
            raise ValueError("Stored run-length dataset uses a different guarded_compose_rule.")
        stored_symbol_alphabet_size = int(
            task_config.get("symbol_alphabet_size", metadata.get("symbol_alphabet_size", 2))
        )
        if stored_symbol_alphabet_size != normalize_symbol_alphabet_size(args):
            raise ValueError("Stored run-length dataset uses a different symbol_alphabet_size.")

    def summary_payload_aliases(self, summary: Any) -> JsonDict:
        return {
            "max_bits": summary.max_size,
            "per_bit_accuracy": {str(size): score for size, score in summary.per_size_accuracy.items()},
            "max_bits_at_90_accuracy": max(
                [size for size, score in summary.per_size_accuracy.items() if score >= 0.90],
                default=None,
            ),
        }


class MultiplicationTask(SelfImprovementTask):
    name = "multiplication"
    size_label = "digits"
    size_alias_singular = "digit"
    size_alias_plural = "digits"

    def validate_args(self, args: Any) -> None:
        if args.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if args.initial_min_size < args.block_size:
            raise ValueError("initial_min_size must be >= block_size for multiplication.")
        if args.initial_max_size < args.initial_min_size:
            raise ValueError("initial_max_size must be >= initial_min_size for multiplication.")
        if args.expand_num_size % args.block_size != 0:
            raise ValueError("expand_num_size must be a multiple of block_size for blocked multiplication.")
        if args.corruption_rate < 0.0 or args.corruption_rate > 1.0:
            raise ValueError("corruption_rate must be between 0 and 1.")
        if not getattr(args, "oracle_aggregation", True):
            raise ValueError("Multiplication is workshop-scoped to oracle aggregation only.")
        format_version = normalize_task_format_version(args)
        if format_version not in MULTIPLICATION_FORMATS:
            raise ValueError(f"Unsupported multiplication format_version={format_version!r}.")

    def serialize_example(self, example: MultiplicationExample) -> JsonDict:
        return {
            "a": example.a,
            "b": example.b,
            "digits": example.digits,
            "result": example.result,
            "operand_width": example.operand_width,
            "format_version": example.format_version,
            "target_override": example.target_override,
        }

    def deserialize_example(self, payload: JsonDict) -> MultiplicationExample:
        return MultiplicationExample(
            a=int(payload["a"]),
            b=int(payload["b"]),
            digits=int(payload["digits"]),
            result=int(payload["result"]),
            operand_width=int(payload["operand_width"]),
            format_version=str(payload.get("format_version", "legacy")),
            target_override=payload.get("target_override"),
        )

    def save_component_map(self, path: Path, component_map: Dict[Tuple[int, int, int], Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {encode_multiplication_key(key): value for key, value in component_map.items()}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load_component_map(self, path: Path) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return {decode_multiplication_key(key): dict(value) for key, value in raw.items()}

    def prepare_initial_splits(
        self,
        rng: random.Random,
        args: Any,
    ) -> Tuple[Dict[SplitName, List[MultiplicationExample]], Dict[SplitName, set[Tuple[int, int, int]]]]:
        records: Dict[SplitName, set[Tuple[int, int, int]]] = {name: set() for name in ("train", "validation", "test")}
        if args.initial_min_size == args.block_size and args.initial_max_size == args.block_size:
            splits = build_multiplication_seed_dataset(
                block_size=args.block_size,
                per_split_counts={
                    "train": args.initial_train_per_size,
                    "validation": args.initial_eval_per_size,
                    "test": args.initial_eval_per_size,
                },
                rng=rng,
                record_keys=records,
                progress_name="initial",
                format_version=normalize_task_format_version(args),
            )
        else:
            splits = build_multiplication_long_dataset(
                min_digits=args.initial_min_size,
                max_digits=args.initial_max_size,
                per_digit_counts={
                    "train": args.initial_train_per_size,
                    "validation": args.initial_eval_per_size,
                    "test": args.initial_eval_per_size,
                },
                rng=rng,
                block_size=args.block_size,
                record_keys=records,
                progress_name="initial",
                format_version=normalize_task_format_version(args),
            )
        return splits, records

    def prepare_composed_train(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[MultiplicationExample]],
        base_records: Dict[SplitName, set[Tuple[int, int, int]]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Tuple[int, int, int]]] = None,
    ) -> Tuple[List[MultiplicationExample], Dict[Tuple[int, int, int], Dict[str, Any]], set[Tuple[int, int, int]]]:
        if max_size < min_size or args.expand_train_per_size <= 0:
            return [], {}, set()
        effective_min_size = max(args.block_size * 2, ((min_size + args.block_size - 1) // args.block_size) * args.block_size)
        if max_size < effective_min_size:
            return [], {}, set()
        composed_records: Dict[SplitName, set[Tuple[int, int, int]]] = {"train": set(), "validation": set(), "test": set()}
        component_records: Dict[SplitName, Dict[Tuple[int, int, int], Dict[str, Any]]] = {
            "train": {},
            "validation": {},
            "test": {},
        }
        exclude = set().union(*base_records.values())
        if additional_exclude:
            exclude.update(additional_exclude)
        composed = build_multiplication_long_dataset(
            min_digits=effective_min_size,
            max_digits=max_size,
            per_digit_counts={"train": args.expand_train_per_size, "validation": 0, "test": 0},
            rng=rng,
            block_size=args.block_size,
            exclude_keys=exclude,
            record_keys=composed_records,
            progress_name="composed",
            record_components=component_records,
            format_version=normalize_task_format_version(args),
        )
        return composed.get("train", []), component_records.get("train", {}), composed_records.get("train", set())

    def prepare_composed_eval(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[MultiplicationExample]],
        base_records: Dict[SplitName, set[Tuple[int, int, int]]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Tuple[int, int, int]]] = None,
    ) -> Tuple[List[MultiplicationExample], Dict[Tuple[int, int, int], Dict[str, Any]], set[Tuple[int, int, int]]]:
        if max_size < min_size or args.composed_eval_per_size <= 0:
            return [], {}, set()
        effective_min_size = max(args.block_size * 2, ((min_size + args.block_size - 1) // args.block_size) * args.block_size)
        if max_size < effective_min_size:
            return [], {}, set()
        composed_records: Dict[SplitName, set[Tuple[int, int, int]]] = {"train": set(), "validation": set(), "test": set()}
        component_records: Dict[SplitName, Dict[Tuple[int, int, int], Dict[str, Any]]] = {
            "train": {},
            "validation": {},
            "test": {},
        }
        exclude = set().union(*base_records.values())
        if additional_exclude:
            exclude.update(additional_exclude)
        composed = build_multiplication_long_dataset(
            min_digits=effective_min_size,
            max_digits=max_size,
            per_digit_counts={"train": 0, "validation": 0, "test": args.composed_eval_per_size},
            rng=rng,
            block_size=args.block_size,
            exclude_keys=exclude,
            record_keys=composed_records,
            progress_name="composed-eval",
            record_components=component_records,
            format_version=normalize_task_format_version(args),
        )
        return composed.get("test", []), component_records.get("test", {}), composed_records.get("test", set())

    def prepare_eval_examples(
        self,
        rng: random.Random,
        args: Any,
        min_size: int,
        max_size: int,
        exclude: set[Tuple[int, int, int]],
    ) -> List[MultiplicationExample]:
        generated = build_multiplication_long_dataset(
            min_digits=min_size,
            max_digits=max_size,
            per_digit_counts={"train": 0, "validation": 0, "test": args.eval_per_size},
            rng=rng,
            block_size=args.block_size,
            exclude_keys=exclude,
            progress_name="evaluation",
            format_version=normalize_task_format_version(args),
        )
        return list(generated.get("test", []))

    def split_composed_eval_slices(
        self,
        examples: Sequence[MultiplicationExample],
        component_map: Dict[Tuple[int, int, int], Dict[str, Any]],
    ) -> Dict[str, List[MultiplicationExample]]:
        slices = {
            "low_overlap_low_carry": [],
            "low_overlap_high_carry": [],
            "high_overlap_low_carry": [],
            "high_overlap_high_carry": [],
            "unknown": [],
        }
        for example in examples:
            payload = component_map.get(multiplication_key(example))
            if payload is None:
                slices["unknown"].append(example)
                continue
            slice_name = get_multiplication_slice_name(payload, int(payload.get("block_size", 2)))
            slices[slice_name].append(example)
        return slices

    def keys_for_examples(self, examples: Sequence[MultiplicationExample]) -> set[Tuple[int, int, int]]:
        return {multiplication_key(example) for example in examples}

    def rebuild_records(
        self,
        splits: Dict[SplitName, List[MultiplicationExample]],
    ) -> Dict[SplitName, set[Tuple[int, int, int]]]:
        return {split: {multiplication_key(example) for example in splits.get(split, [])} for split in ("train", "validation", "test")}

    def key_for_example(self, example: MultiplicationExample) -> Tuple[int, int, int]:
        return multiplication_key(example)

    def clone_with_override(self, example: MultiplicationExample, override: Optional[str]) -> MultiplicationExample:
        return clone_multiplication_with_override(example, override)

    def size_of(self, example: MultiplicationExample) -> int:
        return example.digits

    def prediction_parser(self, text: str, example: Optional[MultiplicationExample] = None) -> Optional[str]:
        return parse_multiplication_prediction(text, example)

    def token_initializers(self, args: Any) -> Dict[str, str]:
        if normalize_task_format_version(args) == "symbolic_v1":
            return {"×": "*"}
        return {}

    def derive_round_targets(
        self,
        model: Any,
        tokenizer: Any,
        composed_examples: Sequence[MultiplicationExample],
        component_map: Dict[Tuple[int, int, int], Dict[str, Any]],
        target_max_size: int,
        base_examples: Sequence[MultiplicationExample],
        *,
        batch_size: int,
        decode_max_new_tokens: int,
        args: Any,
        rng: random.Random,
    ) -> Tuple[List[MultiplicationExample], int, JsonDict]:
        candidate_examples = [example for example in composed_examples if example.digits <= target_max_size]
        if args.pseudo_label_mode == "direct":
            return build_direct_pseudo_examples(
                candidate_examples,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                decode_max_new_tokens=decode_max_new_tokens,
                key_getter=self.key_for_example,
                prediction_parser=self.prediction_parser,
                clone_builder=self.clone_with_override,
                mode="direct",
            )
        if args.pseudo_label_mode not in {"compose", "compose_corrupt"}:
            return [], 0, {
                "mode": args.pseudo_label_mode,
                "candidate_total": len(candidate_examples),
                "retained_total": 0,
                "missing_total": 0,
            }

        component_examples: Dict[Tuple[int, int, int], MultiplicationExample] = {}
        for example in candidate_examples:
            payload = component_map.get(multiplication_key(example))
            if payload is None:
                continue
            for partial in payload.get("partials", []):
                component = MultiplicationExample(
                    a=int(partial["a"]),
                    b=int(partial["b"]),
                    digits=args.block_size,
                    result=int(partial["a"]) * int(partial["b"]),
                    operand_width=args.block_size,
                    format_version=normalize_task_format_version(args),
                )
                component_examples[multiplication_key(component)] = component

        component_predictions = generate_prediction_map(
            model=model,
            tokenizer=tokenizer,
            examples=list(component_examples.values()),
            batch_size=batch_size,
            max_new_tokens=decode_max_new_tokens,
            key_getter=multiplication_key,
            prediction_parser=self.prediction_parser,
        )

        pseudo_examples: List[MultiplicationExample] = []
        missing_total = 0
        corrupted_component_total = 0
        corrupted_example_total = 0

        for example in candidate_examples:
            payload = component_map.get(multiplication_key(example))
            if payload is None:
                missing_total += 1
                continue
            partial_predictions: List[Tuple[int, int]] = []
            example_corrupted = False
            missing = False
            for partial in payload.get("partials", []):
                component = MultiplicationExample(
                    a=int(partial["a"]),
                    b=int(partial["b"]),
                    digits=args.block_size,
                    result=int(partial["a"]) * int(partial["b"]),
                    operand_width=args.block_size,
                    format_version=normalize_task_format_version(args),
                )
                prediction = component_predictions.get(multiplication_key(component))
                if prediction is None:
                    missing = True
                    break
                numeric_prediction = int(prediction)
                if args.pseudo_label_mode == "compose_corrupt" and rng.random() < args.corruption_rate:
                    numeric_prediction += 1
                    corrupted_component_total += 1
                    example_corrupted = True
                partial_predictions.append((numeric_prediction, int(partial["shift"])))
            if missing:
                missing_total += 1
                continue
            if example_corrupted:
                corrupted_example_total += 1
            composed_value = sum(value * (10**shift) for value, shift in partial_predictions)
            pseudo_examples.append(
                clone_multiplication_with_override(
                    example,
                    format_multiplication_target(composed_value, example.digits, example.format_version),
                )
            )

        diagnostics: JsonDict = {
            "mode": args.pseudo_label_mode,
            "target_max_digits": int(target_max_size),
            "candidate_total": len(candidate_examples),
            "retained_total": len(pseudo_examples),
            "missing_total": missing_total,
            "retained_fraction": len(pseudo_examples) / len(candidate_examples) if candidate_examples else math.nan,
            "oracle_aggregation": bool(args.oracle_aggregation),
            "corruption_rate": args.corruption_rate if args.pseudo_label_mode == "compose_corrupt" else 0.0,
            "corrupted_component_total": corrupted_component_total,
            "corrupted_example_total": corrupted_example_total,
        }
        return pseudo_examples, missing_total, diagnostics

    def build_task_metadata(self, args: Any, final_max_size: int) -> JsonDict:
        return {
            "block_size": args.block_size,
            "oracle_aggregation": bool(args.oracle_aggregation),
            "pseudo_label_mode": args.pseudo_label_mode,
            "corruption_rate": args.corruption_rate,
            "format_version": normalize_task_format_version(args),
        }

    def metadata_aliases(self, args: Any, final_max_size: int) -> JsonDict:
        return {
            "block_size": args.block_size,
            "oracle_aggregation": bool(args.oracle_aggregation),
            "pseudo_label_mode": args.pseudo_label_mode,
            "corruption_rate": args.corruption_rate,
            "composed_max_digits": final_max_size,
            "format_version": normalize_task_format_version(args),
        }

    def validate_loaded_metadata(
        self,
        args: Any,
        metadata: JsonDict,
        final_max_size: int,
        dynamic_composed: bool,
    ) -> None:
        task_config = metadata.get("task_config", {}) if isinstance(metadata.get("task_config"), dict) else {}
        stored_block_size = int(task_config.get("block_size", metadata.get("block_size", args.block_size)))
        if stored_block_size != args.block_size:
            raise ValueError("Stored multiplication dataset uses a different block_size.")
        stored_oracle = bool(task_config.get("oracle_aggregation", metadata.get("oracle_aggregation", True)))
        if not stored_oracle:
            raise ValueError("Stored multiplication dataset is not oracle-aggregation based and is no longer supported.")
        stored_format = str(task_config.get("format_version", metadata.get("format_version", "legacy")))
        if stored_format != normalize_task_format_version(args):
            raise ValueError("Stored multiplication dataset uses a different format_version.")

    def summary_payload_aliases(self, summary: Any) -> JsonDict:
        return {
            "max_digits": summary.max_size,
            "per_digit_accuracy": {str(size): score for size, score in summary.per_size_accuracy.items()},
            "max_digits_at_90_accuracy": max(
                [size for size, score in summary.per_size_accuracy.items() if score >= 0.90],
                default=None,
            ),
        }
