from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.addition_pipeline import (
    ADDITION_SAMPLING_BALANCED_VISIBLE_LENGTHS,
    ADDITION_WIDTH_FIXED_MIXED_PROMPT,
    COMPOSITION_PATH_FIXED_BINARY,
    AdditionExample,
    build_composed_datasets,
    build_length_bucket_dataset,
    build_composed_pseudo_map,
    compose_examples,
    example_key,
    generate_addition_pair,
)
from self.self_improvement_core import evaluate_accuracy_with_breakdown, extract_numeric_answer, generate_prediction_map
from self import self_improvement_tasks as tasks


def test_majority_parse_and_compose():
    left = tasks.MajorityExample(bitstring="010", bits=3, ones=1, majority=0)
    right = tasks.MajorityExample(bitstring="11", bits=2, ones=2, majority=1)
    composed = tasks.compose_majority_examples(left, right)

    assert composed.bitstring == "01011"
    assert composed.ones == 3
    assert composed.majority == 1
    assert tasks.parse_majority_prediction("Answer: 3 1") == "3|1"


def test_symbolic_majority_prompt_and_target_prefix():
    example = tasks.MajorityExample(bitstring="01011010", bits=8, ones=4, majority=1, format_version="symbolic_v1")
    assert example.prompt() == "majority(01011010)="
    assert example.target() == "4|1"
    assert example.target_prefix() == ""


def test_plain_output_majority_target_and_parser():
    example = tasks.MajorityExample(bitstring="01011010", bits=8, ones=4, majority=1, target_mode="plain_output")
    assert example.target() == "1"
    assert tasks.parse_majority_prediction("prediction: 1", example) == "1"
    assert tasks.parse_majority_prediction("prediction: 3", example) is None


def test_majority_seed_split_can_reserve_heldout_first():
    generated = tasks.build_majority_length_bucket_dataset(
        min_bits=4,
        max_bits=4,
        per_bit_counts={"train": 100, "validation": 4, "test": 4},
        rng=random.Random(0),
        split_order=("validation", "test", "train"),
    )

    assert len(generated["validation"]) == 4
    assert len(generated["test"]) == 4
    assert len(generated["train"]) == 8


def test_majority_exact2_composed_dataset_records_two_children():
    base_examples = [
        tasks.MajorityExample(bitstring="0000", bits=4, ones=0, majority=0),
        tasks.MajorityExample(bitstring="00000", bits=5, ones=0, majority=0),
        tasks.MajorityExample(bitstring="111111", bits=6, ones=6, majority=1),
    ]
    component_records = {"train": {}, "validation": {}, "test": {}}

    generated = tasks.build_majority_composed_dataset(
        base_splits={"train": list(base_examples), "validation": [], "test": []},
        min_bits=9,
        max_bits=9,
        per_bit_counts={"train": 3, "validation": 0, "test": 0},
        rng=random.Random(0),
        record_components=component_records,
        compose_arity="exact2",
    )

    assert len(generated["train"]) == 3
    for example in generated["train"]:
        component_keys = component_records["train"][tasks.majority_key(example)]
        assert len(component_keys) == 2
        assert sum(key[0] for key in component_keys) == 9


def test_majority_task_compose_corrupt_flips_one_component(monkeypatch):
    class FixedRng:
        def random(self) -> float:
            return 0.0

        def randrange(self, _: int) -> int:
            return 0

    task = tasks.MajorityTask()
    base_examples = [
        tasks.MajorityExample(bitstring="00", bits=2, ones=0, majority=0),
        tasks.MajorityExample(bitstring="01", bits=2, ones=1, majority=1),
    ]
    composed = tasks.compose_majority_examples(*base_examples)
    component_map = {tasks.majority_key(composed): [tasks.majority_key(example) for example in base_examples]}

    def fake_prediction_map(**kwargs):
        return {tasks.majority_key(example): example.target() for example in kwargs["examples"]}

    monkeypatch.setattr(tasks, "generate_prediction_map", fake_prediction_map)

    pseudo_examples, missing_total, diagnostics = task.derive_round_targets(
        model=None,
        tokenizer=None,
        composed_examples=[composed],
        component_map=component_map,
        target_max_size=4,
        base_examples=base_examples,
        batch_size=1,
        decode_max_new_tokens=8,
        args=SimpleNamespace(pseudo_label_mode="compose_corrupt", corruption_rate=1.0),
        rng=FixedRng(),
    )

    assert missing_total == 0
    assert diagnostics["corrupted_examples"] == 1
    assert pseudo_examples[0].target() == "3|1"


def test_majority_plain_output_guarded_compose_refills_to_exact_count(monkeypatch):
    task = tasks.MajorityTask()
    left_zero = tasks.MajorityExample(bitstring="0000", bits=4, ones=0, majority=0, target_mode="plain_output")
    right_zero = tasks.MajorityExample(bitstring="00000", bits=5, ones=0, majority=0, target_mode="plain_output")
    right_one = tasks.MajorityExample(bitstring="11111", bits=5, ones=5, majority=1, target_mode="plain_output")
    accepted = tasks.compose_majority_examples(left_zero, right_zero)
    rejected = tasks.compose_majority_examples(left_zero, right_one)

    refill_right = tasks.MajorityExample(bitstring="00111", bits=5, ones=3, majority=1, target_mode="plain_output")
    refill_left = tasks.MajorityExample(bitstring="1111", bits=4, ones=4, majority=1, target_mode="plain_output")
    refill_example = tasks.compose_majority_examples(refill_left, refill_right)

    def fake_prediction_map(**kwargs):
        return {
            tasks.majority_key(left_zero): "0",
            tasks.majority_key(right_zero): "0",
            tasks.majority_key(right_one): "1",
            tasks.majority_key(refill_left): "1",
            tasks.majority_key(refill_right): "1",
        }

    def fake_build_majority_composed_dataset(**kwargs):
        record_components = kwargs["record_components"]
        record_components["train"][tasks.majority_key(refill_example)] = [
            tasks.majority_key(refill_left),
            tasks.majority_key(refill_right),
        ]
        return {"train": [refill_example], "validation": [], "test": []}

    monkeypatch.setattr(tasks, "generate_prediction_map", fake_prediction_map)
    monkeypatch.setattr(tasks, "build_majority_composed_dataset", fake_build_majority_composed_dataset)

    pseudo_examples, missing_total, diagnostics = task.derive_round_targets(
        model=None,
        tokenizer=None,
        composed_examples=[accepted, rejected],
        component_map={
            tasks.majority_key(accepted): [tasks.majority_key(left_zero), tasks.majority_key(right_zero)],
            tasks.majority_key(rejected): [tasks.majority_key(left_zero), tasks.majority_key(right_one)],
        },
        target_max_size=9,
        base_examples=[left_zero, right_zero, right_one, refill_left, refill_right],
        batch_size=1,
        decode_max_new_tokens=8,
        args=SimpleNamespace(
            pseudo_label_mode="compose",
            corruption_rate=0.0,
            target_mode="plain_output",
            compose_arity="exact2",
            guarded_compose_rule="majority_agree_pair",
            expand_train_per_size=2,
        ),
        rng=random.Random(0),
    )

    assert missing_total == 0
    assert [example.target() for example in pseudo_examples] == ["0", "1"]
    assert diagnostics["requested_total"] == 2
    assert diagnostics["retained_total"] == 2
    assert diagnostics["rejected_total"] == 1
    assert diagnostics["refill_rounds"] == 1


def test_majority_plain_output_composed_eval_slices_partition_guard():
    task = tasks.MajorityTask()
    accepted_left = tasks.MajorityExample(bitstring="0000", bits=4, ones=0, majority=0, target_mode="plain_output")
    accepted_right = tasks.MajorityExample(bitstring="00000", bits=5, ones=0, majority=0, target_mode="plain_output")
    rejected_left = tasks.MajorityExample(bitstring="1111", bits=4, ones=4, majority=1, target_mode="plain_output")
    rejected_right = tasks.MajorityExample(bitstring="00000", bits=5, ones=0, majority=0, target_mode="plain_output")
    accepted = tasks.compose_majority_examples(accepted_left, accepted_right)
    rejected = tasks.compose_majority_examples(rejected_left, rejected_right)

    slices = task.split_composed_eval_slices(
        [accepted, rejected],
        {
            tasks.majority_key(accepted): [tasks.majority_key(accepted_left), tasks.majority_key(accepted_right)],
            tasks.majority_key(rejected): [tasks.majority_key(rejected_left), tasks.majority_key(rejected_right)],
        },
    )

    assert slices["accepted_by_guard"] == [accepted]
    assert slices["rejected_by_guard"] == [rejected]
    assert slices["all"] == [accepted, rejected]


def test_addition_task_compose_corrupt_applies_numeric_shift(monkeypatch):
    task = tasks.AdditionTask()
    base_example = AdditionExample(a=10, b=20, result=30, digits=2, has_carry=False)
    composed_example = AdditionExample(a=11, b=22, result=33, digits=2, has_carry=False)

    monkeypatch.setattr(
        tasks,
        "generate_prediction_map",
        lambda **kwargs: {tasks.example_key(base_example): "30"},
    )
    monkeypatch.setattr(
        tasks,
        "build_composed_pseudo_map",
        lambda *args, **kwargs: {tasks.example_key(composed_example): "99"},
    )

    pseudo_examples, missing_total, diagnostics = task.derive_round_targets(
        model=None,
        tokenizer=None,
        composed_examples=[composed_example],
        component_map={},
        target_max_size=2,
        base_examples=[base_example],
        batch_size=1,
        decode_max_new_tokens=4,
        args=SimpleNamespace(
            pseudo_label_mode="compose_corrupt",
            composed_strategy="with_carry",
            composition_error_percent=0.0,
            corruption_rate=1.0,
        ),
        rng=random.Random(0),
    )

    assert missing_total == 0
    assert diagnostics["corrupted_total"] == 1
    assert pseudo_examples[0].target() == "100"


def test_addition_composed_dataset_can_target_no_boundary_bucket():
    base_examples = [
        AdditionExample(a=1, b=2, result=3, digits=1, has_carry=False),
        AdditionExample(a=3, b=4, result=7, digits=1, has_carry=False),
        AdditionExample(a=1, b=1, result=2, digits=1, has_carry=False),
        AdditionExample(a=8, b=7, result=15, digits=1, has_carry=True),
    ]
    component_records = {"train": {}, "validation": {}, "test": {}}

    generated = build_composed_datasets(
        base_splits={"train": list(base_examples), "validation": [], "test": []},
        min_digits=2,
        max_digits=2,
        per_digit_counts={"train": 5, "validation": 0, "test": 0},
        rng=random.Random(0),
        record_components=component_records,
        allow_carry=True,
        allow_nocarry=True,
        boundary_carry_policy="no_boundary_carry",
    )

    assert len(generated["train"]) == 5
    for example in generated["train"]:
        status = tasks.get_boundary_carry_status(example, component_records["train"])
        assert status is False


def test_addition_fixed_width_mixed_prompt_formats_internal_zero_padding():
    example = AdditionExample(a=1, b=23, result=24, digits=2, has_carry=False, operand_width=2)

    assert example.prompt() == "Q: 1 + 23 = ?\nA:"
    assert example.target() == "24"
    assert example.formatted_a() == "01"
    assert example.formatted_b() == "23"


def test_addition_fixed_width_generation_allows_short_visible_operands():
    rng = random.Random(0)
    fixed_examples = [
        generate_addition_pair(
            2,
            rng=rng,
            addition_width_mode=ADDITION_WIDTH_FIXED_MIXED_PROMPT,
        )
        for _ in range(200)
    ]
    exact_examples = [generate_addition_pair(2, rng=random.Random(i)) for i in range(50)]

    assert any(example.a < 10 or example.b < 10 for example in fixed_examples)
    assert all(example.a >= 10 and example.b >= 10 for example in exact_examples)


def test_addition_balanced_visible_length_sampling_stratifies_operands():
    generated = build_length_bucket_dataset(
        min_digits=3,
        max_digits=3,
        per_digit_counts={"train": 9, "validation": 0, "test": 0},
        allow_carry=True,
        rng=random.Random(0),
        addition_width_mode=ADDITION_WIDTH_FIXED_MIXED_PROMPT,
        addition_sampling_mode=ADDITION_SAMPLING_BALANCED_VISIBLE_LENGTHS,
    )

    visible_pairs = {(len(str(example.a)), len(str(example.b))) for example in generated["train"]}
    assert visible_pairs == {
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
    }


def test_addition_balanced_visible_length_sampling_requires_fixed_width_mode():
    try:
        build_length_bucket_dataset(
            min_digits=2,
            max_digits=2,
            per_digit_counts={"train": 1, "validation": 0, "test": 0},
            allow_carry=True,
            rng=random.Random(0),
            addition_sampling_mode=ADDITION_SAMPLING_BALANCED_VISIBLE_LENGTHS,
        )
    except ValueError as exc:
        assert "fixed_width_mixed_prompt" in str(exc)
    else:
        raise AssertionError("Expected balanced_visible_lengths to require fixed_width_mixed_prompt.")


def test_addition_fixed_binary_composition_uses_single_split():
    left = AdditionExample(a=1234, b=1111, result=2345, digits=4, has_carry=False, operand_width=4)
    right = AdditionExample(a=12345, b=11111, result=23456, digits=5, has_carry=False, operand_width=5)
    component_records = {"train": {}, "validation": {}, "test": {}}

    generated = build_composed_datasets(
        base_splits={"train": [left, right], "validation": [], "test": []},
        min_digits=9,
        max_digits=9,
        per_digit_counts={"train": 1, "validation": 0, "test": 0},
        rng=random.Random(0),
        record_components=component_records,
        allow_carry=True,
        allow_nocarry=True,
        composition_path_mode=COMPOSITION_PATH_FIXED_BINARY,
    )

    assert len(generated["train"]) == 1
    for example in generated["train"]:
        component_widths = [key[0] for key in component_records["train"][example_key(example)]]
        assert component_widths == [4, 5]


def test_addition_composed_pseudo_map_pads_lower_fixed_width_blocks():
    left = AdditionExample(a=1, b=23, result=24, digits=2, has_carry=False, operand_width=2)
    right = AdditionExample(a=4, b=5, result=9, digits=2, has_carry=False, operand_width=2)
    composed = compose_examples(left, right)
    component_map = {example_key(composed): [example_key(left), example_key(right)]}
    weak_predictions = {example_key(left): "24", example_key(right): "9"}

    pseudo_map = build_composed_pseudo_map(
        {},
        [composed],
        component_map,
        weak_predictions,
        filter_component_carries=False,
    )

    assert composed.prompt() == "Q: 104 + 2305 = ?\nA:"
    assert str(composed.result) == "2409"
    assert pseudo_map[example_key(composed)] == "2409"


def test_multiplication_task_oracle_aggregation_and_corruption(monkeypatch):
    task = tasks.MultiplicationTask()
    example = tasks.MultiplicationExample(a=1234, b=5678, digits=4, result=1234 * 5678, operand_width=4)
    payload = tasks.build_multiplication_component_payload(example, 2)
    component_map = {tasks.multiplication_key(example): payload}

    def fake_prediction_map(**kwargs):
        return {
            tasks.multiplication_key(component): str(component.result)
            for component in kwargs["examples"]
        }

    monkeypatch.setattr(tasks, "generate_prediction_map", fake_prediction_map)

    compose_examples, missing_total, diagnostics = task.derive_round_targets(
        model=None,
        tokenizer=None,
        composed_examples=[example],
        component_map=component_map,
        target_max_size=4,
        base_examples=[],
        batch_size=1,
        decode_max_new_tokens=8,
        args=SimpleNamespace(
            pseudo_label_mode="compose",
            corruption_rate=0.0,
            block_size=2,
            oracle_aggregation=True,
        ),
        rng=random.Random(0),
    )
    assert missing_total == 0
    assert compose_examples[0].target() == str(example.result)

    corrupt_examples, _, corrupt_diagnostics = task.derive_round_targets(
        model=None,
        tokenizer=None,
        composed_examples=[example],
        component_map=component_map,
        target_max_size=4,
        base_examples=[],
        batch_size=1,
        decode_max_new_tokens=8,
        args=SimpleNamespace(
            pseudo_label_mode="compose_corrupt",
            corruption_rate=1.0,
            block_size=2,
            oracle_aggregation=True,
        ),
        rng=random.Random(0),
    )
    assert corrupt_diagnostics["corrupted_component_total"] == len(payload["partials"])
    assert corrupt_examples[0].target() != str(example.result)


def test_symbolic_multiplication_prompt_target_and_parser():
    example = tasks.MultiplicationExample(
        a=3,
        b=47,
        digits=2,
        result=141,
        operand_width=2,
        format_version="symbolic_v1",
    )
    assert example.prompt() == "03×47="
    assert example.target() == "0141"
    assert example.target_prefix() == ""
    assert tasks.parse_multiplication_prediction("141", example) == "0141"


def test_multiplication_slice_partition_is_total():
    task = tasks.MultiplicationTask()
    examples = [
        tasks.MultiplicationExample(a=1234, b=5678, digits=4, result=1234 * 5678, operand_width=4),
        tasks.MultiplicationExample(a=123456, b=654321, digits=6, result=123456 * 654321, operand_width=6),
    ]
    component_map = {
        tasks.multiplication_key(example): tasks.build_multiplication_component_payload(example, 2)
        for example in examples
    }

    slices = task.split_composed_eval_slices(examples, component_map)
    assert sum(len(bucket) for bucket in slices.values()) == len(examples)


def test_run_length_parse_and_compose():
    left = tasks.RunLengthExample(bitstring="0111", bits=4, max_run=3, prefix_run=1, suffix_run=3)
    right = tasks.RunLengthExample(bitstring="1101", bits=4, max_run=2, prefix_run=2, suffix_run=1)
    composed = tasks.compose_run_length_examples(left, right)

    assert composed.bitstring == "01111101"
    assert composed.max_run == 5
    assert composed.prefix_run == 1
    assert composed.suffix_run == 1
    assert tasks.parse_run_length_prediction("5 1 1") == "5|1|1"


def test_symbolic_run_length_prompt_and_target_prefix():
    example = tasks.RunLengthExample(
        bitstring="0111",
        bits=4,
        max_run=3,
        prefix_run=1,
        suffix_run=3,
        format_version="symbolic_v1",
    )
    assert example.prompt() == "runlen(0111)="
    assert example.target() == "3|1|3"
    assert example.target_prefix() == ""


def test_plain_output_run_length_target_and_parser():
    example = tasks.RunLengthExample(
        bitstring="0111",
        bits=4,
        max_run=3,
        prefix_run=1,
        suffix_run=3,
        target_mode="plain_output",
    )
    assert example.target() == "3"
    assert tasks.parse_run_length_prediction("answer 3", example) == "3"


def test_symbol_run_pair_target_parser_and_leftmost_tie_breaking():
    example = tasks.RunLengthExample(
        bitstring="00111222",
        bits=8,
        max_run=3,
        prefix_run=2,
        suffix_run=3,
        target_mode="symbol_run_pair",
    )

    assert tasks.leftmost_max_run_pair(example.bitstring) == ("1", 3)
    assert example.target() == "1|3"
    assert tasks.parse_run_length_prediction("answer 1|3", example) == "1|3"
    assert tasks.parse_run_length_prediction("A: 9|3 then 1|3", example) == "1|3"
    assert tasks.parse_run_length_prediction("answer 1|-3", example) is None
    assert tasks.parse_run_length_prediction("answer 9|3", example) is None


def test_run_length_multisymbol_generation_tracks_runs_of_any_repeated_symbol():
    generated = tasks.build_run_length_length_bucket_dataset(
        min_bits=6,
        max_bits=6,
        per_bit_counts={"train": 16, "validation": 0, "test": 0},
        rng=random.Random(0),
        alphabet="012",
    )

    assert generated["train"]
    for example in generated["train"]:
        assert set(example.bitstring).issubset(set("012"))
        longest = 1
        current = 1
        for prev, ch in zip(example.bitstring, example.bitstring[1:]):
            if ch == prev:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        assert example.max_run == longest


def test_run_length_seed_split_can_reserve_heldout_first():
    generated = tasks.build_run_length_length_bucket_dataset(
        min_bits=4,
        max_bits=4,
        per_bit_counts={"train": 100, "validation": 4, "test": 4},
        rng=random.Random(0),
        split_order=("validation", "test", "train"),
    )

    assert len(generated["validation"]) == 4
    assert len(generated["test"]) == 4
    assert len(generated["train"]) == 8


def test_run_length_exact2_composed_dataset_records_two_children():
    base_examples = [
        tasks.RunLengthExample(bitstring="0000", bits=4, max_run=0, prefix_run=0, suffix_run=0),
        tasks.RunLengthExample(bitstring="11111", bits=5, max_run=5, prefix_run=5, suffix_run=5),
        tasks.RunLengthExample(bitstring="010101", bits=6, max_run=1, prefix_run=0, suffix_run=1),
    ]
    component_records = {"train": {}, "validation": {}, "test": {}}

    generated = tasks.build_run_length_composed_dataset(
        base_splits={"train": list(base_examples), "validation": [], "test": []},
        min_bits=9,
        max_bits=9,
        per_bit_counts={"train": 3, "validation": 0, "test": 0},
        rng=random.Random(0),
        record_components=component_records,
        compose_arity="exact2",
    )

    assert len(generated["train"]) == 3
    for example in generated["train"]:
        component_keys = component_records["train"][tasks.run_length_key(example)]
        assert len(component_keys) == 2
        assert sum(key[0] for key in component_keys) == 9


def test_run_length_task_compose_corrupt_changes_summary(monkeypatch):
    class FixedRng:
        def random(self) -> float:
            return 0.0

        def randrange(self, _: int) -> int:
            return 0

    task = tasks.RunLengthTask()
    left = tasks.RunLengthExample(bitstring="101", bits=3, max_run=1, prefix_run=1, suffix_run=1)
    right = tasks.RunLengthExample(bitstring="010", bits=3, max_run=1, prefix_run=0, suffix_run=0)
    composed = tasks.compose_run_length_examples(left, right)
    component_map = {tasks.run_length_key(composed): [tasks.run_length_key(left), tasks.run_length_key(right)]}

    def fake_prediction_map(**kwargs):
        return {tasks.run_length_key(example): example.target() for example in kwargs["examples"]}

    monkeypatch.setattr(tasks, "generate_prediction_map", fake_prediction_map)

    pseudo_examples, missing_total, diagnostics = task.derive_round_targets(
        model=None,
        tokenizer=None,
        composed_examples=[composed],
        component_map=component_map,
        target_max_size=6,
        base_examples=[left, right],
        batch_size=1,
        decode_max_new_tokens=8,
        args=SimpleNamespace(pseudo_label_mode="compose_corrupt", corruption_rate=1.0),
        rng=FixedRng(),
    )

    assert missing_total == 0
    assert diagnostics["corrupted_examples"] == 1
    assert pseudo_examples[0].target() == "2|1|0"


def test_run_length_plain_output_guarded_compose_retains_only_safe_boundaries(monkeypatch):
    task = tasks.RunLengthTask()
    safe_left = tasks.RunLengthExample(bitstring="0100", bits=4, max_run=2, prefix_run=1, suffix_run=2, target_mode="plain_output")
    safe_right = tasks.RunLengthExample(bitstring="21111", bits=5, max_run=4, prefix_run=1, suffix_run=4, target_mode="plain_output")
    unsafe_left = tasks.RunLengthExample(bitstring="0011", bits=4, max_run=2, prefix_run=2, suffix_run=2, target_mode="plain_output")
    unsafe_right = tasks.RunLengthExample(bitstring="11000", bits=5, max_run=2, prefix_run=2, suffix_run=0, target_mode="plain_output")
    safe_example = tasks.compose_run_length_examples(safe_left, safe_right)
    unsafe_example = tasks.compose_run_length_examples(unsafe_left, unsafe_right)

    def fake_prediction_map(**kwargs):
        return {
            tasks.run_length_key(safe_left): "2",
            tasks.run_length_key(safe_right): "4",
            tasks.run_length_key(unsafe_left): "2",
            tasks.run_length_key(unsafe_right): "2",
        }

    monkeypatch.setattr(tasks, "generate_prediction_map", fake_prediction_map)

    pseudo_examples, missing_total, diagnostics = task.derive_round_targets(
        model=None,
        tokenizer=None,
        composed_examples=[safe_example, unsafe_example],
        component_map={
            tasks.run_length_key(safe_example): [tasks.run_length_key(safe_left), tasks.run_length_key(safe_right)],
            tasks.run_length_key(unsafe_example): [tasks.run_length_key(unsafe_left), tasks.run_length_key(unsafe_right)],
        },
        target_max_size=9,
        base_examples=[safe_left, safe_right, unsafe_left, unsafe_right],
        batch_size=1,
        decode_max_new_tokens=8,
        args=SimpleNamespace(
            pseudo_label_mode="compose",
            corruption_rate=0.0,
            target_mode="plain_output",
            compose_arity="exact2",
            guarded_compose_rule="run_length_no_boundary_continue",
            expand_train_per_size=1,
        ),
        rng=random.Random(0),
    )

    assert missing_total == 0
    assert [example.target() for example in pseudo_examples] == ["4"]
    assert diagnostics["requested_total"] == 1
    assert diagnostics["retained_total"] == 1
    assert diagnostics["rejected_total"] == 1


def test_run_length_symbol_pair_guarded_compose_uses_left_tie(monkeypatch):
    task = tasks.RunLengthTask()
    safe_left = tasks.RunLengthExample(
        bitstring="0100",
        bits=4,
        max_run=2,
        prefix_run=1,
        suffix_run=2,
        target_mode="symbol_run_pair",
    )
    safe_right = tasks.RunLengthExample(
        bitstring="21111",
        bits=5,
        max_run=4,
        prefix_run=1,
        suffix_run=4,
        target_mode="symbol_run_pair",
    )
    tie_left = tasks.RunLengthExample(
        bitstring="0001",
        bits=4,
        max_run=3,
        prefix_run=3,
        suffix_run=1,
        target_mode="symbol_run_pair",
    )
    tie_right = tasks.RunLengthExample(
        bitstring="2221",
        bits=4,
        max_run=3,
        prefix_run=3,
        suffix_run=1,
        target_mode="symbol_run_pair",
    )
    unsafe_left = tasks.RunLengthExample(
        bitstring="0011",
        bits=4,
        max_run=2,
        prefix_run=2,
        suffix_run=2,
        target_mode="symbol_run_pair",
    )
    unsafe_right = tasks.RunLengthExample(
        bitstring="11000",
        bits=5,
        max_run=3,
        prefix_run=2,
        suffix_run=3,
        target_mode="symbol_run_pair",
    )
    safe_example = tasks.compose_run_length_examples(safe_left, safe_right)
    tie_example = tasks.compose_run_length_examples(tie_left, tie_right)
    unsafe_example = tasks.compose_run_length_examples(unsafe_left, unsafe_right)

    def fake_prediction_map(**kwargs):
        return {
            tasks.run_length_key(safe_left): "0|2",
            tasks.run_length_key(safe_right): "1|4",
            tasks.run_length_key(tie_left): "0|3",
            tasks.run_length_key(tie_right): "2|3",
            tasks.run_length_key(unsafe_left): "0|2",
            tasks.run_length_key(unsafe_right): "0|3",
        }

    monkeypatch.setattr(tasks, "generate_prediction_map", fake_prediction_map)

    pseudo_examples, missing_total, diagnostics = task.derive_round_targets(
        model=None,
        tokenizer=None,
        composed_examples=[safe_example, tie_example, unsafe_example],
        component_map={
            tasks.run_length_key(safe_example): [tasks.run_length_key(safe_left), tasks.run_length_key(safe_right)],
            tasks.run_length_key(tie_example): [tasks.run_length_key(tie_left), tasks.run_length_key(tie_right)],
            tasks.run_length_key(unsafe_example): [tasks.run_length_key(unsafe_left), tasks.run_length_key(unsafe_right)],
        },
        target_max_size=9,
        base_examples=[safe_left, safe_right, tie_left, tie_right, unsafe_left, unsafe_right],
        batch_size=1,
        decode_max_new_tokens=8,
        args=SimpleNamespace(
            pseudo_label_mode="compose",
            corruption_rate=0.0,
            target_mode="symbol_run_pair",
            compose_arity="exact2",
            guarded_compose_rule="run_length_no_boundary_continue",
            expand_train_per_size=1,
        ),
        rng=random.Random(0),
    )

    assert missing_total == 0
    assert [example.target() for example in pseudo_examples] == ["1|4", "0|3"]
    assert diagnostics["requested_total"] == 2
    assert diagnostics["retained_total"] == 2
    assert diagnostics["rejected_total"] == 1


def test_run_length_plain_output_composed_eval_slices_partition_guard():
    task = tasks.RunLengthTask()
    safe_left = tasks.RunLengthExample(bitstring="0100", bits=4, max_run=2, prefix_run=1, suffix_run=2, target_mode="plain_output")
    safe_right = tasks.RunLengthExample(bitstring="21111", bits=5, max_run=4, prefix_run=1, suffix_run=4, target_mode="plain_output")
    unsafe_left = tasks.RunLengthExample(bitstring="0011", bits=4, max_run=2, prefix_run=2, suffix_run=2, target_mode="plain_output")
    unsafe_right = tasks.RunLengthExample(bitstring="11000", bits=5, max_run=2, prefix_run=2, suffix_run=0, target_mode="plain_output")
    safe_example = tasks.compose_run_length_examples(safe_left, safe_right)
    unsafe_example = tasks.compose_run_length_examples(unsafe_left, unsafe_right)

    slices = task.split_composed_eval_slices(
        [safe_example, unsafe_example],
        {
            tasks.run_length_key(safe_example): [tasks.run_length_key(safe_left), tasks.run_length_key(safe_right)],
            tasks.run_length_key(unsafe_example): [tasks.run_length_key(unsafe_left), tasks.run_length_key(unsafe_right)],
        },
    )

    assert slices["accepted_by_guard"] == [safe_example]
    assert slices["rejected_by_guard"] == [unsafe_example]
    assert slices["all"] == [safe_example, unsafe_example]


def test_run_length_symbol_pair_composed_eval_slices_partition_guard():
    task = tasks.RunLengthTask()
    safe_left = tasks.RunLengthExample(bitstring="0100", bits=4, max_run=2, prefix_run=1, suffix_run=2, target_mode="symbol_run_pair")
    safe_right = tasks.RunLengthExample(bitstring="21111", bits=5, max_run=4, prefix_run=1, suffix_run=4, target_mode="symbol_run_pair")
    unsafe_left = tasks.RunLengthExample(bitstring="0011", bits=4, max_run=2, prefix_run=2, suffix_run=2, target_mode="symbol_run_pair")
    unsafe_right = tasks.RunLengthExample(bitstring="11000", bits=5, max_run=3, prefix_run=2, suffix_run=3, target_mode="symbol_run_pair")
    safe_example = tasks.compose_run_length_examples(safe_left, safe_right)
    unsafe_example = tasks.compose_run_length_examples(unsafe_left, unsafe_right)

    slices = task.split_composed_eval_slices(
        [safe_example, unsafe_example],
        {
            tasks.run_length_key(safe_example): [tasks.run_length_key(safe_left), tasks.run_length_key(safe_right)],
            tasks.run_length_key(unsafe_example): [tasks.run_length_key(unsafe_left), tasks.run_length_key(unsafe_right)],
        },
    )

    assert slices["accepted_by_guard"] == [safe_example]
    assert slices["rejected_by_guard"] == [unsafe_example]
    assert slices["all"] == [safe_example, unsafe_example]


class DummyPromptExample:
    def __init__(self, prompt_text: str, target_text: str) -> None:
        self._prompt_text = prompt_text
        self._target_text = target_text

    def prompt(self) -> str:
        return self._prompt_text

    def target(self) -> str:
        return self._target_text


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = None
    padding_side = "left"

    def __init__(self) -> None:
        self._token_to_id = {"<pad>": self.pad_token_id}
        self._id_to_token = {self.pad_token_id: "<pad>"}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        for token in text.split():
            if token not in self._token_to_id:
                next_id = len(self._token_to_id)
                self._token_to_id[token] = next_id
                self._id_to_token[next_id] = token
            ids.append(self._token_to_id[token])
        return ids

    def __call__(self, prompts, return_tensors: str = "pt", padding: bool = True, truncation: bool = True):
        encoded = [self.encode(prompt, add_special_tokens=False) for prompt in prompts]
        max_len = max(len(row) for row in encoded)
        padded = []
        attention = []
        for row in encoded:
            pad = [self.pad_token_id] * (max_len - len(row))
            padded.append(pad + row)
            attention.append([0] * len(pad) + [1] * len(row))
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        tokens = []
        for idx in ids:
            token = self._id_to_token.get(int(idx), "")
            if skip_special_tokens and token == "<pad>":
                continue
            tokens.append(token)
        return " ".join(tokens)


class DummyGenerationModel(torch.nn.Module):
    def __init__(self, tokenizer: DummyTokenizer, completions: dict[str, str]) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.completions = completions
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def generate(self, input_ids, attention_mask=None, max_new_tokens: int = 0, do_sample: bool = False):
        rows = []
        for row in input_ids.tolist():
            prompt = self.tokenizer.decode(row, skip_special_tokens=True)
            completion = self.completions[prompt]
            rows.append(row + self.tokenizer.encode(completion, add_special_tokens=False))
        max_len = max(len(row) for row in rows)
        padded = [row + [self.tokenizer.pad_token_id] * (max_len - len(row)) for row in rows]
        return torch.tensor(padded, dtype=torch.long, device=input_ids.device)


def test_generation_helpers_slice_after_full_left_padded_prompt():
    tokenizer = DummyTokenizer()
    examples = [
        DummyPromptExample("Q: 9999 = ?", "7"),
        DummyPromptExample("Q: 1 1 1 1 1 1 1 1 = ?", "42"),
    ]
    completions = {example.prompt(): example.target() for example in examples}
    model = DummyGenerationModel(tokenizer, completions)

    accuracy, _ = evaluate_accuracy_with_breakdown(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=2,
        max_new_tokens=2,
        size_getter=lambda _: 0,
        prediction_parser=extract_numeric_answer,
    )
    prediction_map = generate_prediction_map(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=2,
        max_new_tokens=2,
        key_getter=lambda example: example.prompt(),
        prediction_parser=extract_numeric_answer,
    )

    assert accuracy == 1.0
    assert prediction_map[examples[0].prompt()] == "7"
    assert prediction_map[examples[1].prompt()] == "42"
