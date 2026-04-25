# Fixed-Width Mixed-Prompt Addition

## Motivation

The current addition path uses exact-width operands for every digit bucket. For
example, a 2-digit addition example samples both operands from `10..99`, so
cases such as internal `01 + 23` are excluded. This is convenient for exact-digit
evaluation, but it does not test whether a model can handle fixed-width
composition blocks whose leading zeros disappear in the natural prompt.

The fixed-width mixed-prompt variant keeps composition defined over fixed-width
digit blocks, while showing the model ordinary unpadded arithmetic. Thus an
internal width-2 example may store `01 + 23`, but the model sees:

```text
Q: 1 + 23 = ?
A:
```

The goal is to preserve block-wise compositional structure without making the
prompt itself look like a digit-transduction task.

## Example Format

Each example should carry the following fields:

- `a`: left operand as an integer.
- `b`: right operand as an integer.
- `result`: arithmetic sum as an integer.
- `operand_width`: internal fixed width used for composition.
- `has_carry`: whether the full-width addition contains any carry.
- `target_override`: optional pseudo-label string used during self-improvement.

Prompt formatting uses the raw integer values:

```text
Q: {a} + {b} = ?
A:
```

Targets are also unpadded integer strings. For example:

```text
internal: a=1, b=23, operand_width=2
prompt:   Q: 1 + 23 = ?
target:   24
blocks:   01 + 23
```

The zero-padded block representation is used only by composition and
boundary-carry diagnostics, not by the model-facing prompt.

## Data Generation

The existing exact-digit mode remains unchanged:

- For width `d > 1`, sample each operand from `10^(d-1)..10^d-1`.
- For width `d = 1`, sample each operand from `0..9`.
- A width bucket is equivalent to visible operand digit length.

The new `fixed_width_mixed_prompt` mode changes only the operand sampler:

- For width `d`, sample each operand from `0..10^d-1`.
- A width bucket means internal operand width, not visible prompt length.
- Leading-zero cases are therefore natural and expected.

For non-carry examples, sample digit columns directly so every column sum is
less than `10`, including leading-zero columns. This allows internal blocks such
as `00`, `01`, and `07` while still guaranteeing no carry.

## Composition Rule

Composition concatenates zero-padded operand blocks.

For example, two width-2 components

```text
component 1: a=1,  b=23  -> blocks 01 + 23
component 2: a=4,  b=5   -> blocks 04 + 05
```

compose internally as:

```text
0104 + 2305
```

The composed model-facing prompt is:

```text
Q: 104 + 2305 = ?
A:
```

and the target is the unpadded arithmetic result:

```text
2409
```

Boundary-carry filtering must use the component `operand_width` values, not the
visible prompt digit lengths. In the example above, the boundary lies between
the two width-2 blocks even though the displayed left operand has only three
visible digits.

## Pseudo-Labeling

Direct pseudo-labeling queries the model on natural unpadded prompts.

Composed pseudo-labeling uses the same block structure as composition:

- Query the model on component prompts.
- Parse numeric component predictions.
- Combine component predictions according to the addition composition rule.
- Store the resulting unpadded integer string as `target_override`.

Rows with missing or unparseable numeric predictions are filtered. In
exact-count self-improvement runs, generation should refill until every active
width bucket reaches the requested retained count or fail loudly.

## Evaluation

Held-out splits are bucketed by internal `operand_width`, not by the number of
visible digits in the prompt. A width-2 evaluation bucket may therefore include
examples displayed as `1 + 23`, `99 + 0`, or `7 + 5`.

Primary metrics should report width-bucket accuracy. Optional diagnostics should
also report visible digit-length statistics, such as:

- visible length of `a`;
- visible length of `b`;
- whether either operand has leading zeros internally;
- accuracy grouped by visible length pair.

These diagnostics make it clear whether a model is failing on the newly exposed
mixed-width cases or on the underlying arithmetic.

## Compatibility

Existing exact-digit addition runs remain the default and should not change.

Old artifacts that do not contain `operand_width` should deserialize with:

```text
operand_width = digits
```

Existing paper-facing addition checkpoints and result files should not be
reinterpreted as fixed-width mixed-prompt runs. The new mode should record its
width semantics explicitly in run metadata, for example:

```text
addition_width_mode = fixed_width_mixed_prompt
```

## Expected Tests

Generation tests:

- In exact-digit mode, width-2 examples never include `a < 10` or `b < 10`.
- In fixed-width mixed-prompt mode, width-2 examples may include `a < 10` or
  `b < 10`.
- Non-carry fixed-width generation permits leading-zero columns while preserving
  the no-carry guarantee.

Format tests:

- `a=1, b=23, operand_width=2` prompts as `Q: 1 + 23 = ?\nA:`.
- The same example formats internally as `01 + 23`.
- The target is `24`, not `024`.

Composition tests:

- Two width-2 blocks `(1, 23)` and `(4, 5)` compose internally as
  `0104 + 2305`.
- The composed prompt displays `104 + 2305`.
- Boundary-carry filtering uses the two width-2 component boundaries.

Compatibility tests:

- Old serialized examples without `operand_width` load with
  `operand_width=digits`.
- Existing exact-digit generation, composition, and evaluation behavior is
  unchanged unless the new mode is explicitly selected.
