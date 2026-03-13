# Legacy vs Progressive Comparator Notebook

Use this guide after both comparison jobs finish to build a quick notebook that compares:
- `--capacity-growth-scheme legacy`
- `--capacity-growth-scheme progressive`

The notebook reads:
- `submission_info.txt`
- `legacy/summary.json`
- `progressive/summary.json`

from the latest run directory under:
`artifacts/runs/meta_self_improvement/compare_legacy_vs_progressive_*`.

## 1) Confirm jobs are complete

```bash
sacct -j 5391140,5391141 --format=JobID,State,Elapsed,ExitCode
```

## 2) Create notebook

```bash
jupyter lab
```

Create a new notebook and paste the cells below.

## 3) Notebook cells

### Cell A: imports and run discovery

```python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 240)

ROOT = Path.cwd().resolve()
if not (ROOT / "artifacts").exists() and (ROOT / "logs").exists():
    ROOT = ROOT.parent

base_parent = ROOT / "artifacts" / "runs" / "meta_self_improvement"
candidates = sorted(base_parent.glob("compare_legacy_vs_progressive_*"))
if not candidates:
    raise FileNotFoundError(f"No compare run found under {base_parent}")

run_dir = candidates[-1]
print("Using run_dir:", run_dir)
print("Submission file exists:", (run_dir / "submission_info.txt").exists())
```

### Cell B: load summaries

```python
def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

legacy_path = run_dir / "legacy" / "summary.json"
progressive_path = run_dir / "progressive" / "summary.json"

if not legacy_path.exists():
    raise FileNotFoundError(f"Missing: {legacy_path}")
if not progressive_path.exists():
    raise FileNotFoundError(f"Missing: {progressive_path}")

legacy = load_summary(legacy_path)
progressive = load_summary(progressive_path)

legacy_rounds = pd.DataFrame(legacy.get("rounds", [])).sort_values("round_index").reset_index(drop=True)
progressive_rounds = pd.DataFrame(progressive.get("rounds", [])).sort_values("round_index").reset_index(drop=True)

print("legacy rounds:", len(legacy_rounds))
print("progressive rounds:", len(progressive_rounds))
```

### Cell C: compact side-by-side metrics

```python
def summarize(name: str, payload: dict, rounds: pd.DataFrame) -> dict:
    if rounds.empty:
        return {
            "scheme": name,
            "stop_reason": payload.get("stop_reason"),
            "num_rounds": 0,
            "num_growth_events": len(payload.get("growth_events", [])),
            "final_stage_index": payload.get("final_stage_index"),
            "final_frontier_max_digits": payload.get("final_frontier_max_digits"),
            "best_frontier_val": None,
            "best_initial_val": None,
            "best_full_val": None,
        }
    return {
        "scheme": name,
        "stop_reason": payload.get("stop_reason"),
        "num_rounds": int(len(rounds)),
        "num_growth_events": int(len(payload.get("growth_events", []))),
        "final_stage_index": int(payload.get("final_stage_index", -1)),
        "final_frontier_max_digits": int(payload.get("final_frontier_max_digits", -1)),
        "best_frontier_val": float(rounds["frontier_validation_accuracy"].max()),
        "best_initial_val": float(rounds["initial_validation_accuracy"].max()),
        "best_full_val": float(rounds["full_validation_accuracy"].max()) if "full_validation_accuracy" in rounds else None,
        "last_frontier_val": float(rounds.iloc[-1]["frontier_validation_accuracy"]),
        "last_initial_val": float(rounds.iloc[-1]["initial_validation_accuracy"]),
    }

summary_df = pd.DataFrame([
    summarize("legacy", legacy, legacy_rounds),
    summarize("progressive", progressive, progressive_rounds),
])
summary_df
```

### Cell D: validation curves

```python
fig, ax = plt.subplots(1, 1, figsize=(12, 5))

ax.plot(
    legacy_rounds["round_index"],
    legacy_rounds["frontier_validation_accuracy"],
    label="legacy frontier",
    linewidth=2,
)
ax.plot(
    progressive_rounds["round_index"],
    progressive_rounds["frontier_validation_accuracy"],
    label="progressive frontier",
    linewidth=2,
)
ax.plot(
    legacy_rounds["round_index"],
    legacy_rounds["initial_validation_accuracy"],
    label="legacy initial",
    linestyle="--",
    alpha=0.7,
)
ax.plot(
    progressive_rounds["round_index"],
    progressive_rounds["initial_validation_accuracy"],
    label="progressive initial",
    linestyle="--",
    alpha=0.7,
)

ax.set_title("Legacy vs Progressive: Validation Accuracy")
ax.set_xlabel("Round")
ax.set_ylabel("Accuracy")
ax.legend()
plt.show()
```

### Cell E: stage + frontier progression

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(legacy_rounds["round_index"], legacy_rounds["stage_index"], label="legacy", linewidth=2)
axes[0].plot(progressive_rounds["round_index"], progressive_rounds["stage_index"], label="progressive", linewidth=2)
axes[0].set_ylabel("stage_index")
axes[0].set_title("Stage progression")
axes[0].legend()

axes[1].plot(legacy_rounds["round_index"], legacy_rounds["frontier_max_digits"], label="legacy", linewidth=2)
axes[1].plot(progressive_rounds["round_index"], progressive_rounds["frontier_max_digits"], label="progressive", linewidth=2)
axes[1].set_ylabel("frontier_max_digits")
axes[1].set_xlabel("round_index")
axes[1].set_title("Frontier progression")
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Cell F: growth events

```python
legacy_growth = pd.DataFrame(legacy.get("growth_events", []))
progressive_growth = pd.DataFrame(progressive.get("growth_events", []))

print("Legacy growth events")
display(legacy_growth)
print("Progressive growth events")
display(progressive_growth)
```

## 4) Optional: save figure/table artifacts

```python
out_dir = run_dir / "comparison_notebook_outputs"
out_dir.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(out_dir / "summary_comparison.csv", index=False)
print("Wrote:", out_dir / "summary_comparison.csv")
```
