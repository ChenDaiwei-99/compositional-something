# Legacy vs Progressive Comparator Notebook

Use this guide after a comparison batch finishes to build a quick notebook that compares:
- `--capacity-growth-scheme legacy`
- `--capacity-growth-scheme progressive_depth`
- `--capacity-growth-scheme progressive_depth_width`

The notebook reads:
- `submission_info.txt`
- `legacy/summary.json`
- `progressive_depth/summary.json`
- `progressive_depth_width/summary.json`

from the latest run directory under:
`artifacts/runs/meta_self_improvement/compare_legacy_vs_progressive_*`.

## 1) Confirm jobs are complete

Inspect the job ids from `submission_info.txt`, then run:

```bash
sacct -j <legacy_job>,<progressive_depth_job>,<progressive_depth_width_job> --format=JobID,State,Elapsed,ExitCode
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

def load_rounds(path: Path) -> pd.DataFrame:
    payload = load_summary(path)
    rounds = pd.DataFrame(payload.get("rounds", [])).sort_values("round_index").reset_index(drop=True)
    return payload, rounds

arm_dirs = {
    "legacy": run_dir / "legacy" / "summary.json",
    "progressive_depth": run_dir / "progressive_depth" / "summary.json",
    "progressive_depth_width": run_dir / "progressive_depth_width" / "summary.json",
}

payloads = {}
rounds_by_arm = {}
for arm_name, summary_path in arm_dirs.items():
    if summary_path.exists():
        payloads[arm_name], rounds_by_arm[arm_name] = load_rounds(summary_path)

if not payloads:
    raise FileNotFoundError(f"No summary.json files found under {run_dir}")

for arm_name, rounds in rounds_by_arm.items():
    print(f"{arm_name} rounds:", len(rounds))
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

summary_df = pd.DataFrame(
    [summarize(arm_name, payloads[arm_name], rounds_by_arm[arm_name]) for arm_name in payloads]
)
summary_df
```

### Cell D: validation curves

```python
label_map = {
    "legacy": "Legacy",
    "progressive_depth": "Progressive (Depth)",
    "progressive_depth_width": "Progressive (Depth+Width)",
}

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
for arm_name, rounds in rounds_by_arm.items():
    ax.plot(
        rounds["round_index"],
        rounds["frontier_validation_accuracy"],
        label=f"{label_map.get(arm_name, arm_name)} frontier",
        linewidth=2,
    )
    ax.plot(
        rounds["round_index"],
        rounds["initial_validation_accuracy"],
        label=f"{label_map.get(arm_name, arm_name)} initial",
        linestyle="--",
        alpha=0.7,
    )

ax.set_title("Legacy vs Progressive Variants: Validation Accuracy")
ax.set_xlabel("Round")
ax.set_ylabel("Accuracy")
ax.legend()
plt.show()
```

### Cell E: stage + frontier progression

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for arm_name, rounds in rounds_by_arm.items():
    axes[0].plot(rounds["round_index"], rounds["stage_index"], label=label_map.get(arm_name, arm_name), linewidth=2)
    axes[1].plot(rounds["round_index"], rounds["frontier_max_digits"], label=label_map.get(arm_name, arm_name), linewidth=2)

axes[0].set_ylabel("stage_index")
axes[0].set_title("Stage progression")
axes[0].legend()

axes[1].set_ylabel("frontier_max_digits")
axes[1].set_xlabel("round_index")
axes[1].set_title("Frontier progression")
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Cell F: growth events

```python
for arm_name, payload in payloads.items():
    print(label_map.get(arm_name, arm_name), "growth events")
    display(pd.DataFrame(payload.get("growth_events", [])))
```

## 4) Optional: save figure/table artifacts

```python
out_dir = run_dir / "comparison_notebook_outputs"
out_dir.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(out_dir / "summary_comparison.csv", index=False)
print("Wrote:", out_dir / "summary_comparison.csv")
```
