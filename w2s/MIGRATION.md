# W2S Refactor Migration Guide

Checkpoint commit before refactor: `78b17ec`.

This refactor intentionally drops old path compatibility. Use the new paths and module entrypoints below.

## Path Mapping

- `w2s/self_improvement.py` -> `w2s/self/self_improvement.py`
- `w2s/self_improvement_composition_error_experiment.py` -> `w2s/self/self_improvement_composition_error_experiment.py`
- `w2s/self_improvement_experiment.py` -> `w2s/self/self_improvement_experiment.py`
- `w2s/meta_self_improvement_rope.py` -> `w2s/meta/train_meta_self_improvement_rope.py`
- `w2s/weak_to_strong_addition_experiment_v2.py` -> `w2s/core/addition_pipeline.py`

### Launchers

- `w2s/run_self_improvement_mig_boundary_eval.sbatch` -> `w2s/launchers/self/run_self_improvement_mig_boundary_eval.sbatch`
- `w2s/run_self_improvement_qwen_no_growth.sbatch` -> `w2s/launchers/self/run_self_improvement_qwen_no_growth.sbatch`
- `w2s/run_composition_error_sweep_self_improvement.sh` -> `w2s/launchers/self/run_composition_error_sweep_self_improvement.sh`
- `w2s/run_meta_self_improvement_rope.sbatch` -> `w2s/launchers/meta/run_meta_self_improvement_rope.sbatch`
- `w2s/run_meta_selfimp_watchdog.sbatch` -> `w2s/launchers/meta/run_meta_selfimp_watchdog.sbatch`
- `w2s/watch_meta_self_improvement.sh` -> `w2s/launchers/meta/watch_meta_self_improvement.sh`

### Legacy Archive

Former weak-to-strong scripts are archived under:

- `w2s/legacy/weak_to_strong/`

## Artifact Relocation

Generated files moved to `artifacts/` (git-ignored):

- `logs/` -> `artifacts/logs/`
- `models/` -> `artifacts/models/`
- `w2s/self_improvement_runs/` -> `artifacts/runs/self_improvement/`
- `w2s/meta_self_improvement_runs/` -> `artifacts/runs/meta_self_improvement/`
