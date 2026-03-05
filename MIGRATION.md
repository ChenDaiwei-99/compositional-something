# Migration Guide

This repository has had two major layout changes:

- `78b17ec` checkpoint before refactoring.
- `e2662ba` split into `w2s/core`, `w2s/self`, `w2s/meta`, `w2s/launchers`, `w2s/legacy`.
- Current change: removed the `w2s/` namespace and moved those folders to repository root.

## Path Mapping (latest)

- `w2s/core/` -> `core/`
- `w2s/self/` -> `self/`
- `w2s/meta/` -> `meta/`
- `w2s/launchers/` -> `launchers/`
- `w2s/legacy/` -> `legacy/`

## Module Invocation Mapping

- `python -m w2s.self.self_improvement` -> `python -m self.self_improvement`
- `python -m w2s.self.self_improvement_composition_error_experiment` -> `python -m self.self_improvement_composition_error_experiment`
- `python -m w2s.meta.train_meta_self_improvement_rope` -> `python -m meta.train_meta_self_improvement_rope`

## Artifact Paths

Artifacts remain in `artifacts/`:

- `artifacts/logs/`
- `artifacts/models/`
- `artifacts/runs/self_improvement/`
- `artifacts/runs/meta_self_improvement/`
