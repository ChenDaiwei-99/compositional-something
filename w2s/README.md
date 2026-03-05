# W2S Experiment Layout

This directory is organized into:

- `w2s/core`: shared addition/composition pipeline utilities
- `w2s/self`: self-improvement experiment entrypoints
- `w2s/meta`: meta self-improvement experiment entrypoints
- `w2s/launchers`: Slurm/shell launch scripts
- `w2s/legacy/weak_to_strong`: archived weak-to-strong scripts

## Canonical Commands

Run self-improvement:

```bash
python -m w2s.self.self_improvement --help
```

Run self-improvement composition wrapper:

```bash
python -m w2s.self.self_improvement_composition_error_experiment --help
```

Run meta self-improvement:

```bash
python -m w2s.meta.train_meta_self_improvement_rope --help
```

Artifact outputs default to `artifacts/`.
