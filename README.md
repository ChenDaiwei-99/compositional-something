# Experiment Layout

Repository modules are organized into:

- `core`: shared addition/composition pipeline utilities
- `self`: self-improvement experiment entrypoints
- `meta`: meta self-improvement experiment entrypoints
- `launchers`: Slurm/shell launch scripts
- `legacy/weak_to_strong`: archived weak-to-strong scripts

## Canonical Commands

Run self-improvement:

```bash
python -m self.self_improvement --help
```

Run self-improvement composition wrapper:

```bash
python -m self.self_improvement_composition_error_experiment --help
```

Run meta self-improvement:

```bash
python -m meta.train_meta_self_improvement_rope --help
```

Artifact outputs default to `artifacts/`.
