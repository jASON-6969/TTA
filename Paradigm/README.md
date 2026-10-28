# GUI TTA framework

This directory contains the TTA framework used by the desktop GUI. The legacy
method snapshot folders that the GUI did not load have been removed.

| Directory | Purpose |
| --- | --- |
| `tta_composition/` | GUI, runner, five composable TTA methods and tests |
| `Baseline/` | Source-only documentation; no test-time adaptation |

`launch_gui.py` loads `tta_composition.gui`. Starting an experiment launches
`Paradigm.tta_composition.run`, which calls `CompositionEngine` and the selected
methods in `tta_composition/methods/`. Selecting no methods runs Source-only.
The model, image loading and metrics are supplied by
`code/feature_alignment_2d_pilot/` at the repository root.

## Composition runner

The independent combination framework is in
[`tta_composition/`](tta_composition/README.md). It supports the empty set and
all 31 non-empty subsets of `VPTTA`, `DLTTA`, `TestFit`, `GraTa`, and `SmaRT`.
The runner uses a frozen source reference, an isolated adaptable student, and a
central proposal/commit scheduler. It records parameter ownership, learning
rate arbitration, source-buffer checks, per-image updates, and post-adaptation
metrics under `composition_runs/` (relative to the repository root).

After following the root README to install dependencies and restore the
bundled Git LFS data/checkpoint, run from the repository root:

```powershell
python -m Paradigm.tta_composition.run --methods VPTTA,DLTTA,TestFit,GraTa,SmaRT --max-cases 8 --seed 42 --output .\composition_runs\smoke_all_five
```

The composition modules are local variants inspired by the five methods; their
losses, parameter boundaries and prediction rules differ from the author
implementations. First use builds a complete target Source-only baseline,
then each experiment compares the same selected cases against that baseline.
The GUI supports editing hyperparameters; see the composition README for the
saved configuration, cache rules and comparison outputs.
