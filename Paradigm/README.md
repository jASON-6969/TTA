# Method Paradigm Classification

This directory is organized according to the paradigm taxonomy used by the
local `vendor/MedSeg-TTA` registry. The copied method folders are source
snapshots for reference and attribution; they are not all executed by the
small CXR reproduction runner.

| Paradigm | Methods |
| --- | --- |
| Input-level Translation | AIF-SFDA, DL-TTA, RSA, SFDA-FSM, STDR |
| Feature-level Alignment | DANN, GraTa, TestFit, UDA-MIMA |
| Output-level Regularization | DeTTA, DG-TTA, SaTTCA, SmaRT, TENT, UPL-SFDA |
| Prior Estimation | AdaMI, ExploringTTA, PASS, ProSFDA, VPTTA |
| Baseline | Source-only |

`Source-only` is documented in `Baseline/README.md` because it performs no
test-time adaptation. The taxonomy describes the primary adaptation locus. A
method can use an auxiliary loss from another family without changing its
primary folder.

## Local CXR reproduction status

- Executed target-only methods: `TENT`, `TestFit`, and `GraTa`.
- Executed baseline: `Source-only`.
- Pilot-only method: `DANN`; this local pilot is source-assisted because it
  consumes source images and masks, so it is excluded from the strict
  source-free result table.
- Audit-only method: `UDA-MIMA`; the available wrapper constructs a source
  loader and is not included in the strict target-only result table.
- The remaining folders are benchmark method snapshots and are not claimed as
  executed by this CXR bundle.

## Classification notes

- `TENT` is grouped under Output-level Regularization, matching the benchmark
  registry, even though its implementation updates BatchNorm affine parameters
  using an entropy objective.
- `DeTTA` is grouped under Output-level Regularization, matching the current
  registry entry and package path.
- `VPTTA` is grouped under Prior Estimation, matching the benchmark taxonomy;
  its implementation also contains prompt and input-translation operations.

## Composition runner

The independent combination framework is in
[`tta_composition/`](tta_composition/README.md). It supports the empty set and
all 31 non-empty subsets of `VPTTA`, `DLTTA`, `TestFit`, `GraTa`, and `SmaRT`.
The runner uses a frozen source reference, an isolated adaptable student, and a
central proposal/commit scheduler. It records parameter ownership, learning
rate arbitration, source-buffer checks, per-image updates, and post-adaptation
metrics under `test_v2/composition_runs/` (relative to the repository root).

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
