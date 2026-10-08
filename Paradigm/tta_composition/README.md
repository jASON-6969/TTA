# CXR TTA composition runner

`tta_composition` is the local combination framework for the five selected
test-time adaptation mechanisms: VPTTA, DLTTA, TestFit, GraTa and SmaRT. It
supports the empty set (Source-only) and every non-empty subset. The order in
`--methods` is treated as presentation input; the engine always uses the
fixed lifecycle `VPTTA -> signal construction -> central update -> prediction -> prompt memory`
and the canonical method order in its audit file.

The package bridges the existing five-level 2D UNet at
`code/feature_alignment_2d_pilot/model.py` inside this repository. The source checkpoint is
loaded into a frozen reference model and a separate student. Target records
passed to the engine have `mask_path=None`; Montgomery masks are read only by
the post-adaptation evaluator in `run.py`.

## Run

Clone with Git LFS to obtain the included benchmark data and source checkpoint,
then install the pinned environment using the [root README](../../README.md). From the
repository root (`test_v2` in this checkout), with that environment active:

```powershell
python -m Paradigm.tta_composition.run --methods VPTTA,DLTTA,TestFit,GraTa,SmaRT --max-cases 8 --seed 42 --output .\composition_runs\smoke_all_five
```

The default checkpoint is
`base_model/benchmark_cxr_base_no_source_aug/source_checkpoint.pth`. Use
`--checkpoint`, `--output`, `--config`, or `--device cpu` to override it. An
omitted `--methods` value with no config file runs Source-only. Results are written under
`composition_runs/seed_<seed>_<methods>/` unless `--output` is given.
Each run contains the resolved configuration, checkpoint hash, ownership
audit, JSONL per-image update records, predictions, and post-adaptation
metrics.

To open the desktop interface from the repository root (requires Tkinter):

```powershell
python launch_gui.py
```

The launcher validates the bundled data/split/weights, reports missing LFS
objects before starting, and opens the GUI with its default Montgomery inputs.
On Windows, `start_gui.cmd` reopens it using the clone's `.venv` after setup.
The GUI's background runner uses the same Python interpreter as the GUI.

The GUI uses the same runner in a background process, streams per-image
progress, and adds a timestamped run directory under the selected output
folder to avoid overwriting earlier results.

The desktop interface supports Montgomery, SZ-CXR, and a custom target image
folder. Use **檢查資料** to inspect image and label coverage before starting.
Choose a case limit or **全部影像**. Running, cancelled and failed experiments
have distinct statuses. On success, both the result panel and the log show
mean Dice, HD95 and IoU, the evaluated case count, and the output folder.
The complete UTF-8 log is saved as `execution.log`; numeric results are saved
in `summary.json` and per-case `metrics.csv`. Scores are calculated after the
entire adaptation stream, never used as adaptation supervision.

## Hyperparameters

Use **超參數…** to edit the next experiment. The dialog has a general tab and
one tab per method, with **套用**, **取消**, and **恢復預設**. Settings for an
unchecked method are retained. The controls are disabled during an experiment.

The five learning rates, input size/normalization, VPTTA prompt size/strength and
memory capacity, DLTTA memory capacity, TestFit confidence/feature weight,
GraTa consistency/feature weights and noise strength, and SmaRT EMA/structure/
consistency weights are connected to the actual adapters. Defaults remain in
`config.py`. Rates, weights and noise strengths must be finite and non-negative;
memory/prompt sizes are positive integers, confidence is in `[0, 1]`, EMA decay
is in `(0, 1]`, and the input size must be a positive multiple of 16.

Each GUI run saves `gui_config.json` before starting the child process and the
resolved `config.json` inside its result directory. CLI runs can use `--config`
with a JSON file, for example:

```json
{
  "methods": ["GraTa"],
  "lrs": {"GraTa": 0.0002},
  "grata_consistency_weight": 0.5,
  "grata_feature_weight": 0.1,
  "grata_noise_std": 0.03,
  "image_size": 256,
  "normalization": "zscore"
}
```

Save this as `experiment.json`, then run from the repository root:

```powershell
python -m Paradigm.tta_composition.run --config .\experiment.json --max-cases 8 --output .\composition_runs\trial_grata
```

## Full-dataset baseline and matched comparison

The first experiment on new target images first performs frozen Source-only
inference on **every image in the selected folder**, including unlabelled images,
regardless of the TTA case limit. Completed predictions are recorded under
`<output-folder>/baselines/<cache-key>/`. CLI runs default to a `baselines`
directory beside their run output; `--baseline-cache` can select a shared cache.

The cache key includes every image's content and path, the checkpoint hash,
input size/normalization, actual device, inference code and runtime versions.
Changing only methods, seed, TTA learning rates, loss weights or case limit
reuses the baseline. Changed source inputs or preprocessing create a new cache.
Masks are excluded from prediction identity; current labels are reevaluated
after TTA. Each completed cache contains `baseline_manifest.json`, `READY.json`
and a binary prediction PNG for every image. Partial or corrupted caches are
not reused; rebuilding preserves damaged caches in a quarantine directory.

All baseline predictions finish before adaptation starts. Label pixels are
opened only after the current experiment's predictions finish. The shared cache
then receives full-dataset `summary.json`, `metrics.csv` and a label manifest.
Source-only experiments copy their selected predictions from this cache.

Every run compares Base and TTA on **the same selected image IDs and masks**.
For example, an eight-case TTA run uses the baseline scores of those eight
images, while the complete baseline record still covers the full folder.
The result panel/log show Base, TTA and signed deltas (`TTA - Base`): larger
Dice/IoU is better, while smaller HD95 is better. Without masks, only prediction
pixel-change rates are recorded; they are not quality-improvement scores.

In addition to the existing outputs, each run contains:

- `comparison.json` and `comparison.csv`: matched aggregate and per-image
  Base/TTA scores, deltas and prediction changes.
- `baseline_summary.json` and `baseline_metrics.csv`: the complete baseline
  evaluation snapshot used by this experiment.
- `label_manifest.json`: mask paths/hashes for this evaluation, preserving old
  experiment provenance when the shared cache is reevaluated with new labels.

Run all focused GUI/cache/hyperparameter/comparison regressions without pytest:

```powershell
python -B -X utf8 -m unittest discover -s Paradigm\tta_composition\tests -p 'test_*.py' -v
```

The standalone functions in `tests/test_composition.py` additionally cover all
32 method subsets; they can be run with pytest when it is installed.

## Choose target data

`--dataset montgomery` is the default. It reads `data/raw/montgomery/images`
and evaluates using the union of the matching `left_mask` and `right_mask`.
`--dataset sz_cxr` reads `data/raw/sz_cxr/images` with optional matching
`masks/<image>_mask.png`. Selecting SZ-CXR changes the target domain; it does
not reproduce the standard SZ-CXR-to-Montgomery experiment.

For another binary 2D segmentation dataset, select **自訂資料夾** in the GUI,
or use:

```powershell
python -m Paradigm.tta_composition.run --dataset custom --image-dir 'C:\data\images' --mask-dir 'C:\data\masks' --max-cases 8
```

Image formats are PNG, JPG/JPEG, BMP and TIF/TIFF in the selected folder (no
recursive scan). Masks must share the image stem or use `<stem>_mask`; their
extension may differ. Duplicate image stems or ambiguous mask matches are
rejected. Binary masks encoded as 0/1 or 0/255 are supported; non-zero pixels
are foreground. Images are resized and converted to grayscale using the
existing checkpoint preprocessing; this entry point does not read DICOM or
3D volumes. HD95 retains the existing benchmark definition in pixels on the
resized image.

The mask folder is optional for custom data. Without labels, predictions and
adaptation logs are still saved; the GUI explicitly reports that Dice/HD95/IoU
were not evaluated. With partial labels, only paired cases enter the metric
summary, and both predicted and evaluated counts are shown. Montgomery needs
both lung masks for a complete ground truth. Mask discovery inspects filenames
only; mask pixels are first opened after all adaptation and predictions finish.

Run the focused desktop/data checks without pytest:

```powershell
python -X utf8 Paradigm\tta_composition\tests\run_gui_checks.py
```

## Composition boundaries

`adaptation_steps` defaults to 3 and can be set in the general GUI tab, JSON
configuration, or `--adaptation-steps`. Each iteration creates a fresh forward
graph, commits one update, runs `after_commit` once, and updates the SmaRT EMA.
DLTTA stores one feature entry per update, so a three-step image adds three
entries. VPTTA publishes its prompt only once after the final prediction.

Each image's JSONL record retains the final update fields for compatibility
and an `iterations` array with every update's loss, LR, gradient norms and
duration. Top-level duration covers the whole image; `method_metrics` describes
the last iteration. New runs save `provenance.json` and copy this code snapshot
into `audit.json`. Existing run artifacts cannot be overwritten; use a fresh
`--output` directory.

- VPTTA owns only its low-frequency prompt and private prompt memory.
- TestFit owns student non-BatchNorm parameters and uses the frozen source
  reference for pseudo-labels and feature targets.
- GraTa owns only BatchNorm affine parameters. Running statistics remain in
  evaluation mode and are never overwritten.
- SmaRT owns only its identity-initialised style encoder/modulation and
  zero-initialised output correction. Its EMA model is a separate supervision
  state.
- DLTTA owns no model parameter when another adapter supplies the host. It
  returns a distribution-based learning-rate hint. When DLTTA is the only
  model adapter, it hosts BN-affine entropy minimisation and is labelled
  `DLTTA + entropy host` by the ownership audit.

All gradients are proposed before any parameter is changed. The scheduler
checks finite values, rejects duplicate ownership, combines simultaneous
positive learning-rate hints with a geometric mean, and applies the update as
one batch. A zero hint skips that parameter group. Gradient norms are accumulated
in float64 and checked before parameters change, so a finite float32 gradient
does not become an infinite audit value through float32 norm overflow.

VPTTA keeps the same historical prompt memory for this image's adaptation and
prediction. Its updated prompt enters the FIFO only after prediction succeeds;
failed prediction does not publish that prompt. This uses the separate
`after_prediction` lifecycle hook.

SmaRT keeps nonzero hidden encoder features and zero-initialises only the final
style projection, preserving initial identity. That projection learns first;
subsequent steps pass gradients into earlier encoder layers. EMA consistency is
the class-summed KL **averaged across batch and pixels**, keeping its scale
consistent when the input resolution changes; the EMA target is detached.

The local structural loss combines neighborhood TV with an excess-region
surrogate. It selects 4-connected regions from detached predicted foreground
probabilities above 0.5 and keeps the two largest (row-major tie breaking).
Each remaining region contributes `0.01 * mean(P_foreground within region)`;
these contributions are summed per image and averaged across the batch.
Only selection is detached: the excess foreground probability loss backpropagates.
Zero, one or two regions have no excess penalty. `foreground_components` reports
the batch mean and `extra_component_loss` reports the unweighted excess term.

## Scope and limitations

Saved results copied from `test_v1` are historical records. A current default
does not retroactively change their saved configuration. Generate the source
index with `python -m Paradigm.tta_composition.run_catalog`; entries distinguish
copies, unversioned results and results matching the current adaptation code.
Historical runs without adaptation hashes cannot establish which code was used.
The index preserves the original files.

`verify_tta_improvements.py` runs all five single methods and the full
composition, writes a batch `verification_summary.json`, and returns a nonzero
exit code if any experiment fails. For a subset, use for example
`--methods DLTTA SmaRT VPTTA,DLTTA,TestFit,GraTa,SmaRT`. Mean improvement is
measured against the matching source-only cases; no positive gain is assumed.

Run the complete regression checks, including the function-based 32-subset
composition tests at 1/3/5 updates, with
`python -B -m Paradigm.tta_composition.tests.run_checks`. Headless Linux GUI
tests need a virtual display; use `xvfb-run -a` before this command.

Medical data and source weights are included through Git LFS. New experiment
directories and baseline caches remain ignored. After dependencies are installed
and LFS objects are restored, a fresh clone can open the GUI and start benchmark
tests without training a source model. `run_catalog` can index current
outputs without a sibling `test_v1`; historical-copy detection uses that sibling
only when it is available.

These are local adaptation variants inspired by the five named methods. Their
losses, parameter boundaries and prediction rules differ from the author
implementations; label results as local composition variants. The structural
region selection is a piecewise fragment-suppression heuristic, not an exact
differentiable topology constraint. It permits empty/single-region predictions
and does not invent missing anatomy. The requested
8-image runs are functional validation only and do not support performance
claims or hyperparameter tuning.

