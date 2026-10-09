

git lfs install
git clone https://github.com/jASON-6969/TTA.git test_v2
Set-Location .\test_v2
git lfs pull

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

python launch_gui.py


# CXR Test-Time Adaptation Reproduction (test_v2)

This directory can be used as the root of an independent GitHub repository.
It contains the source-model pipeline and the five-method TTA composition
framework. A Git LFS clone includes code, tests, the complete local datasets,
the fixed source split and the trained source checkpoint. Install the Python
dependencies, then open the GUI and start testing with its default Montgomery
data and checkpoint. Dataset downloading and source-model retraining are optional.
New TTA outputs and caches are excluded by `.gitignore`; `data/` and `base_model/`
are included. No sibling `test_v1` or parent virtual environment is needed.

The experiment direction is **SZ-CXR -> Montgomery**. Source masks are used
only for supervised source training and source validation. Montgomery images
are passed to test-time adaptation without masks; Montgomery masks are loaded
only after adaptation for evaluation.

## Bundle layout

```text
test_v2/                        # repository root (its name can change)
  .github/workflows/            # CPU tests on Windows and Linux
  .gitignore
  .gitattributes                # Git LFS for medical PNGs and source weights
  requirements.txt             # pinned core runtime
  requirements-dev.txt         # runtime plus type checker
  scripts/check_reproduction.py
  launch_gui.py                 # validates bundled assets, then opens the GUI
  start_gui.cmd                 # Windows launcher using this folder's .venv
  data/
    raw/
      sz_cxr/
        images/                 # 662 source images
        masks/                  # 566 exact paired masks
      montgomery/
        images/                 # 138 target images
        left_mask/              # 138 official masks
        right_mask/             # 138 official masks
        merged_mask/            # 138 pixelwise unions used for evaluation
    splits/
      source_split_seed42.json  # reproducible split metadata and IDs
      source_train.txt
      source_validation.txt
  code/feature_alignment_2d_pilot/
    ...                         # runnable reproduction code
  base_model/                   # bundled source checkpoint and frozen results
  selected_methods/             # generated legacy TTA results; ignored
  composition_runs/             # generated composition/cache results; ignored
  Paradigm/tta_composition/     # runnable five-method framework and tests
  Paradigm/                     # additional classification/vendor snapshots
  reports/tta_validation.md     # previously recorded local validation
  verify_tta_improvements.py    # five single methods plus full combination
  manifest.json                 # annotated historical test_v1 inventory
```

The `Paradigm/` classification and local execution status are documented in
`Paradigm/README.md`.

## Data and split

The source archive contains 662 SZ-CXR images but only 566 exact image-mask
pairs. The remaining 96 images are retained for provenance completeness and
are excluded from supervised training because no matching mask is available.

The 566 paired IDs are shuffled with Python `random.Random(42)` and split with
`validation_fraction=0.2`, producing 453 training IDs and 113 validation IDs.
The generated ID lists in `data/splits/` should be used when auditing or
reproducing the split. The current code also records the same seed and fraction
in its protocol audit.

Montgomery contains 138 images and official left/right masks. `merged_mask/`
is a convenience copy of the pixelwise union; the runner can regenerate it
from `left_mask/` and `right_mask/` if it is absent.

### Provenance and limitations

- SZ-CXR images are mirrored from the NLM/Open-i collection through the
  `Famatsu123/montgomery-shenzhen-tuberculosis-cxr` dataset.
- SZ-CXR masks come from `yoctoman/shcxr-lung-mask` and are documented as
  CC BY-NC-SA 4.0. Keep the attribution in `data/raw/sz_cxr/SOURCE.md`.
- Montgomery is the NIH/NLM Montgomery County CXR Set with official left/right
  masks.
- The paper protocol describes 662 source masks, but the locally audited public
  mask archives provide 566 exact masks. This bundle therefore reproduces the
  local, paired-data protocol and does not claim numerical equality with the
  paper's complete-data results.
- The copied medical images and annotations remain subject to their original
  dataset terms. Do not redistribute them without checking those terms.

### Included data and optional restoration

`data/` and `base_model/` are part of this repository. Medical PNGs and the
source checkpoint are managed by Git LFS. The data occupies about 4.11 GiB;
the base-model directory occupies about 29.8 MiB. Install Git LFS before
cloning and run `git lfs pull` after clone if needed. The quickstart in the
Environment section uses the bundled files and does not require retraining.

<details>
<summary>Optional: restore datasets from their original archives</summary>

Download the Shenzhen and Montgomery image archives from the
[dataset mirror](https://huggingface.co/datasets/Famatsu123/montgomery-shenzhen-tuberculosis-cxr).
The example pins its repository revision rather than using a changing `main`.
Download version 1 of the [Shenzhen mask archive](https://www.kaggle.com/datasets/yoctoman/shcxr-lung-mask)
through Kaggle (sign in if required), placing it at
`downloads/shcxr-lung-mask.zip`. Keep the attribution in
[`data/SZ_CXR_SOURCE.md`](data/SZ_CXR_SOURCE.md).

After installing the environment below, run from the repository root in PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path .\downloads | Out-Null
$revision = '4f02780016fe097c1f0919b14909ef50f261d5d1'
$mirror = "https://huggingface.co/datasets/Famatsu123/montgomery-shenzhen-tuberculosis-cxr/resolve/$revision"
Invoke-WebRequest "$mirror/ChinaSet_AllFiles.zip" -OutFile .\downloads\ChinaSet_AllFiles.zip
Invoke-WebRequest "$mirror/MontgomerySet.zip" -OutFile .\downloads\MontgomerySet.zip
Expand-Archive .\downloads\ChinaSet_AllFiles.zip .\downloads\source
Expand-Archive .\downloads\MontgomerySet.zip .\downloads\target
Expand-Archive .\downloads\shcxr-lung-mask.zip .\downloads\source_masks

$sourceImages = (Get-ChildItem .\downloads\source -Directory -Recurse | Where-Object Name -eq 'CXR_png' | Select-Object -First 1).FullName
$targetImages = (Get-ChildItem .\downloads\target -Directory -Recurse | Where-Object Name -eq 'CXR_png' | Select-Object -First 1).FullName
$leftMasks = (Get-ChildItem .\downloads\target -Directory -Recurse | Where-Object Name -eq 'leftMask' | Select-Object -First 1).FullName
$rightMasks = (Get-ChildItem .\downloads\target -Directory -Recurse | Where-Object Name -eq 'rightMask' | Select-Object -First 1).FullName
$sourceMasks = (Get-ChildItem .\downloads\source_masks -Directory -Recurse | Where-Object Name -eq 'mask' | Select-Object -First 1).FullName
if (-not ($sourceImages -and $targetImages -and $leftMasks -and $rightMasks -and $sourceMasks)) {
    throw 'Expected archive directories were not found; inspect the extracted layout.'
}
New-Item -ItemType Directory -Force -Path .\data\raw\sz_cxr\images, .\data\raw\sz_cxr\masks, .\data\raw\montgomery\images, .\data\raw\montgomery\left_mask, .\data\raw\montgomery\right_mask | Out-Null
Copy-Item "$sourceImages\CHNCXR_*.png" .\data\raw\sz_cxr\images\
Copy-Item "$sourceMasks\CHNCXR_*_mask.png" .\data\raw\sz_cxr\masks\
Copy-Item "$targetImages\MCUCXR_*.png" .\data\raw\montgomery\images\
Copy-Item "$leftMasks\MCUCXR_*.png" .\data\raw\montgomery\left_mask\
Copy-Item "$rightMasks\MCUCXR_*.png" .\data\raw\montgomery\right_mask\
python scripts/check_reproduction.py --data
```

The check must report 662 source images, 566 paired masks, 453 training cases,
113 validation cases and 138 paired Montgomery cases, with the exact IDs and
order saved in `data/splits/`. It checks names, coverage and the split; it
does not establish image-byte identity with the original experiment. Source
masks must be named `<image-stem>_mask.png`; Montgomery masks must use the
same filename as their image. The merged target masks are generated by source
training; the local merged copies are also included in the LFS bundle.

</details>

## Environment

The recorded experiment used Python 3.12.10, PyTorch 2.11.0+cu128, CUDA 12.8,
NumPy 1.26.4, SciPy 1.17.1 and Pillow 12.3.0 on Windows. Use Python 3.12;
the requirements pin those four Python packages. CPU execution is supported.
The full dependency graph, GPU driver and hardware are not locked, and
retraining or switching CPU/GPU can change floating-point results.

Clone this repository and run the following commands in PowerShell:

```powershell
git lfs install
git clone https://github.com/jASON-6969/TTA.git test_v2
Set-Location .\test_v2
git lfs pull
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
$env:PYTHONPATH = (Resolve-Path .\code).Path
python scripts/check_reproduction.py --data --checkpoint
python launch_gui.py
```

The GUI defaults to the bundled checkpoint and Montgomery image/mask folders.
Choose methods and start the experiment; no source training is needed for
testing. The first run creates a complete frozen target baseline before TTA.
After setup, Windows users can reopen the GUI by double-clicking
`start_gui.cmd`. Both GUI and its worker use the interpreter that launched
the application, including a virtual environment located inside the clone.

Git LFS must be installed on the cloning machine and the repository's LFS
objects must have been uploaded. A GitHub ZIP or a clone made with LFS download
disabled can contain pointer files instead of images/weights. The launcher
checks for these and reports `git lfs pull` before opening the GUI.

If PowerShell blocks activation, run the commands with
`.\.venv\Scripts\python.exe` instead of `python`. For the recorded GPU setup,
replace the CPU-wheel install above with:

```powershell
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

On Linux/macOS, create the environment with `python3.12 -m venv .venv`,
activate it with `source .venv/bin/activate`, and use the same pip commands
(macOS uses its supported default PyTorch wheel). For the source-model and
legacy runners, set `export PYTHONPATH="$PWD/code"`. All commands below are
run from the repository root. Tkinter is needed only for the desktop GUI and
its GUI tests; on Linux install the distribution's Tk support and use a
virtual display for headless GUI tests.

## Test TTA and optionally retrain the source model

The installed GUI above and the TTA commands below use the bundled source
checkpoint directly. To retrain the source UNet instead, run these optional
commands; they replace the checkpoint and its source-training outputs:

```powershell
$env:PYTHONPATH = (Resolve-Path .\code).Path
python scripts/check_reproduction.py --data
python -m feature_alignment_2d_pilot.run_base_model
python scripts/check_reproduction.py --data --checkpoint
```

Training writes the selected source checkpoint to
`base_model/benchmark_cxr_base_no_source_aug/source_checkpoint.pth` and evaluates
the frozen Source-only model. An existing compatible checkpoint can be checked
with `--checkpoint PATH` and passed to the composition runner with the same
option. Using other weights changes the experiment. The recorded checkpoint
SHA-256 is `d08d23748329fdfaef5842c6257ee418b55d2ec8a083790c9c8b2eff3d67cfb2`;
it is included through Git LFS. A retraining run is not expected to recover
identical checkpoint bytes across every environment.

Run an eight-case functional check, then all five single methods and their
full combination on the 138-case target stream:

```powershell
python -m Paradigm.tta_composition.run --methods GraTa --max-cases 8 --seed 42 --output .\composition_runs\smoke_grata
python verify_tta_improvements.py --seed 42 --max-cases 138
```

The composition pipeline defaults to three updates per image, persistent
adaptation state and image-only target inputs. It first builds a frozen
baseline for all target images, even when `--max-cases` limits adaptation.
The verification script uses the default checkpoint path; for custom weights,
run individual method sets through the composition CLI with `--checkpoint`.
Every run must use a fresh output directory. `verify_tta_improvements.py`
creates a timestamped batch and exits with a nonzero status if any run fails.

Source-only and a custom combination can also be run separately:

```powershell
python -m Paradigm.tta_composition.run --max-cases 138 --seed 42 --output .\composition_runs\source_only
python -m Paradigm.tta_composition.run --methods VPTTA,DLTTA,TestFit,GraTa,SmaRT --max-cases 138 --seed 42 --output .\composition_runs\all_five
```

With `--methods` omitted and no config file, the runner selects Source-only.
The optional legacy pipeline uses different adapters/defaults and one update
per target image; it is not the five-method composition experiment:

```powershell
python -m feature_alignment_2d_pilot.run_selected_methods
```

Outputs are written under:

- `base_model/benchmark_cxr_base_no_source_aug/`
- `selected_methods/benchmark_cxr_methods_no_source_aug/`
- `composition_runs/verify_batch_<timestamp>_<unique>/verification_summary.json`
- `composition_runs/<run>/`: `config.json`, `provenance.json`, `audit.json`,
  `adaptation_log.jsonl`, `summary.json`, predictions and matched comparisons.
- `composition_runs/baselines/`: reusable full-target frozen predictions.

The source training protocol is: 256 x 256 grayscale input, five encoder
blocks, base width 32, Adam with learning rate 1e-3, batch size 2, maximum 25
epochs, early stopping patience 5, seed 42, per-image z-score normalization,
and no source augmentation. The checkpoint is selected by source validation
Dice. Target labels are evaluation-only and must not select hyperparameters.

## Test the bundle code

These checks use synthetic cases and do not require downloaded images or
source weights. Install `requirements-dev.txt` first:

```powershell
python -m pip install -r requirements-dev.txt
python scripts/check_reproduction.py
python -B -X utf8 -m unittest scripts.test_check_reproduction scripts.test_launch_gui -v
python -B -X utf8 -m Paradigm.tta_composition.tests.run_checks
python -m mypy --strict scripts/check_reproduction.py scripts/test_check_reproduction.py launch_gui.py scripts/test_launch_gui.py
python -m mypy --check-untyped-defs --follow-imports=skip --ignore-missing-imports verify_tta_improvements.py Paradigm/tta_composition/tests/test_verification.py
python -m mypy --check-untyped-defs --follow-imports=skip --ignore-missing-imports Paradigm/tta_composition/gui.py Paradigm/tta_composition/tests/test_gui_features.py
python -m compileall -q code Paradigm/tta_composition scripts launch_gui.py
```

The composition suite covers all 32 method subsets, 1/3/5 adaptation steps,
mask isolation, lifecycle callbacks, baseline reuse, GUI callbacks and failure
handling. On headless Linux, prefix the composition suite with `xvfb-run -a`.
The legacy `test_benchmark_cxr.py` additionally requires the real benchmark
data and pytest; it is not part of the data-free CI job.

Local preparation validation on 2026-10-08 installed the pinned CPU runtime
in a new environment and tested an isolated copy containing only Git-shareable
files. All 143 composition tests and 8 reproduction-tool tests passed; type
checks passed for the new tool/tests and modified batch runner/tests. A real
single-case CPU GraTa run completed baseline inference, three adaptation
updates and matched evaluation with the existing source checkpoint. The saved
full dataset/split and checkpoint architecture/hash were also checked. This
preparation did not retrain the source model or rerun the full six-method
benchmark. GitHub Actions is configured to run imports, type checks and these
synthetic regressions on Windows/Linux; its remote execution is not yet verified.
It does not download medical data or establish numerical equivalence with the
recorded GPU experiments.

The bundled-GUI follow-up on 2026-10-08 verified that all 1,929 files under
`data/` and `base_model/` remain eligible for Git, including the source weights.
Git LFS staged 1,781 image/weight objects and restored the real files into an
independent working tree. With its own Python 3.12 CPU environment, the GUI
found the default bundled checkpoint and Montgomery folders and completed an
eight-case GraTa run (24 updates) after a 138-case frozen baseline. All 144
composition and 12 startup/check regressions passed, including a separate
pointer-only checkout matching CI's `lfs: false` setting. Strict type checks
passed for the four startup/check files; the GUI and its regression file also
passed `--check-untyped-defs --follow-imports=skip --ignore-missing-imports`.
This was a local LFS checkout test, not a remote GitHub clone or upload.

## Historical legacy single-seed results

These values are copied legacy pipeline results, not results from the current
three-step composition defaults. They use 138 Montgomery cases. HD95 is
measured in pixels on 256 x 256 resized masks.

| Method | Dice mean | HD95 mean | Cases |
| --- | ---: | ---: | ---: |
| Source-only | 0.9396 | 2.6404 | 138 |
| TENT | 0.9501 | 1.8278 | 138 |
| TestFit | 0.9436 | 2.3571 | 138 |
| GraTa | 0.9486 | 1.9520 | 138 |

These are local single-seed reproduction results. TestFit and GraTa are local
protocol adapters, not claims of line-by-line reproduction of the official
upstream wrappers.

The previously recorded current-code five-method comparison is documented in
[`reports/tta_validation.md`](reports/tta_validation.md). In that matched,
single-seed batch, GraTa's Dice was 0.9414993318 and the full combination's Dice
was 0.9408018767, versus Source-only 0.9395699204. Individual cases can regress;
these results are local adaptation variants, not author-method reproductions.
The referenced raw run directories are local generated artifacts and are
excluded from GitHub. Recreate them with the commands above.

## GitHub contents and historical utilities

Publish the contents of `test_v2` as the repository root so that
`.github/workflows/reproduction.yml` is discovered by GitHub. The ignore file
keeps new TTA output/cache directories, temporary downloaded archives,
credentials, virtual environments and local PDF/PPT exports out of new Git
additions. It includes all files under `data/` and `base_model/`, as well as code,
source attribution, split IDs and Markdown validation records. `.gitattributes`
uses Git LFS for `data/raw/**/*.png` and `base_model/**/*.pth` so cloning can
restore the real assets. Upload the LFS objects together with the Git source;
the repository must have enough LFS storage and download bandwidth for this
bundle. See [GitHub's LFS documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).
Ignore rules do not remove files already tracked by Git.

`manifest.json` preserves the original 2,585-entry `test_v1` inventory and is
explicitly marked historical. Its hashes describe that old snapshot, not the
current GitHub source tree. Current experiment code identity is saved by the
runner in `provenance.json` and `audit.json`. The old PDF builder under `reports/`
expects a legacy `output/experiments` layout and is outside the supported
clone/GUI/composition workflow. Reference method folders retain upstream
provenance; their presence does not make every vendor wrapper runnable or
assign a new license to third-party code.

## Method classification

See [`Paradigm/README.md`](Paradigm/README.md). In brief:

- Feature-level Alignment: DANN, GraTa, TestFit, UDA-MIMA
- Input-level Translation: AIF-SFDA, DL-TTA, RSA, SFDA-FSM, STDR
- Output-level Regularization: DeTTA, DG-TTA, SaTTCA, SmaRT, TENT, UPL-SFDA
- Prior Estimation: AdaMI, ExploringTTA, PASS, ProSFDA, VPTTA
- Baseline: Source-only (documented in `Paradigm/Baseline/`)

The method folders are small vendor source snapshots copied from the local
`vendor/MedSeg-TTA` registry and retained for classification and attribution.
# TTA

