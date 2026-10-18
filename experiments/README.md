# Phase classifier: real vs. synthetic data experiment

Trains a 4-class mitosis-phase classifier (`prometaphase`, `metaphase`,
`ring_shape`, `anaphase_telophase`) under four data conditions --
real-only, real+synthetic (MitoSim-simulated masks), real+synthetic (real
masks), and synthetic-only -- to measure the value of MitoSim-generated
training data, and separately whether that value comes from the GAN's
photorealistic context-recombination alone or specifically from MitoSim's
ability to generate mask *shapes* beyond the small real training set.
Test set is always the same held-out real data for all four conditions,
so results are directly comparable.

## Why 4 classes, not 5

The expert-rated ground truth (`Atypical_MIDOG_TUPAC.csv`)
has no real-world "prophase" category at all -- pathologists don't
operationally count prophase nuclei as mitotic figures -- and never
distinguishes anaphase from telophase (always "anaphase-telophase"). The
classifier's classes follow the real labels directly: `prometaphase`,
`metaphase`, `ring_shape` (a real, separately-labelled category -- a
plate viewed near face-on can present as a ring), and `anaphase_telophase`
(merged, matching the real label exactly). There is no synthetic-only
"prophase" class in this experiment.

## Data

- **Filtering**: real rows kept only where both experts agree on the
  label AND neither calls it atypical (`data/extract_real_patches.py`).
  2324 of 3720 total rows qualify. `ring_shape` is the rarest class by a
  wide margin (34 total, 23 in the training split) -- itself part of the
  motivation for the synthetic-augmentation experiment.
- **Patches**: extracted directly from the local MIDOG21/TUPAC16 whole-
  slide images, 128x128 at level 0 centred on (x, y) -- the same
  convention as `AtypicalMF_Dataset/ExtractAtypical.ipynb`.
- **Split**: grouped by `(dataset, slide)` -- no slide ever straddles
  train/val/test, since slide-level staining/scanner/tissue
  characteristics would otherwise let a classifier learn a same-slide
  shortcut instead of genuinely distinguishing phases (`data/build_splits.py`,
  `StratifiedGroupKFold`: ~20% test, then ~15% of the remainder as val).
  Test is fixed and real-only, shared by all four conditions.
- **Synthetic data**: `data/generate_synthetic.py` generates 5x the real
  training-set size, split evenly across the 4 classes (not matched to
  real's imbalanced proportions -- the point is to shore up what real
  data under-represents). `ring_shape` and `metaphase` come from a single
  rejection-sampling process gated on hollowness
  (`mitosim/score/descriptors.py`'s `hollowness`, checked directly on the
  actual rendered mask, independent of connected-component count since
  rings often fragment into several arcs under blur): metaphase sampled
  at a broad tilt range and kept when the rendered mask's hollowness
  stays below threshold;
  ring_shape sampled at tilt concentrated near face-on (70-90deg,
  pronounced-hollow params) and kept when it clears that same threshold.
  Every numeric `PhaseParams` field is jittered +/-15% per sample. A
  separate, smaller synthetic validation set is generated at the same
  ratio real_val bears to real_train.
- **Mask-source ablation** (`real_synth_realmask`): same GAN, same 5x
  scale, but masks are drawn with replacement from the real training
  patches' own masks instead of MitoSim-simulated ones
  (`data/generate_realmask_ablation.py`) -- isolates whether the benefit
  comes from the GAN's photorealistic context-recombination alone, or
  specifically from MitoSim's ability to generate mask shapes beyond what
  the small real training set has.

Run once, in order:

```bash
cd experiments/data
python extract_real_patches.py
python build_splits.py
python generate_synthetic.py        # slow (~1.5-2h locally): ring_shape's
                                     # rejection-sampling yield is only ~3-10%
python gan_render_synthetic.py      # GPU; turns the masks above into images
python generate_realmask_ablation.py  # GPU; the real_synth_realmask condition
```

## Training

`training/train.py` runs one (condition, seed) pair: `timm` EfficientNet-B0
(pretrained, 4-class head), standard augmentation (flips/rotation/affine/
brightness-contrast/blur/noise) + albumentations' native `A.HEStain`
(needs albumentations>=2.0), class-weighted cross-entropy (real data is
heavily imbalanced), early stopping on validation loss (patience 5),
logged to Weights & Biases. Device auto-picks CUDA, then MPS, then CPU, so
the same script runs unmodified on a Mac or on the cluster.

```bash
cd experiments/training
python train.py --condition real --seed 0
```

### Full experiment (5 repeats x 4 conditions = 20 runs)

Locally (sequential -- one GPU/MPS device):

```bash
cd experiments
python run_experiment.py --seeds 5
python aggregate_results.py
```

On the HPC cluster (20 separate SLURM jobs, run concurrently): see below.

## HPC (hpc3)

```bash
# one-time: copy code + generated data, bootstrap the environment
bash hpc/sync_to_hpc.sh
ssh hpc 'bash -s' < hpc/env_setup.sh
ssh hpc 'cd ~/code/MitosSim/experiments && wandb login'   # or export WANDB_API_KEY

# submit all 20 runs
ssh hpc 'cd ~/code/MitosSim/experiments && bash hpc/submit_all.sh'

# once done, pull results back and aggregate locally
rsync -avz hpc:~/code/MitosSim/experiments/results/ results/
python aggregate_results.py
```

`hpc/submit.sbatch` follows the cluster's own template (`research`
partition, `gres=shard:1` -- a 24GB VRAM share, plenty for EfficientNet-B0
at 128x128). Time limit is set to 8h (generous headroom over the original
4h template, since `real_synth` has ~6x real-only's data) -- tune down
once you've seen an actual run's wall-clock time, which `train.py`'s
result JSON records (`wall_time_s`).

## Design choices made without an explicit spec (easy to change)

- Synthetic data generated in even 1/4 splits per class, not matched to
  real's class proportions.
- wandb project name: `mitosim-phase-classifier`.
- 80/20 real train+val/test split, 85/15 train/val within that.
- Class-weighted cross-entropy loss (inverse frequency), computed per
  condition's own training set.
