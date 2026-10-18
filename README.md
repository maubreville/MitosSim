# MitosSim

MitosSim is a parametric framework that can be used to create realistically looking mitotic figures in H&E tissue. The framework first creates a 3D model of the chromatids in mitosis, then projects this model into 2D space creating a mask, and finally transforms the mask into a tissue context in H&E.


<img width="1280" height="527" alt="Figure1" src="https://github.com/user-attachments/assets/299b3243-03b0-46cc-a691-6ce67c1f0ada" />

## Data

We use the Ami-Br dataset (Bertram et al., Proc. BVM 2025) for all experiments and for training the context encoder GAN model. Furthermore, we use chromosome masks generated from the Ami-Br context images themselves as the context-encoder GAN's training targets.

## Install

Run the following scripts to install a venv for MitosSim:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Core simulation pipeline

The 3D-to-mask pipeline lives in `mitosim/` as a library, not a standalone CLI -- see it in use directly in the notebooks under [Figures](#figures)
below, or in `experiments/data/generate_synthetic.py` (the phase-classifier experiment's own dataset generator). The short version:

```python
from mitosim.phases import metaphase
from mitosim.phases.base import PhaseParams
from mitosim.render.optics import OpticsParams, render_chromatin_mask
import numpy as np

rng = np.random.default_rng(0)
state = metaphase.configure(rng, PhaseParams())
mask = render_chromatin_mask(state, s=0.0, t=3.0, params=OpticsParams())
```

`render_chromatin_mask` voxelises the chromatids, integrates a defocus-dependent Beer-Lambert path length along the slicing axis, and thresholds it to a binary chromatin mask -- the structural input the GAN context encoder paints a realistic image from. 

## Context-encoder GAN

`gan_context_encoder.py` turns a chromatin mask into a realistic H&E image. It erases an ellipse-shaped hole (sized to the mask) from a real context photo and inpaints it, conditioned on the chromatin mask, with a UNet generator trained adversarially 
against a PatchGAN discriminator (Isola et al. 2017) plus an L1 term on the untouched context region - the same recipe as Pathak et al.'s original context encoder.

```bash
# train from scratch against a real image/mask corpus (see weights/README.md for layout)
.venv/bin/python gan_context_encoder.py --mode train --images-dir weights/context_images --masks-dir weights/context_masks

# generate one image from a trained checkpoint
.venv/bin/python gan_context_encoder.py --mode generate --checkpoint weights/checkpoint_best.pt --images-dir weights/context_images --masks-dir weights/context_masks
```

`mitosim/render/match_context.py` is what picks which real context photo to condition on for a given synthetic mask, matched by size/orientation category (uniformly at random within the category, for background variety). 

## Phase-classifier experiment

`experiments/` contains the paper's downstream ML experiment: training a 4-class mitosis-phase classifier on real microscopy patches, GAN-synthetic patches (from both MitoSim-simulated and real masks), and mixtures of the two, to measure how much 
value the synthetic data adds over real data alone. The 64 run-result JSON files backing the paper's numbers are included under `experiments/results/`; see [experiments/README.md](experiments/README.md) for the full setup and the generation/training scripts used to produce them.

## Human Expert experiment

We carried out a human rater experiment, see [notebooks/human_eval_200.ipynb](notebooks/human_eval_200.ipynb).

## Figures

These notebooks produce the paper's qualitative figures and stimulus sets. All need the `notebooks` extra (`.venv/bin/pip install -e ".[notebooks]"`):

- [notebooks/chromatid_model_figure.ipynb](notebooks/chromatid_model_figure.ipynb)
  -- a single chromatid's swept-tube geometry (Figure 2a)
- [notebooks/five_phases_figure.ipynb](notebooks/five_phases_figure.ipynb)
  -- one representative rendered cell per mitotic phase, all 23
  chromosome pairs, side by side (Figure 2b).
- [notebooks/synthetic_examples_figure.ipynb](notebooks/synthetic_examples_figure.ipynb)
  -- a qualitative grid of example synthetic training images, sampled from
  the phase-classifier experiment's own 5x synthetic training set.

