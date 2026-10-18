# Weights

This directory ships `checkpoint_best.pt`, the trained GAN context-encoder checkpoint loaded by `gan_context_encoder.py` and
`experiments/data/gan_render_synthetic.py`.

Full inference also needs a context-image corpus: real H&E patches and their chromatin masks that the renderer composites simulated mitotic figures into. That corpus is not redistributed here, since it is derived from the public MIDOG21 and TUPAC16 challenge datasets.

How to use:

Place your own `context_images/`, `context_masks/`, and `context_ellipse_db.csv` alongside `checkpoint_best.pt` in this directory
(or in the directory pointed to by `MITOSIM_WEIGHTS_DIR`) to enable them.
