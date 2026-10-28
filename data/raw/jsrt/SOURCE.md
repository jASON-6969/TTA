# JSRT target dataset

This folder contains a local 247-image JSRT lung-segmentation mirror and the
corresponding binary lung-field masks used as an optional target-only test set.

- Dataset source: Japanese Society of Radiological Technology (JSRT),
  <http://db.jsrt.or.jp/eng.php>
- Local mirror used to prepare the files:
  <https://www.kaggle.com/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset>
- Local layout: `images/<case>.png` and `masks/<case>.png`
- Coverage: 247 images and 247 masks, with matching case stems
- Image size: 224 x 224 pixels before the runner's 256 x 256 preprocessing
- The source masks use multiple grayscale label values. The copied masks were
  normalized to binary `0/255` by retaining source value `255` (lung field)
  only; heart and outside-lung labels are excluded.
- The local source archive SHA-256 is
  `22644FC10F98F86C25A6655255C3758599676D415845F3E290CC07CC582EAA27`.
- Use in this project: target inference and post-adaptation evaluation only;
  these images and masks must not be added to source-model training

The dataset and annotations remain subject to the terms of the original JSRT
distribution and the mask release used to prepare this folder. Check those
terms before redistribution.
