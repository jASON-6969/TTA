# SZ-CXR source data

- Images: Shenzhen Hospital X-ray Set, mirrored in the Hugging Face dataset `Famatsu123/montgomery-shenzhen-tuberculosis-cxr` from the NLM/Open-i collection.
- Masks: `yoctoman/shcxr-lung-mask` on Kaggle, downloaded from `https://www.kaggle.com/api/v1/datasets/download/yoctoman/shcxr-lung-mask?datasetVersionNumber=1`.
- Mask license: CC BY-NC-SA 4.0. Publications must attribute the National Library of Medicine and the Computer Engineering Department, Faculty of Informatics and Computer Engineering, National Technical University of Ukraine, as requested by the mask dataset authors; cite the associated dataset papers.
- The source archive contains 662 images and 566 unique manually segmented masks. This experiment trains only on the 566 exact image-mask pairs; 96 unannotated images are excluded from supervised training.
- Image filename: `CHNCXR_####_0.png` or `CHNCXR_####_1.png`; mask filename: `<image-stem>_mask.png`.
