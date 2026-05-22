# Histopathology (WSI sources)

BreastGPT's Histopathology branch is trained on three public whole-slide
imaging (WSI) sources. All three are processed by the same Trident-based
patch / feature pipeline (`run_batch_of_slides.py`, `run_single_slide.py`,
`preprocess.py`).

| Source | Slides | URL | License |
| --- | --- | --- | --- |
| TCGA-BRCA | ~1.1k diagnostic FFPE slides | https://portal.gdc.cancer.gov/ (project `TCGA-BRCA`, browse biospecimen -> slide image) | Open access |
| BCNB | ~1k early breast cancer slides | https://bcnb.grand-challenge.org/ and https://bupt-ai-cz.github.io/BCNB/ | Research use |
| HISTAI-breast | breast subset of HISTAI | https://huggingface.co/datasets/histai (subset `histai/HISTAI-breast`) | CC BY-NC 4.0 |

## Expected on-disk layout

```
Histopathology/
├── TCGA-BRCA/WSI/<uuid>.svs
├── BCNB/WSI/<id>.jpg
├── HISTAI-breast/WSI/<id>.tiff
├── preprocess.py
├── run_batch_of_slides.py        # Trident batch driver
└── run_single_slide.py           # Trident per-slide driver
```

## Processing

Patches are tiled with [Trident](https://github.com/mahmoodlab/trident) at
20x magnification, 512x512, 0 overlap; tissue is segmented with `hest`;
features are extracted with the CONCH v1.5 patch encoder.

```bash
# Per source (parallel, GPU=0)
python preprocess.py process_TCGA   --image_dir ./TCGA-BRCA/WSI       --job_dir ./TCGA-BRCA/Processed
python preprocess.py process_bcnb   --image_dir ./BCNB/WSI            --job_dir ./BCNB/Processed
python preprocess.py process_histai --image_dir ./HISTAI-breast/WSI   --job_dir ./HISTAI-breast/Processed
```

Each call writes the standard Trident layout:

```
<source>/Processed/20x_512px_0px_overlap/
├── patches/<slide>_patches.h5            # coords
└── features_conch_v15/<slide>.h5         # CONCH features (used by BreastGPT's GigaPixel branch)
```

To also dump a representative subset of patches as PNGs (e.g. for `resize.py`
downstream):

```bash
python preprocess.py extract_patch_from_trident --basedir . --sample_num 30
```

This writes `<source>/Processed/20x_512px_0px_overlap/images_30/<slide>/<i>.png`.

For one-off slides you can call the per-slide driver directly:

```bash
python run_single_slide.py --slide_path path/to/slide.svs --job_dir ./out --mag 20 --patch_size 512
```

## Citation

```bibtex
@article{zhang2025trident,
  title   = {Accelerating Data Processing and Benchmarking of AI Models for Pathology},
  author  = {Zhang, Andrew and others},
  journal = {arXiv preprint arXiv:2502.06750},
  year    = {2025}
}

@inproceedings{xu2024conch,
  title     = {A vision-language foundation model for computational pathology},
  author    = {Lu, Ming Y. and Chen, Bowen and Williamson, Drew F. K. and others},
  booktitle = {Nature Medicine},
  year      = {2024}
}

@inproceedings{xu2021bcnb,
  title     = {Predicting axillary lymph node metastasis in early breast cancer using deep learning on primary tumor biopsy slides},
  author    = {Xu, Feng and Zhu, Chuang and Tang, Wenqi and others},
  booktitle = {Frontiers in Oncology},
  year      = {2021}
}
```
