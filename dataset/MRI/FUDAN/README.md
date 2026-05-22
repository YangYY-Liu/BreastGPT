# FUDAN (clinical breast MRI cohort)

A multi-modality breast MRI cohort (T1, T1w, T1dyn, T2w, DWI, ADC) collected
at Fudan University Shanghai Cancer Center. Each study has paired Chinese
radiology reports and biopsy-confirmed pathology.

## Source & download

The raw MRI volumes are **not publicly available** — they were collected
under a clinical Data Use Agreement and contain protected health
information. To request access, contact the BreastGPT team via the
ModelScope dataset page:

- ModelScope (BreastStage): https://www.modelscope.cn/datasets/YYangYang/BreastStage

You will be asked to provide an institutional review letter and sign a
research DUA.

## Expected on-disk layout

```
FUDAN/
├── fdzl_meta.csv                       # per-patient modality manifest
├── <source raw tree>                   # referenced by the meta CSV's "root" column
└── preprocess.py
```

## Processing

The script exposes four sub-commands (via `fire`) corresponding to the four
stages of the pipeline:

```bash
# 1. Collect raw volumes per patient under images/<pid>/
python preprocess.py cp_data --meta_csv fdzl_meta.csv --out_dir images

# 2. Reformat to canonical RAS / int16 with MONAI
python preprocess.py monai_preprocess --input_dir images --output_dir images_format

# 3. Foreground-crop every modality to a shared ROI
python preprocess.py rename_crop --src_root images_rename --dst_root images_crop

# 4. Rigid registration of every modality to T1w with ANTs
python preprocess.py rename_registration --src_root images_rename --dst_root images_reg
```

## Citation

If you use this cohort in a paper, please cite the BreastGPT paper (TBD)
and acknowledge the Fudan University Shanghai Cancer Center.
