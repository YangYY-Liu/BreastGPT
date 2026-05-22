# ZHE2 (clinical breast MRI cohort)

A second multi-modality breast MRI cohort (T1, T1dyn, ADC, DWI plus
optional T2w) with biopsy-confirmed lesion masks (label 1 = breast,
2 = malignant, 3 = benign) and bilingual radiology reports.

## Source & download

The ZHE2 volumes are **not publicly available** — they were collected
under a clinical Data Use Agreement. To request access, contact the
BreastGPT team via the ModelScope dataset page:

- ModelScope (BreastStage): https://www.modelscope.cn/datasets/YYangYang/BreastStage

## Expected on-disk layout

```
ZHE2/
├── images/<pid>/                       # raw nii.gz files (one per modality)
│   ├── T1.nii.gz
│   ├── T1dyn.nii.gz
│   ├── ADC.nii.gz
│   ├── DWI.nii.gz
│   └── Mask.nii.gz                     # 0/1/2/3 label map (see above)
└── preprocess.py
```

## Processing

```bash
python preprocess.py monai_preprocess --input_dir images --output_dir images_format
```

Output: `images_format/<pid>/<modality>.nii.gz`, each volume re-oriented to
RAS, resampled to a uniform (0.725, 0.725, 2) mm grid and cast to int16.

## Citation

If you use this cohort in a paper, please cite the BreastGPT paper (TBD)
and acknowledge the collaborating clinical institution.
