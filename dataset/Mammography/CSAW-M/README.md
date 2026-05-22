# CSAW-M

CSAW-M is an ordinal mammographic-masking dataset built from the Cohort of
Screen-Aged Women: roughly 10k mammograms (preprocessed PNGs) annotated by
expert radiologists on an 8-level ordinal masking-of-cancer scale, plus
interval-cancer / "if cancer" flags.

## Source & download

- Figshare (SciLifeLab): https://figshare.scilifelab.se/articles/dataset/CSAW-M_An_Ordinal_Classification_Dataset_for_Benchmarking_Mammographic_Masking_of_Cancer/14687271
- License: CSAW DUA (request via the project page)

## Expected on-disk layout

```
CSAW-M/
├── images/preprocessed/<study_prefix>/<filename>.png
├── labels/
│   ├── CSAW-M_train.csv
│   └── CSAW-M_test.csv
└── preprocess.py
```

## Processing

```bash
python preprocess.py
```

For every image listed in the two label CSVs the script applies
`extract_breast` (background suppression + breast-region crop) and writes a
PNG into `./Processed/<basename>.png`.

## Citation

```bibtex
@inproceedings{sorkhei2021csawm,
  title     = {CSAW-M: An Ordinal Classification Dataset for Benchmarking Mammographic Masking of Cancer},
  author    = {Sorkhei, Moein and Liu, Yue and Azizpour, Hossein and others},
  booktitle = {NeurIPS Datasets and Benchmarks},
  year      = {2021}
}
```
