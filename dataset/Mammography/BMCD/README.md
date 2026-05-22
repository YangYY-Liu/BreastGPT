# BMCD

The Breast Mammography Cancer Dataset (BMCD) is a digital mammography
collection (~100 patients, both normal and suspicious cases) with per-case
BI-RADS density / classification and biopsy-derived pathology.

## Source & download

- Zenodo record: https://zenodo.org/records/5036062
- License: CC BY 4.0

## Expected on-disk layout

```
BMCD/
├── Description.xlsx
├── Normal_cases/<folder>/*.dcm
├── Suspicious_cases/<folder>/*.dcm
└── preprocess.py
```

## Processing

```bash
python preprocess.py
```

For every DICOM the script reads the pixel array, crops to the breast
region (`ExtractBreast`), normalises to uint8 and writes a PNG into
`./Processed/<case_type>_<folder>_<basename>.png`.

## Citation

```bibtex
@dataset{loizidou2021bmcd,
  title     = {{BMCD}: A Breast Mammography Cancer Dataset},
  author    = {Loizidou, Kosmia and Skouroumouni, Galateia and Pitris, Costas and others},
  year      = {2021},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.5036062}
}
```
