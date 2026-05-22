# CDD-CESM

Categorized Digital Database for Low Energy and Subtracted Contrast Enhanced
Spectral Mammography (CDD-CESM). 2,006 CESM images (low-energy + subtracted)
from 326 patients with radiologist annotations, segmentation masks and
biopsy results.

## Source & download

- TCIA collection: https://www.cancerimagingarchive.net/collection/cdd-cesm/
- License: TCIA Data Usage Policy (open access)

## Expected on-disk layout

```
CDD-CESM/
├── Radiology-manual-annotations.xlsx
├── Low energy images of CDD-CESM/*.jpg
├── input_image_new_v2/*.jpg            # alternate naming in some releases
└── preprocess.py
```

## Processing

```bash
python preprocess.py
```

For each "DM" (digital mammography / low-energy) image referenced in the
spreadsheet, the script reads the JPEG, crops to the breast region, applies
min-max normalisation, and writes a PNG into `./Processed/`.

## Citation

```bibtex
@article{khaled2022cddcesm,
  title   = {Categorized contrast enhanced mammography dataset for diagnostic and artificial intelligence research},
  author  = {Khaled, Rana and Helal, Maha and Alfarghaly, Omar and others},
  journal = {Scientific Data},
  volume  = {9},
  pages   = {122},
  year    = {2022},
  doi     = {10.1038/s41597-022-01238-0}
}
```
