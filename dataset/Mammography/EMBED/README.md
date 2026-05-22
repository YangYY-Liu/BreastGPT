# EMBED

The Emory Breast Imaging Dataset (EMBED) — approximately 3.4M screening and
diagnostic mammographic images from 110k patients, with linked clinical and
pathological outcomes.

## Source & download

- AWS Open Data registry: https://registry.opendata.aws/emory-breast-imaging-dataset-embed/
- Project / tooling: https://github.com/Emory-HITI/EMBED_Open_Data
- License: EMBED Data Use Agreement (request via the Emory team)

## Expected on-disk layout

```
EMBED/
├── tables/
│   ├── EMBED_OpenData_metadata.csv      # per-image DICOM metadata (incl. ROI_coords)
│   └── EMBED_OpenData_clinical.csv      # per-exam clinical / pathology (not used by this script)
├── images/.../<dicom_path>.dcm          # the raw DICOM tree referenced from anon_dicom_path
└── preprocess.py
```

## Processing

```bash
python preprocess.py
```

For every row of the metadata table the script (parallelised with `mpire`):

1. Reads the DICOM, applies the VOI LUT and inverts `MONOCHROME1`.
2. Converts each ROI from EMBED's `(ymin, xmin, ymax, xmax)` order into an
   aligned mask.
3. Crops to the breast region and saves:

```
Processed/
├── image/<patient_id>_<exam_id>/<unique_id>.png
└── finding/<patient_id>_<exam_id>/<unique_id>_<i>.png
```

The image index is streamed to `processed.csv` so the run can be resumed.

## Citation

```bibtex
@article{jeong2023embed,
  title   = {The EMory BrEast imaging Dataset (EMBED): A racially diverse, granular dataset of 3.4 million screening and diagnostic mammographic images},
  author  = {Jeong, Jiwoong Jason and Vey, Brianna L. and Bhimireddy, Ananth and others},
  journal = {Radiology: Artificial Intelligence},
  year    = {2023},
  doi     = {10.1148/ryai.220047}
}
```
