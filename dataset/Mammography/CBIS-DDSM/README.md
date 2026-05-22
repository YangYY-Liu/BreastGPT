# CBIS-DDSM

Curated Breast Imaging Subset of DDSM (CBIS-DDSM). Approximately 3k digitised
screen-film mammograms with mass / calcification ROIs, BI-RADS assessments
and biopsy-proven pathology.

## Source & download

- TCIA collection: https://www.cancerimagingarchive.net/collection/cbis-ddsm/
- License: TCIA Data Usage Policy (open access; cite per TCIA guidance)

Use the NBIA Data Retriever or `aws s3 sync` (as documented on the TCIA page)
to fetch the DICOMs and the four case-description CSVs.

## Expected on-disk layout

```
CBIS-DDSM/
├── csv/
│   ├── dicom_info.csv
│   ├── mass_case_description_{train,test}_set.csv
│   └── calc_case_description_{train,test}_set.csv
├── <UID-tree of DICOM files extracted by the NBIA tool>
└── preprocess.py
```

## Processing

```bash
python preprocess.py
```

Outputs (multi-process):

```
Processed/
├── breast/<patient_id>.png   # full breast crop
├── finding/<patient_id>.png  # tightest crop around the lesion mask
└── mask/<patient_id>.png     # binary lesion mask aligned to breast crop
```

The script also auto-detects and corrects the well-known CBIS-DDSM
"cropped vs mask file path swap" bug.

## Citation

```bibtex
@article{lee2017cbisddsm,
  title   = {A curated mammography data set for use in computer-aided detection and diagnosis research},
  author  = {Lee, Rebecca Sawyer and Gimenez, Francisco and Hoogi, Assaf and others},
  journal = {Scientific Data},
  volume  = {4},
  number  = {1},
  pages   = {170177},
  year    = {2017}
}
```
