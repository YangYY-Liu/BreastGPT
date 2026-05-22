# VinDr-Mammo

VinDr-Mammo is a large-scale full-field digital mammography benchmark:
5,000 four-view exams (~20k images) with breast-level BI-RADS / density
assessments and per-lesion bounding-box annotations.

## Source & download

- PhysioNet: https://physionet.org/content/vindr-mammo/1.0.0/
- License: PhysioNet Credentialed Health Data License 1.5.0
  (you must complete CITI training and sign the DUA via PhysioNet)

## Expected on-disk layout

```
VinDr/
├── metadata.csv
├── breast-level_annotations.csv
├── finding_annotations.csv
├── images/<study_id>/<image_id>.dicom
└── preprocess.py
```

## Processing

```bash
python preprocess.py
```

The script groups annotations by (patient, exam) and (in parallel) for each
image:

1. Reads the DICOM, applies the VOI LUT, inverts `MONOCHROME1`.
2. Crops to the breast region while simultaneously cropping all lesion bboxes.
3. Writes `Processed/breast/<image_id>.png`, per-lesion crops to
   `Processed/finding/<image_id>_<i>.png` and a tissue-only "normal" patch
   of matching size to `Processed/normal/<image_id>_<i>.png` for contrastive
   training.

Progress is streamed to `processed.csv` so the run can be resumed.

## Citation

```bibtex
@article{nguyen2023vindrmammo,
  title   = {VinDr-Mammo: A large-scale benchmark dataset for computer-aided detection and diagnosis in full-field digital mammography},
  author  = {Nguyen, Hieu T. and Nguyen, Ha Q. and Pham, Hieu H. and others},
  journal = {Scientific Data},
  volume  = {10},
  pages   = {277},
  year    = {2023},
  doi     = {10.1038/s41597-023-02100-7}
}
```
