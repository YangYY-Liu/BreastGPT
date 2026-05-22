# RSNA Screening Mammography Breast Cancer Detection

The RSNA Screening Mammography Breast Cancer Detection challenge: roughly
55k mammographic exams (54k from Australia, 8k from the United States) with
expert annotations and follow-up pathology.

## Source & download

- Kaggle competition: https://www.kaggle.com/competitions/rsna-breast-cancer-detection
- AWS Open Data mirror: https://registry.opendata.aws/rsna-screening-mammography-breast-cancer-detection/
- License: competition rules (research use; redistribution restricted)

## Expected on-disk layout

```
RSNA/
├── train.csv                            # official label table
├── train_images/<patient_id>/<image_id>.dcm
└── preprocess.py
```

## Processing

```bash
python preprocess.py
```

For every (patient_id, image_id) the script reads the DICOM, applies the
VOI LUT, inverts `MONOCHROME1`, strips a 10-pixel border, and writes
`./Processed/<exam_id>_<image_id>.png`. Progress is streamed to
`processed.csv` so the run can be resumed.

## Citation

```bibtex
@misc{carr2023rsnamammo,
  title  = {RSNA Screening Mammography Breast Cancer Detection},
  author = {Carr, Chris and Kitamura, Felipe and Partridge, Greg and others},
  year   = {2023},
  howpublished = {Kaggle competition},
  url    = {https://www.kaggle.com/competitions/rsna-breast-cancer-detection}
}
```
