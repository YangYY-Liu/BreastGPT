# DMID

Digital Mammography Image Database for breast cancer diagnosis research.
510 mammograms (DICOM + 16-bit TIFF) with masses, calcifications,
architectural distortion and asymmetry annotations, plus pixel-level masks
and per-image radiology reports.

## Source & download

- Figshare release: https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883
- License: research / educational use only (per the Figshare deposit)

## Expected on-disk layout

```
DMID/
├── Metadata.csv                  # per-lesion annotations (image_id, x_center, y_center, radius, ...)
├── DICOM Images/*.dcm
├── ROI Masks/*.tif               # pixel-level lesion masks
├── Reports/<image_id>.txt        # radiology report per image
└── preprocess.py
```

## Processing

```bash
# Step 1 (optional) - convert the raw TIFF / DICOM dumps to PNG under Original/
python -c "from preprocess import tif_to_png; tif_to_png('DICOM Images','image'); tif_to_png('ROI Masks','mask')"

# Step 2 - breast crop + per-lesion mask + finding patch
python preprocess.py
```

Outputs:

```
Processed/
├── image/<image_id>.png
├── mask/<image_id>_<mask_id>.png
└── finding/<image_id>_<mask_id>.png
```

## Citation

```bibtex
@article{khoulqi2024dmid,
  title   = {Digital mammography dataset for breast cancer diagnosis research (DMID) with breast mass segmentation analysis},
  author  = {Khoulqi, Ilyasse and Idrissi, Najlae},
  journal = {Biomedical Engineering Letters},
  year    = {2024},
  doi     = {10.1007/s13534-023-00339-y}
}
```
