# BUS-CoT

Breast ultrasound (BUS) with chain-of-thought (CoT) annotations. The release
ships approximately 4.5k expert-curated breast US frames, each accompanied by
one or more lesion masks, a structured ultrasound report (lesion edge,
boundary, calcification, echo, blood flow) and a free-text CoT reasoning
trace.

## Source & download

- Figshare release: https://doi.org/10.6084/m9.figshare.29036876.v1
- License: CC BY 4.0

## Expected on-disk layout

Unpack the Figshare archive so that the per-lesion folders sit directly
under this directory:

```
BUS-CoT/
├── BUS-Expert/
│   └── <iid>/
│       ├── <iid>@raw.png         # original ultrasound frame
│       └── <iid>@mask*.png       # one or more lesion masks
└── preprocess.py
```

## Processing

```bash
# 1. Mirror raw frames + masks into ./Processed/<iid>/
python preprocess.py mirror_images --input_dir ./BUS-Expert --output_dir ./Processed

# 2. (optional) Cache bounding boxes derived from masks
python preprocess.py compute_bboxes --processed_dir ./Processed --output_json bboxes.json
```

`Processed/` is what the BreastStage `RESIZED/` step in the parent
`dataset/README.md` expects to consume.

## Citation

```bibtex
@article{wu2025buscot,
  title   = {BUS-CoT: A chain-of-thought ultrasound dataset for breast lesion reasoning},
  author  = {BUS-CoT contributors},
  year    = {2025},
  doi     = {10.6084/m9.figshare.29036876.v1}
}
```
