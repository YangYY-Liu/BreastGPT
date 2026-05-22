# CT-RATE

CT-RATE is a large-scale chest CT dataset with paired radiology reports
(approximately 50k non-contrast and contrast chest volumes from 21k patients).
BreastGPT only uses the **non-contrast** subset whose field of view includes
breast tissue.

## Source & download

- HuggingFace dataset: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
- License: CC BY 4.0

You will need a HuggingFace account; follow the instructions on the dataset
page to clone via `git-lfs` or `huggingface-cli`.

## Expected on-disk layout

```
CT-RATE/
├── train_info.csv               # CT-RATE official manifest
├── dataset/                     # downloaded NIfTI tree (mirrors HF layout)
│   └── train/.../*.nii.gz
└── preprocess.py
```

## Processing

```bash
# Filter to non-contrast chest CTs that include breast tissue, then mirror
# the selected .nii.gz volumes into ./Processed/.
python preprocess.py \
  --manifest train_info.csv \
  --source_root . \
  --output_root ./Processed \
  --write_manifest filtered.csv
```

The MONAI-driven resize (longest in-plane edge to 384, 48 slices, int16) is
performed downstream by `scripts/swift_train_full_node.sh` and the global
`resize.py` in the parent `dataset/` folder — this script intentionally does
**not** resample voxels.

## Citation

```bibtex
@article{hamamci2024ctrate,
  title   = {A foundation model utilizing chest CT volumes and radiology reports for supervised-level zero-shot detection of abnormalities},
  author  = {Hamamci, Ibrahim Ethem and Er, Sezgin and Almas, Furkan and others},
  journal = {arXiv preprint arXiv:2403.17834},
  year    = {2024}
}
```
