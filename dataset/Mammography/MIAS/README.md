# mini-MIAS

The Mammographic Image Analysis Society (MIAS) database — 322 digitised
mammograms (1024x1024, 200 micron pixel edge, PGM format) with radiologist
"truth" markings of abnormality location and radius.

## Source & download

- PEIPA (University of Essex) mirror: http://peipa.essex.ac.uk/info/mias.html
- Cambridge repository (v1.21): https://www.repository.cam.ac.uk/items/b6a97f0c-3b9b-40ad-8f18-3d121eef1459
- License: free for non-commercial research (per the PEIPA terms)

## Expected on-disk layout

```
MIAS/
├── Info.txt              # one row per image, with optional lesion (x, y, r)
├── all-mias/             # 322 *.pgm files
└── preprocess.py
```

## Processing

```bash
python preprocess.py
```

For every PGM the script:

1. Calls `ExtractBreastWithOffsets` to crop the breast region and recover
   the (x, y) offset.
2. Saves the breast-cropped PNG into `./Processed/breast/<ref>.png`.
3. For every lesion in `Info.txt`, remaps the centre / radius into cropped
   coordinates and saves the corresponding patch into
   `./Processed/finding/<ref>_<i>.png`.

## Citation

```bibtex
@inproceedings{suckling1994mias,
  title     = {The Mammographic Image Analysis Society digital mammogram database},
  author    = {Suckling, John and Parker, Jonathan and Dance, David and others},
  booktitle = {Excerpta Medica International Congress Series},
  volume    = {1069},
  pages     = {375--378},
  year      = {1994}
}
```
