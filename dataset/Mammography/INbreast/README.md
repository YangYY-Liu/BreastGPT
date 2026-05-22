# INbreast

INbreast is a full-field digital mammography database with 115 cases
(410 images) collected at the Breast Centre in CHSJ, Porto. It includes
masses, calcifications, asymmetries, distortions and accurate contours in
XML.

## Source & download

- Official paper (Academic Radiology, 2012): https://www.sciencedirect.com/science/article/abs/pii/S107663321100451X
- Historical project page (now intermittently down):
  http://medicalresearch.inescporto.pt/breastresearch/GetINbreastDatabase.html
- Academic Torrents mirror: https://academictorrents.com/details/ce1ecade37814701ac95193a910a3c6917ea43b3
- License: research-use only (see paper)

## Expected on-disk layout

```
INbreast/
├── INbreast.xlsx              # per-image annotations
├── AllDICOMs/*.dcm
└── preprocess.py
```

## Processing

```bash
python preprocess.py
```

For every row of the spreadsheet the script reads the corresponding DICOM
(matched by file-name prefix), applies the VOI LUT, inverts
`MONOCHROME1`, breast-crops, and writes a PNG into
`./Processed/<File Name>.png`.

## Citation

```bibtex
@article{moreira2012inbreast,
  title   = {INbreast: Toward a full-field digital mammographic database},
  author  = {Moreira, In{\^e}s C. and Amaral, Igor and Domingues, In{\^e}s and others},
  journal = {Academic Radiology},
  volume  = {19},
  number  = {2},
  pages   = {236--248},
  year    = {2012},
  doi     = {10.1016/j.acra.2011.09.014}
}
```
