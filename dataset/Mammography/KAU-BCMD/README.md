# KAU-BCMD

The King Abdulaziz University Breast Cancer Mammogram Dataset (KAU-BCMD).
1,416 cases (5,662 mammograms across CC + MLO views of both breasts) annotated
by three radiologists, plus 205 paired ultrasound cases.

## Source & download

- Kaggle: https://www.kaggle.com/datasets/asmaasaad/king-abdulaziz-university-mammogram-dataset
- Paper / DOI: https://doi.org/10.3390/data6110111
- License: research / educational use only (per the Kaggle page)

## Expected on-disk layout

After unzipping the Kaggle archive you should see four BI-RADS-keyed
folders. Place them under `./archive/`:

```
KAU-BCMD/
├── archive/
│   ├── Birad1/<case>/*.jpg
│   ├── Birad3/<case>/*.jpg
│   ├── Birad4/<case>/*.jpg
│   └── Birad5/<case>/*.jpg
└── preprocess.py
```

## Processing

```bash
python preprocess.py
```

The script walks each BI-RADS folder, reads every JPEG, applies
`ExtractBreast`, and writes the breast-cropped PNG to `./Processed/`.

## Citation

```bibtex
@article{alsolami2021kaubcmd,
  title   = {King Abdulaziz University Breast Cancer Mammogram Dataset (KAU-BCMD)},
  author  = {Alsolami, Asmaa S. and Shalash, Wafaa and Alsaggaf, Wafaa and others},
  journal = {Data},
  volume  = {6},
  number  = {11},
  pages   = {111},
  year    = {2021},
  doi     = {10.3390/data6110111}
}
```
