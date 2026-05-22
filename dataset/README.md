# Dataset preparation

BreastGPT is trained on **BreastStage** — a workflow-aligned breast imaging instruction corpus assembled from public + private sources across 5 modalities. The instruction annotations are hosted on ModelScope:

- **[YYangYang/BreastStage](https://www.modelscope.cn/datasets/YYangYang/BreastStage)** — training corpus (≈ 1.86 M instruction-following pairs)
- **[YYangYang/BreastStage-Bench](https://www.modelscope.cn/datasets/YYangYang/BreastStage-Bench)** — held-out evaluation split

The annotation JSONs reference image paths but **do not ship the images themselves** — each upstream dataset must be downloaded under its own license, then preprocessed locally.

## Pipeline

```
raw download             dataset/<Modality>/<Source>/preprocess.py            resize.py
   │                                          │                                  │
   ▼                                          ▼                                  ▼
   <DATASET>/                       <DATASET>/Processed/             RESIZED/<Modality>/<Source>/
   ├── DICOM / NIfTI / WSI          ├── breast-cropped PNGs          ├── token-budget PNGs
   └── manifest                     └── selected NIfTI               └── 384×48 NIfTI
```

Two stages. Per-dataset `preprocess.py` files do dataset-specific work (DICOM → PNG, breast-region extraction, mask rendering, …). The top-level `resize.py` then applies the canonical BreastGPT input resize, modality-uniform.

## Per-dataset docs

| Modality | Dataset | Source page |
| --- | --- | --- |
| BUS | [BUS-CoT](./BUS/BUS-CoT/README.md) | https://doi.org/10.6084/m9.figshare.29036876.v1 |
| CT | [CT-RATE](./CT/CT-RATE/README.md) | https://huggingface.co/datasets/ibrahimhamamci/CT-RATE |
| Histopathology | [TCGA-BRCA · BCNB · HISTAI-breast](./Histopathology/README.md) | (see README) |
| MRI | [FUDAN](./MRI/FUDAN/README.md) · [ZHE2](./MRI/ZHE2/README.md) | Private — DUA required |
| Mammography | [BMCD](./Mammography/BMCD/README.md) | https://zenodo.org/records/5036062 |
| | [CBIS-DDSM](./Mammography/CBIS-DDSM/README.md) | https://www.cancerimagingarchive.net/collection/cbis-ddsm/ |
| | [CDD-CESM](./Mammography/CDD-CESM/README.md) | https://www.cancerimagingarchive.net/collection/cdd-cesm/ |
| | [CSAW-M](./Mammography/CSAW-M/README.md) | https://figshare.scilifelab.se/articles/dataset/14687271 |
| | [DMID](./Mammography/DMID/README.md) | https://figshare.com/articles/dataset/24522883 |
| | [EMBED](./Mammography/EMBED/README.md) | https://registry.opendata.aws/emory-breast-imaging-dataset-embed/ |
| | [INbreast](./Mammography/INbreast/README.md) | https://www.academictorrents.com/details/ce1ecade37814701ac95193a910a3c6917ea43b3 |
| | [KAU-BCMD](./Mammography/KAU-BCMD/README.md) | https://www.kaggle.com/datasets/asmaasaad/king-abdulaziz-university-mammogram-dataset |
| | [MIAS](./Mammography/MIAS/README.md) | http://peipa.essex.ac.uk/info/mias.html |
| | [RSNA](./Mammography/RSNA/README.md) | https://www.kaggle.com/competitions/rsna-breast-cancer-detection |
| | [VinDr](./Mammography/VinDr/README.md) | https://physionet.org/content/vindr-mammo/1.0.0/ |

## Stage 1 — per-dataset preprocessing

Each per-dataset folder ships a `preprocess.py` and a `README.md`. Each preprocess script:

1. Reads the dataset's native format (DICOM / NIfTI / WSI / PNG)
2. Applies dataset-specific image extraction — windowing, breast-region cropping (`ExtractBreast`-style column/row statistics), masking, format conversion
3. Writes the result into a `Processed/` subfolder

See each dataset's README for the expected on-disk layout and the exact CLI.

## Stage 2 — uniform BreastGPT resize

`resize.py` is the second stage. It takes the `Processed/` folder of stage 1 and produces the resized image cache that BreastStage's `SWIFT_RESIZED_*.json` annotations reference:

| Input | Operation | Output |
| --- | --- | --- |
| `.png` / `.jpg` (BUS, Mammography) | Aspect-preserving rescale into a 1024-token budget (1024 × 32 × 32 pixels) | `.png` |
| `.png` (Histopathology patches) | Aspect-preserving rescale with `--maxsize 256` | `.png` |
| `.nii` / `.nii.gz` (CT, MRI) | Longest in-plane edge → 384, depth → 48 slices, int16 | `.nii.gz` |

```bash
# Single file
python resize.py one input.png output.png

# Whole tree (parallel, 64 workers by default)
python resize.py tree /data/Images/Mammography /data/RESIZED/Mammography

# Histopathology patches: pass --maxsize 256
python resize.py tree /data/Images/Histopathology /data/RESIZED/Histopathology --maxsize 256
```

Use the path-substitution snippet in the [BreastStage dataset README](https://www.modelscope.cn/datasets/YYangYang/BreastStage) to repoint the JSON annotations to your local `RESIZED/` root.

## Histopathology patching (upstream of `preprocess.py`)

WSIs (TCGA-BRCA / BCNB / HISTAI-breast) are first tiled into 512 × 512 patches at 20× magnification with 0 overlap before resize. The [`Histopathology/run_batch_of_slides.py`](./Histopathology/run_batch_of_slides.py) and [`run_single_slide.py`](./Histopathology/run_single_slide.py) scripts drive this stage; the resulting layout per dataset:

```
<dataset>/Processed/20x_512px_0px_overlap/
├── images_32/                # 32-pixel-resized JPEG patches (input to resize.py)
└── features_conch_v15/       # CONCH v1.5 patch embeddings (consumed by BreastGPT's GigaPixel branch)
```

## Licensing

Each upstream dataset has its own license / DUA. Follow the terms on the original source page. The BreastStage *instruction annotations* are CC BY-NC 4.0; the *images* remain under their original licenses.
