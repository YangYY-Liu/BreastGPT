import os
import pandas as pd
import numpy as np
import cv2
import pydicom
import tifffile as tiff
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut


def np_CountUpContinuingOnes(b_arr):
    left = np.arange(len(b_arr))
    left[b_arr > 0] = 0
    left = np.maximum.accumulate(left)

    rev_arr = b_arr[::-1]
    right = np.arange(len(rev_arr))
    right[rev_arr > 0] = 0
    right = np.maximum.accumulate(right)
    right = len(rev_arr) - 1 - right[::-1]

    return right - left - 1


def ExtractBreast(img, mask=None):
    img_copy = img.copy()
    img = np.where(img <= 10, 0, img)
    height, _ = img.shape

    # ---- column crop ----
    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].std(axis=0) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    col_ind = np.where(continuing_ones == continuing_ones.max())[0]

    # ---- row crop (on column-cropped image) ----
    img_col = img[:, col_ind]
    _, width = img_col.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img_col[:, x_b:x_a].std(axis=1) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    row_ind = np.where(continuing_ones == continuing_ones.max())[0]

    x_min = int(col_ind.min())
    x_max = int(col_ind.max()) + 1
    y_min = int(row_ind.min())
    y_max = int(row_ind.max()) + 1

    cropped = img_copy[y_min:y_max, x_min:x_max]
    if mask is not None:
        if isinstance(mask, list):
            mask = [m[y_min:y_max, x_min:x_max] for m in mask]
        else:
            mask = mask[y_min:y_max, x_min:x_max]
    return cropped, mask


def read_image(source):
    """Read DICOM / TIFF / PNG and return an 8-bit single-channel image."""
    if source is None:
        return source
    if source.endswith('.dcm'):
        dicom = pydicom.dcmread(source)
        image = dicom.pixel_array
        image = apply_voi_lut(image, dicom, prefer_lut=False)
        image = (image - image.min()) / (image.max() - image.min())
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            image = 1 - image
        image = (image * 255).astype(np.uint8)
    elif source.endswith('.tif'):
        image = np.array(tiff.imread(source))
    else:
        image = np.array(Image.open(source))
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim != 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def process_and_save():
    """Iterate over INbreast.xlsx and convert all referenced DICOMs to breast-cropped PNGs."""
    df = pd.read_excel('./INbreast.xlsx')
    df = df[~df['File Name'].isna()]
    DCM_PATH = "./AllDICOMs"
    OUTPUT_BASE_PATH = "./Processed"
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    ALL_DCMs = {i.split('_')[0]: i for i in os.listdir(DCM_PATH) if i.endswith('.dcm')}

    for _, row in df.iterrows():
        key = str(row['File Name']).split('.')[0]
        if key not in ALL_DCMs:
            print(f"DICOM file for {key} not found.")
            continue
        dcm_name = ALL_DCMs[key]
        file_name = key + '.png'
        dcm_path = os.path.join(DCM_PATH, dcm_name)
        if os.path.exists(dcm_path):
            image = read_image(dcm_path)
            image, _ = ExtractBreast(image)
            img_output_path = os.path.join(OUTPUT_BASE_PATH, file_name)
            Image.fromarray(image).save(img_output_path)
            print(f"Processed {file_name}")
        else:
            print(f"DICOM file for {file_name} not found.")


if __name__ == '__main__':
    process_and_save()
