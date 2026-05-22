import os
import cv2
import numpy as np


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


def ExtractBreastWithOffsets(img):
    """Crop the breast region and return the crop together with the (x, y) top-left offsets."""
    img_binary = np.where(img <= 20, 0, img)
    height, width = img.shape
    y_a, y_b = height // 2 + int(height * 0.4), height // 2 - int(height * 0.4)
    b_arr_col = img_binary[y_b:y_a].std(axis=0) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr_col)
    col_ind = np.where(continuing_ones == continuing_ones.max())[0]
    img_binary = img_binary[:, col_ind]
    x_a, x_b = len(col_ind) // 2 + int(len(col_ind) * 0.4), len(col_ind) // 2 - int(len(col_ind) * 0.4)
    b_arr_row = img_binary[:, x_b:x_a].std(axis=1) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr_row)
    row_ind = np.where(continuing_ones == continuing_ones.max())[0]
    return img[row_ind][:, col_ind], col_ind[0], row_ind[0]


def generate_dataset():
    TXT_PATH = './Info.txt'
    PGM_PATH = './all-mias'
    BREAST_OUT = './Processed/breast'
    FINDING_OUT = './Processed/finding'

    for d in [BREAST_OUT, FINDING_OUT]:
        os.makedirs(d, exist_ok=True)

    # 1. Parse Info.txt for lesion centres / radii
    image_info = {}
    with open(TXT_PATH, 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        p = line.split()
        ref = p[0]
        if ref not in image_info:
            image_info[ref] = []
        if len(p) > 4:
            image_info[ref].append({'x': int(p[4]), 'y': 1024 - int(p[5]), 'r': int(p[6])})

    # 2. Process each raw PGM
    for refnum, lesions in image_info.items():
        pgm_path = os.path.join(PGM_PATH, f"{refnum}.pgm")
        if not os.path.exists(pgm_path):
            continue

        # A. Extract breast & normalise
        img_raw = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        img_ext, off_x, off_y = ExtractBreastWithOffsets(img_raw)
        img_norm = cv2.normalize(img_ext, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(BREAST_OUT, f"{refnum}.png"), img_norm)

        # B. Re-map lesion coordinates and crop finding patches
        for idx, roi in enumerate(lesions):
            new_x, new_y = roi['x'] - off_x, roi['y'] - off_y
            r = roi['r']
            x1, y1, x2, y2 = new_x - r, new_y - r, new_x + r, new_y + r
            crop_pos = img_norm[
                max(y1, 0):min(y2, img_norm.shape[0]),
                max(x1, 0):min(x2, img_norm.shape[1]),
            ]
            if crop_pos.size > 0:
                cv2.imwrite(os.path.join(FINDING_OUT, f"{refnum}_{idx + 1}.png"), crop_pos)

        print(f"Processed: {refnum}")


if __name__ == '__main__':
    generate_dataset()
