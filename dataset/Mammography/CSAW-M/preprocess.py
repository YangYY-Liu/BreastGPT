import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image


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


def extract_breast(img, mask=None):
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

    # ---- bbox in original image coordinates ----
    x_min = int(col_ind.min())
    x_max = int(col_ind.max()) + 1
    y_min = int(row_ind.min())
    y_max = int(row_ind.max()) + 1

    cropped = img_copy[y_min:y_max, x_min:x_max]
    if mask is not None:
        mask = mask[y_min:y_max, x_min:x_max] * 255
    return cropped, mask


def main():
    train_csv_file = 'labels/CSAW-M_train.csv'
    test_csv_file = 'labels/CSAW-M_test.csv'
    image_dir = 'images/preprocessed'
    output_img_dir = 'Processed'
    os.makedirs(output_img_dir, exist_ok=True)

    train_df = pd.read_csv(train_csv_file, delimiter=';')
    test_df = pd.read_csv(test_csv_file, delimiter=';')
    case_info = pd.concat([train_df, test_df], ignore_index=True).rename(
        columns={'Filename': 'image_path'}
    )
    case_info['image_path'] = case_info['image_path'].map(
        lambda x: os.path.join(image_dir, x.split('_')[0], x)
    )

    for _, row in case_info.iterrows():
        image_path = row['image_path']
        output_image_path = os.path.join(
            output_img_dir,
            f"{os.path.splitext(os.path.basename(image_path))[0]}.png",
        )
        if not os.path.exists(output_image_path):
            image = Image.open(image_path)
            image = np.array(image)
            image, _ = extract_breast(image)
            Image.fromarray(image).save(output_image_path)
            print(f"Saved: {output_image_path}")
        else:
            print(f"Exists: {output_image_path}")


if __name__ == '__main__':
    main()
