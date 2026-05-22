import os
import pandas as pd
import numpy as np
import cv2


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


def ExtractBreast(img):
    img_copy = img.copy()
    img = np.where(img <= 40, 0, img)
    height, _ = img.shape
    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].std(axis=0) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    col_ind = np.where(continuing_ones == continuing_ones.max())[0]
    img = img[:, col_ind]
    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].std(axis=1) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    row_ind = np.where(continuing_ones == continuing_ones.max())[0]

    return img_copy[row_ind][:, col_ind]


def process_all_data():
    excel_path = 'Radiology-manual-annotations.xlsx'
    image_dir = 'Low energy images of CDD-CESM'
    sub_dir = 'input_image_new_v2'
    output_base_dir = 'Processed'
    df = pd.read_excel(excel_path, sheet_name='all')
    df = df[df['Image_name'].str.contains('DM')]
    os.makedirs(output_base_dir, exist_ok=True)
    for _, row in df.iterrows():
        image_name = row['Image_name'].strip()
        image_path = os.path.join(image_dir, image_name + '.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join(sub_dir, image_name + '.jpg')
        if os.path.exists(image_path):
            output_image_path = os.path.join(output_base_dir, image_name + '.png')
            if not os.path.exists(output_image_path):
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img = ExtractBreast(img)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite(output_image_path, img)
        else:
            print(f"Image not found: {image_path}")


if __name__ == '__main__':
    process_all_data()
