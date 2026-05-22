import os
import numpy as np
import pandas as pd
import pydicom
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
    img = np.where(img <= 20, 0, img)
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


def process_and_save_image(case_type, folder_name, file_name, image_path, output_base_dir):
    os.makedirs(output_base_dir, exist_ok=True)
    output_image_path = os.path.join(
        output_base_dir,
        f"{case_type}_{folder_name}_{os.path.basename(file_name).split('.')[0]}.png",
    )
    if not os.path.exists(output_image_path):
        dicom_data = pydicom.dcmread(image_path, force=True)
        image = dicom_data.pixel_array
        image = ExtractBreast(image)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
        cv2.imwrite(output_image_path, image)
    print(f"Saved: {output_image_path}")


def main():
    case_xlsx = 'Description.xlsx'
    normal_cases_dir = 'Normal_cases'
    suspicious_cases_dir = 'Suspicious_cases'
    output_img_dir = 'Processed'

    cases_df = pd.read_excel(case_xlsx, sheet_name=None)
    cases_df['Normal_cases']['case_type'] = 'Normal'
    cases_df['Suspicious_cases']['case_type'] = 'Suspicious'
    df = pd.concat(
        [cases_df['Normal_cases'], cases_df['Suspicious_cases']], ignore_index=True
    )

    for _, row in df.iterrows():
        case_type = row['case_type']
        folder_name = str(row['Folder #'])
        case_dir = normal_cases_dir if case_type == 'Normal' else suspicious_cases_dir
        folder_path = os.path.join(case_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue
        for file_name in os.listdir(folder_path):
            if (file_name.endswith('.dcm') or file_name.endswith('.DCM')) and not file_name.startswith('._'):
                image_path = os.path.join(folder_path, file_name)
                process_and_save_image(
                    case_type, folder_name, file_name, image_path, output_img_dir
                )


if __name__ == '__main__':
    main()
