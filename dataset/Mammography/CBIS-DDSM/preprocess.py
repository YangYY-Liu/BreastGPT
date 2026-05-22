import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import multiprocessing as mp


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

    # also return coordinates so we can align the mask later
    return img_copy[row_ind][:, col_ind], row_ind.min(), row_ind.max(), col_ind.min(), col_ind.max()


def mask_to_bbox_xyxy(mask):
    if mask is None:
        return None
    ys, xs = np.where(mask != 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return [int(x1), int(y1), int(x2), int(y2)]


def process_single_row(args):
    """Process one row: full breast PNG + finding crop + aligned mask."""
    index, row_dict, dicom_map = args

    def get_real_path(csv_path):
        if not isinstance(csv_path, str) or pd.isnull(csv_path):
            return None
        parts = csv_path.split('/')
        if len(parts) < 3:
            return None
        return dicom_map.get((parts[-3], parts[-2]))

    full_path = get_real_path(row_dict.get('image_path'))
    crop_path = get_real_path(row_dict.get('cropped_path'))
    mask_path = get_real_path(row_dict.get('mask_path'))

    if not full_path:
        return None

    b_id = row_dict['image_path'].split('/')[0]
    f_id = str(row_dict['cropped_path']).split('/')[0]
    m_id = str(row_dict['mask_path']).split('/')[0]

    b_out = f"Processed/breast/{b_id}.png"
    f_out = f"Processed/finding/{f_id}.png"
    m_out = f"Processed/mask/{m_id}.png"

    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    breast_img, y_min, y_max, x_min, x_max = ExtractBreast(img)
    breast_img_norm = cv2.normalize(breast_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(b_out, breast_img_norm)

    # ---------------------------------
    # Process Finding crop / Mask
    # ---------------------------------
    raw_crop, raw_mask = None, None
    if crop_path:
        tmp_c = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        if tmp_c is not None:
            raw_crop = cv2.normalize(tmp_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if mask_path:
        tmp_m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if tmp_m is not None:
            raw_mask = cv2.normalize(tmp_m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    f_img, m_img = raw_crop, raw_mask

    # A few mask/crop files in CBIS-DDSM are mis-labelled; auto-detect
    if (raw_crop is not None) and (raw_mask is not None):
        if raw_crop.shape == img.shape and raw_mask.shape != img.shape:
            print(f"[{index}] Mislabeled paths. Swapping...")
            f_img, m_img = raw_mask, raw_crop
        elif raw_crop.shape != img.shape and raw_mask.shape != img.shape:
            if raw_crop.mean() > 30:
                m_img = None
            elif raw_mask.mean() < 30:
                f_img = None
                m_img = cv2.resize(raw_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        elif raw_crop.shape == img.shape and raw_mask.shape == img.shape:
            f_img = None
    else:
        if raw_crop is not None and raw_crop.shape == img.shape:
            m_img, f_img = raw_crop, None
        if raw_mask is not None and raw_mask.shape != img.shape:
            f_img, m_img = raw_mask, None

    # --- Align mask to the cropped breast ---
    mask_aligned = np.zeros_like(breast_img)
    if m_img is not None:
        if m_img.shape == img.shape:
            mask_aligned = m_img[y_min:y_max + 1, x_min:x_max + 1]
        else:
            mask_aligned = cv2.resize(m_img, (breast_img.shape[1], breast_img.shape[0]))

    if f_img is not None and m_img is None:
        try:
            res = cv2.matchTemplate(breast_img_norm, f_img, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            th, tw = f_img.shape[:2]
            mask_aligned[max_loc[1]:max_loc[1] + th, max_loc[0]:max_loc[0] + tw] = 255
        except Exception as e:
            print(f'[ERROR {crop_path}] matchTemplate failed: {e}')
            f_img = None

    # --- Derive finding crop from aligned mask if needed ---
    bbox = mask_to_bbox_xyxy(mask_aligned)
    if f_img is None and bbox is not None:
        x1, y1, x2, y2 = bbox
        if x2 > x1 and y2 > y1:
            f_img = breast_img_norm[y1:y2, x1:x2]

    if f_img is not None:
        cv2.imwrite(f_out, f_img)
    if mask_aligned.max() > 0:
        cv2.imwrite(m_out, mask_aligned)

    return b_out


def process_pipeline(additional_df, num_workers=None):
    for d in ["breast", "finding", "mask"]:
        os.makedirs(f"Processed/{d}", exist_ok=True)

    df_dicom_info = pd.read_csv('csv/dicom_info.csv')
    df_dicom_info['image_path'] = df_dicom_info['image_path'].apply(
        lambda x: str(x).replace('CBIS-DDSM/', '')
    )
    dicom_map = df_dicom_info.set_index(['StudyInstanceUID', 'SeriesInstanceUID'])['image_path'].to_dict()

    tasks = []
    for index, row in additional_df.iterrows():
        tasks.append((index, row.to_dict(), dicom_map))

    workers = num_workers if num_workers else mp.cpu_count() - 1
    print(f"Starting multiprocessing with {workers} workers...")
    with mp.Pool(processes=workers) as pool:
        for _ in tqdm(pool.imap(process_single_row, tasks), total=len(tasks)):
            pass
    print("Pipeline processing complete.")


def read_info():
    description_files = [
        "csv/mass_case_description_train_set.csv",
        "csv/mass_case_description_test_set.csv",
        "csv/calc_case_description_train_set.csv",
        "csv/calc_case_description_test_set.csv",
    ]
    dfs = []
    for csv in description_files:
        if os.path.exists(csv):
            df = pd.read_csv(csv).rename(columns={
                'image file path': 'image_path',
                'ROI mask file path': 'mask_path',
                'cropped image file path': 'cropped_path',
            })
            dfs.append(df)
    additional_df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    return additional_df


if __name__ == '__main__':
    additional_df = read_info()
    process_pipeline(additional_df)
