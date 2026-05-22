import os
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


def save_image(file_path, source_folder, target_folder):
    file_name = os.path.splitext(os.path.basename(file_path))[0].replace(' ', '') + '.png'
    target_img_path = os.path.join(target_folder, file_name)
    image = cv2.imread(os.path.join(source_folder, file_path), cv2.IMREAD_GRAYSCALE)
    processed_image = ExtractBreast(image)
    cv2.imwrite(target_img_path, processed_image)


def process_all_data(source_folder, target_folder):
    categories = ['Birad1', 'Birad3', 'Birad4', 'Birad5']
    os.makedirs(target_folder, exist_ok=True)

    for category in categories:
        category_folder = os.path.join(source_folder, category)
        for root, dirs, files in os.walk(category_folder):
            for file in files:
                if file.endswith('.jpg') and not file.startswith('._'):
                    relative_path = os.path.relpath(os.path.join(root, file), source_folder)
                    save_image(relative_path, source_folder, target_folder)
        print(f'Processed all images in {category} category.')


if __name__ == "__main__":
    source_folder = "./archive"
    target_folder = "./Processed"
    process_all_data(source_folder, target_folder)
