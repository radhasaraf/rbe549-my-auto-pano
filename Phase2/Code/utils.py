import argparse
import csv
import os
from random import randint

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--mscoco_data_path", type=str, help="Abs. path to images for synthetic data generation")
args = parser.parse_args()

MSCOCO_DATA_PATH = args.train_data_path
if not MSCOCO_DATA_PATH:
    MSCOCO_DATA_PATH = '/home/radha/WPI/CV/hw1/rrsaraf_p1/Phase2/Data/Train/'


def generate_data(patch_size: int = 128, perturb_max: int = 32, pixel_buffer_len: int = 150):
    """
    Generates synthetic data (original & warped images, homography labels)
    using the approach as described in https://arxiv.org/pdf/1606.03798.pdf
    """
    # Create new folders to save generated data at
    save_orig_data_to = "./Phase2/Data/Raw/Orig/"
    if not os.path.exists(save_orig_data_to):
        os.makedirs(save_orig_data_to, exist_ok=True)

    save_warped_data_to = "./Phase2/Data/Raw/Warped/"
    if not os.path.exists(save_warped_data_to):
        os.makedirs(save_warped_data_to, exist_ok=True)

    stride = int(0.25 * patch_size)
    rho_range = [-perturb_max, perturb_max]

    images = os.listdir(MSCOCO_DATA_PATH)
    img_paths = [MSCOCO_DATA_PATH + image for image in images]

    with open(os.path.join('./Phase2/Data', 'labels.csv'), 'w') as labels_file:
        writer = csv.writer(labels_file)

        for img_path in img_paths:
            img_name = img_path.split('/')[-1].split('.')[0]  # The index of image in MSCOCO dataset given

            # Convert to grayscale
            img = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Create bounds of active region
            h, w = img_gray.shape
            active_h = [pixel_buffer_len, h - pixel_buffer_len]
            active_w = [pixel_buffer_len, w - pixel_buffer_len]

            image_counter = 0
            for i in range(active_h[0], active_h[1], stride):
                if active_h[1] - i < patch_size:  # Ignore dimensions that give poor data
                    continue

                for j in range(active_w[0], active_w[1], stride):
                    if active_w[1] - j < patch_size:  # Ignore dimensions that give poor data
                        continue

                    # Get original and perturbed corners
                    patch_corners = np.array(
                        [(j, i), (j + patch_size, i), (j, i + patch_size), (j + patch_size, i + patch_size)]
                    )
                    perturbations = [(randint(*rho_range), randint(*rho_range)) for _ in range(4)]  # H4pt
                    perturbed_corners = patch_corners + perturbations

                    # Estimate homography between the two sets of corners
                    h, _ = cv2.findHomography(patch_corners, perturbed_corners)
                    transformed_img = cv2.warpPerspective(img_gray, np.linalg.inv(h), img_gray.shape, flags=cv2.INTER_LINEAR)

                    # Get the corresponding original & perturbed patches
                    img_crop = img_gray[j: j + patch_size, i: i + patch_size]
                    transformed_img_crop = transformed_img[j: j + patch_size, i: i + patch_size]

                    image_counter += 1
                    patch_name = f'{img_name}_{image_counter}.jpg'
                    cv2.imwrite(os.path.join(save_orig_data_to, "orig_" + patch_name), img_crop)
                    cv2.imwrite(os.path.join(save_warped_data_to + "warped_" + patch_name), transformed_img_crop)

                    writer.writerow([f'orig_{patch_name}', list(np.array(perturbations).flatten())])


if __name__ == '__main__':
    generate_data()
