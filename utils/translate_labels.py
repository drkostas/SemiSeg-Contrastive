from glob import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

colors = [[128, 64, 128],
          [244, 35, 232],
          [70, 70, 70],
          [102, 102, 156],
          [190, 153, 153],
          [153, 153, 153],
          [250, 170, 30],
          [220, 220, 0],
          [107, 142, 35],
          [152, 251, 152],
          [70, 130, 180],
          [220, 20, 60],
          [255, 0, 0],
          [0, 0, 142],
          [0, 0, 70],
          [0, 60, 100],
          [0, 80, 100],
          [0, 0, 230],
          [119, 11, 32],
          ]

# --- Modified to work with the MiniFrance dataset --- #
# https://ieee-dataport.org/open-access/minifrance

# --- Define folder structure --- #
# Set the folder paths
base_folder = '/data/datasets/DFC2022/Raw_data'
lbl_tr_dir = f'{base_folder}/labeled_train'
unlbl_tr_dir = f'{base_folder}/unlabeled_train'
lbl_val_dir = f'{base_folder}/val'
# --- Get the city names --- #
lbl_tr_cities = os.listdir(lbl_tr_dir)
# Print the city names
print("Labeled Train Cities:")
print(lbl_tr_cities)
# Default Attributes for every image
TRUTH_attr = 'UrbanAtlas'

# --- Define dictionaries with datapaths --- #
lbl_tr_paths = defaultdict(list)
for city in lbl_tr_cities:
    truth_path = f"{lbl_tr_dir}/{city}/{TRUTH_attr}"
    for truth_img_path in glob(f"{truth_path}/*"):
        lbl_tr_paths[city].append(truth_img_path)

# --- Load and transform Data --- #

for city, paths in lbl_tr_paths.items():
    for img_path in tqdm(paths):
        image = cv2.imread(img_path)
        results = np.ones_like(image[:, :, 0]) * 250

        for i in range(len(colors)):
            color_i = colors[i]
            class_i_image1 = image[:, :, 0] == color_i[2]
            class_i_image2 = image[:, :, 1] == color_i[1]
            class_i_image3 = image[:, :, 2] == color_i[0]

            class_i_image = class_i_image1 & class_i_image2 & class_i_image3

            results[class_i_image] = i
        img_path = img_path.replace(f'{TRUTH_attr}', f'{TRUTH_attr}_translated')
        img_folder = f'{os.sep}'.join((img_path.split(os.sep)[:-1]))

        os.makedirs(img_folder, exist_ok=True)
        cv2.imwrite(img_path, results)
