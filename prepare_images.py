import os
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import rasterio

# CHANGE THIS if your dataset is in a different location
TRAIN_ROOT = "dataset/SN7_buildings_train/train"
OUT_ROOT = "data/processed_images"

os.makedirs(OUT_ROOT, exist_ok=True)

for aoi in sorted(os.listdir(TRAIN_ROOT)):
    aoi_path = os.path.join(TRAIN_ROOT, aoi)
    img_dir = os.path.join(aoi_path, "images")

    if not os.path.isdir(img_dir):
        continue

    out_dir = os.path.join(OUT_ROOT, aoi)
    os.makedirs(out_dir, exist_ok=True)

    tif_files = sorted(glob(os.path.join(img_dir, "*.tif")))

    for tif in tqdm(tif_files, desc=aoi):

        with rasterio.open(tif) as src:
            img = src.read([1, 2, 3])      # RGB bands
            img = np.transpose(img, (1, 2, 0))

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

        out_name = os.path.splitext(os.path.basename(tif))[0] + ".png"
        cv2.imwrite(os.path.join(out_dir, out_name), img)

print("✅ All images converted successfully.")