import os
import json
from glob import glob

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import shape
from tqdm import tqdm


def make_mask_for_aoi(aoi_dir, out_dir):
    img_dir = os.path.join(aoi_dir, "images")
    geo_dir = os.path.join(aoi_dir, "labels_match_pix")

    os.makedirs(out_dir, exist_ok=True)

    image_files = sorted(glob(os.path.join(img_dir, "*.tif")))

    for img_path in tqdm(image_files, desc=os.path.basename(aoi_dir)):
        base = os.path.splitext(os.path.basename(img_path))[0]

        geojson_path = os.path.join(
            geo_dir,
            base + "_Buildings.geojson"
        )

        with rasterio.open(img_path) as src:
            h = src.height
            w = src.width

        mask = np.zeros((h, w), dtype=np.uint8)

        if os.path.exists(geojson_path):
            with open(geojson_path) as f:
                data = json.load(f)

            polygons = []

            for feature in data["features"]:
                geom = feature["geometry"]

                if geom is not None:
                    polygons.append((shape(geom), 1))

            if len(polygons) > 0:
                mask = rasterize(
                    polygons,
                    out_shape=(h, w),
                    fill=0,
                    default_value=1,
                    transform=Affine.identity(),
                    dtype=np.uint8
                )

        cv2.imwrite(
            os.path.join(out_dir, base + ".png"),
            mask * 255
        )


if __name__ == "__main__":
    train_root = "dataset/SN7_buildings_train/train"
    out_root = "data/processed_masks"

    os.makedirs(out_root, exist_ok=True)

    for aoi in sorted(os.listdir(train_root)):
        aoi_path = os.path.join(train_root, aoi)

        if os.path.isdir(aoi_path):
            make_mask_for_aoi(
                aoi_path,
                os.path.join(out_root, aoi)
            )

    print("\nFinished generating masks.")