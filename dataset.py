import os
import cv2
import numpy as np
import rasterio

from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])


def get_val_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(),
        ToTensorV2(),
    ])


class SegDataset(Dataset):

    def __init__(self, img_dir, mask_dir, transform=None):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(img_dir))
        self.transform = transform

        # Path to VIIRS folder
        self.viirs_root = r"C:\urban development kaggle 3\dataset\VIIRS"

    def __len__(self):
        return len(self.files)

    def get_month(self, filename):

        parts = filename.split("_")

        year = parts[2]
        month = parts[3]

        return year, month

    def get_viirs_file(self, year, month):

        folder = os.path.join(self.viirs_root, year)

        for f in os.listdir(folder):

            if f.endswith(".tif") and ".avg_rade9h.masked.tif" in f:

                if f"{year}{month}" in f:
                    return os.path.join(folder, f)

        raise FileNotFoundError(
            f"No VIIRS image found for {year}-{month}"
        )

    def get_original_tif(self, filename):

        parts = filename.replace(".png", "").split("_")

        year = parts[2]
        month = parts[3]

        aoi = "_".join(parts[5:])

        tif_name = filename.replace(".png", ".tif")

        tif_path = os.path.join(
            "dataset",
            "SN7_buildings_train",
            "train",
            aoi,
            "images",
            tif_name
        )

        return tif_path

    def __getitem__(self, idx):

        fname = self.files[idx]

        img_path = os.path.join(self.img_dir, fname)

        # ---------------- RGB Image ----------------
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ---------------- Mask ----------------
        mask = cv2.imread(
            os.path.join(self.mask_dir, fname),
            cv2.IMREAD_GRAYSCALE
        )

        mask = (mask > 127).astype(np.float32)

        # ---------------- Get Year & Month ----------------
        year, month = self.get_month(fname)

        # ---------------- Find Corresponding VIIRS ----------------
        viirs_path = self.get_viirs_file(year, month)

        # ---------------- Convert Bounds ----------------
        tif_path = self.get_original_tif(fname)

        print(tif_path)
        print(os.path.exists(tif_path))

        with rasterio.open(tif_path) as src:

            bounds = transform_bounds(
                src.crs,
                "EPSG:4326",
                *src.bounds
            )

        # ---------------- Read VIIRS ----------------
        with rasterio.open(viirs_path) as viirs:

            window = from_bounds(
                *bounds,
                transform=viirs.transform
            )

            window = window.round_offsets().round_lengths()

            night = viirs.read(
                1,
                window=window,
                out_shape=(img.shape[0], img.shape[1]),
                boundless=True,
                fill_value=0
            )

        # ---------------- Normalize VIIRS ----------------
        night = night.astype(np.float32)

        if night.max() > night.min():
            night = (
                night - night.min()
            ) / (
                night.max() - night.min()
            )
        else:
            night = np.zeros_like(
                night,
                dtype=np.float32
            )

        # ---------------- Stack RGB + VIIRS ----------------
        img = np.dstack([img, night])

        # ---------------- Augmentation ----------------
        if self.transform:

            aug = self.transform(
                image=img,
                mask=mask
            )

            img = aug["image"]
            mask = aug["mask"]

        mask = mask.unsqueeze(0)

        return img, mask