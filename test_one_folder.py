import os
import glob
import rasterio
import pandas as pd


# ============================================================
# PATHS
# ============================================================

TILE_FOLDER = r"C:\urban development kaggle 3\dataset\SN7_buildings_train\train\L15-0331E-1257N_1327_3160_13"

IMAGES_FOLDER = os.path.join(TILE_FOLDER, "images")

OUTPUT_FOLDER = r"C:\urban development kaggle 3\test_csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# FIND TIFF FILES
# ============================================================

tif_files = sorted(glob.glob(os.path.join(IMAGES_FOLDER, "*.tif")))

print("=" * 70)
print("SN7 TIFF TEST")
print("=" * 70)

print(f"Images folder : {IMAGES_FOLDER}")
print(f"TIFF files    : {len(tif_files)}")
print()


if len(tif_files) == 0:
    print("ERROR: No TIFF files found.")
    exit()


# ============================================================
# PROCESS TIFF FILES
# ============================================================

for tif_file in tif_files:

    print("-" * 70)
    print("Processing:")
    print(os.path.basename(tif_file))

    try:

        with rasterio.open(tif_file) as src:

            print("Width  :", src.width)
            print("Height :", src.height)
            print("Bands  :", src.count)
            print("CRS    :", src.crs)
            print("Transform:", src.transform)

            # Read first band
            data = src.read(1)

            transform = src.transform

            rows = []
            cols = []

            # Find valid pixels
            import numpy as np

            valid = np.isfinite(data) & (data > 0)

            row_indices, col_indices = np.where(valid)

            values = data[valid]

            # Convert pixel coordinates to geographic coordinates
            xs, ys = rasterio.transform.xy(
                transform,
                row_indices,
                col_indices
            )

            df = pd.DataFrame({
                "longitude": xs,
                "latitude": ys,
                "luminous_value": values
            })

            output_name = os.path.splitext(
                os.path.basename(tif_file)
            )[0] + ".csv"

            output_file = os.path.join(
                OUTPUT_FOLDER,
                output_name
            )

            df.to_csv(
                output_file,
                index=False
            )

            print("CSV saved :", output_file)
            print("Rows      :", len(df))

    except Exception as e:

        print("ERROR:", e)


print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)