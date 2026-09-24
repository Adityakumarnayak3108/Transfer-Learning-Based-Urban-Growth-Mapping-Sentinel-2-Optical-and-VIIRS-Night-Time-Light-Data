"""
Convert TIFF files to individual CSV files.

Each TIFF gets its own CSV containing:
longitude, latitude, luminous_value, folder, filename
"""

import os
import glob
import rasterio
import pandas as pd
import numpy as np
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

BASE_DATASET_PATH = r"C:\urban development kaggle 3\dataset"

OUTPUT_DIR = r"C:\urban development kaggle 3\test_csv"


# ============================================================
# SETTINGS
# ============================================================

SKIP_ZERO_VALUES = True
SKIP_NEGATIVE_VALUES = True

# Number of pixels written at one time
CHUNK_SIZE = 500_000


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIND TRAIN DIRECTORY
# ============================================================

TRAIN_PATH = os.path.join(
    BASE_DATASET_PATH,
    "SN7_buildings_train",
    "train"
)


if not os.path.exists(TRAIN_PATH):
    print("ERROR: Train folder not found:")
    print(TRAIN_PATH)
    exit()


# ============================================================
# FIND ALL TILE FOLDERS
# ============================================================

tile_folders = sorted([
    f for f in os.listdir(TRAIN_PATH)
    if os.path.isdir(os.path.join(TRAIN_PATH, f))
])


print("=" * 80)
print("TIFF → INDIVIDUAL CSV CONVERTER")
print("=" * 80)

print(f"Dataset : {TRAIN_PATH}")
print(f"Output  : {OUTPUT_DIR}")
print(f"Folders : {len(tile_folders)}")
print("=" * 80)


# ============================================================
# STATISTICS
# ============================================================

total_tif = 0
successful = 0
failed = 0
total_records = 0


# ============================================================
# PROCESS EACH TILE FOLDER
# ============================================================

for folder_number, tile_folder in enumerate(tile_folders, 1):

    tile_path = os.path.join(TRAIN_PATH, tile_folder)

    images_path = os.path.join(
        tile_path,
        "images"
    )

    if not os.path.exists(images_path):
        print(
            f"[{folder_number}/{len(tile_folders)}] "
            f"{tile_folder} → NO IMAGES FOLDER"
        )
        continue


    # --------------------------------------------------------
    # FIND TIFF FILES
    # --------------------------------------------------------

    tif_files = sorted(
        glob.glob(
            os.path.join(images_path, "*.tif")
        )
    )


    if not tif_files:
        print(
            f"[{folder_number}/{len(tile_folders)}] "
            f"{tile_folder} → NO TIFF FILES"
        )
        continue


    print()
    print("-" * 80)

    print(
        f"[{folder_number:02d}/{len(tile_folders)}] "
        f"{tile_folder}"
    )

    print(
        f"TIFF files: {len(tif_files)}"
    )

    print("-" * 80)


    # ========================================================
    # PROCESS EACH TIFF
    # ========================================================

    for tif_number, tif_file in enumerate(tif_files, 1):

        total_tif += 1

        tif_filename = os.path.basename(tif_file)

        # Remove .tif
        tif_name = os.path.splitext(
            tif_filename
        )[0]


        # ----------------------------------------------------
        # Create safe folder name
        # ----------------------------------------------------

        safe_folder = "".join(
            c if c.isalnum() or c in "_-"
            else "_"
            for c in tile_folder
        )


        # ----------------------------------------------------
        # Output CSV
        # ----------------------------------------------------

        output_filename = (
            f"{safe_folder}__{tif_name}.csv"
        )

        output_csv = os.path.join(
            OUTPUT_DIR,
            output_filename
        )


        print()
        print(
            f"  [{tif_number}/{len(tif_files)}] "
            f"{tif_filename}"
        )


        try:

            # =================================================
            # OPEN TIFF
            # =================================================

            with rasterio.open(tif_file) as src:

                print(
                    f"      Size: "
                    f"{src.width} x {src.height}"
                )

                print(
                    f"      CRS: {src.crs}"
                )

                print(
                    f"      Converting..."
                )


                # ------------------------------------------------
                # READ FIRST BAND
                # ------------------------------------------------

                data = src.read(1)

                height, width = data.shape

                transform = src.transform


                # ------------------------------------------------
                # PROCESS ROW CHUNKS
                # ------------------------------------------------

                csv_written = False

                file_records = 0


                for row_start in range(
                    0,
                    height,
                    max(
                        1,
                        CHUNK_SIZE // width
                    )
                ):

                    row_end = min(
                        row_start +
                        max(
                            1,
                            CHUNK_SIZE // width
                        ),
                        height
                    )


                    # --------------------------------------------
                    # Read chunk
                    # --------------------------------------------

                    chunk = data[
                        row_start:row_end,
                        :
                    ]


                    rows, cols = np.where(
                        np.ones(
                            chunk.shape,
                            dtype=bool
                        )
                    )


                    values = chunk[
                        rows,
                        cols
                    ].astype(np.float64)


                    # --------------------------------------------
                    # Filter values
                    # --------------------------------------------

                    mask = np.ones(
                        len(values),
                        dtype=bool
                    )


                    if SKIP_ZERO_VALUES:

                        mask &= (
                            values != 0
                        )


                    if SKIP_NEGATIVE_VALUES:

                        mask &= (
                            values >= 0
                        )


                    rows = rows[mask]
                    cols = cols[mask]
                    values = values[mask]


                    if len(values) == 0:
                        continue


                    # --------------------------------------------
                    # Convert row/column to coordinates
                    # --------------------------------------------

                    actual_rows = (
                        rows +
                        row_start
                    )


                    longitudes = (
                        transform.c +
                        (
                            cols *
                            transform.a
                        )
                    )


                    latitudes = (
                        transform.f +
                        (
                            actual_rows *
                            transform.e
                        )
                    )


                    # --------------------------------------------
                    # Create DataFrame
                    # --------------------------------------------

                    df = pd.DataFrame({

                        "longitude":
                            longitudes,

                        "latitude":
                            latitudes,

                        "luminous_value":
                            values,

                        "folder":
                            tile_folder,

                        "filename":
                            tif_filename
                    })


                    # --------------------------------------------
                    # Write chunk directly to CSV
                    # --------------------------------------------

                    df.to_csv(
                        output_csv,
                        mode="a",
                        header=not csv_written,
                        index=False
                    )


                    csv_written = True

                    file_records += len(df)

                    total_records += len(df)


                    # Free memory
                    del df
                    del chunk
                    del values
                    del rows
                    del cols
                    del actual_rows
                    del longitudes
                    del latitudes


                # ------------------------------------------------
                # Finished TIFF
                # ------------------------------------------------

                if csv_written:

                    print(
                        f"      ✓ CSV created"
                    )

                    print(
                        f"      Records: "
                        f"{file_records:,}"
                    )

                    print(
                        f"      Output: "
                        f"{output_csv}"
                    )

                    successful += 1

                else:

                    print(
                        "      ⚠ No valid pixels found"
                    )


        except Exception as e:

            failed += 1

            print(
                f"      ✗ ERROR: {e}"
            )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("CONVERSION COMPLETED")
print("=" * 80)

print(
    f"Total TIFF files : {total_tif}"
)

print(
    f"Successful       : {successful}"
)

print(
    f"Failed           : {failed}"
)

print(
    f"Total records    : {total_records:,}"
)

print()
print(
    f"CSV files saved in:"
)

print(
    OUTPUT_DIR
)

print("=" * 80)