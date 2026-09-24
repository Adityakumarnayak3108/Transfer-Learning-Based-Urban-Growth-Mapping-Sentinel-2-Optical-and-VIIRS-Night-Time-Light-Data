#!/usr/bin/env python3

import os
import glob
import csv
import rasterio
from rasterio.transform import xy
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

BASE_DATASET_PATH = r"C:\urban development kaggle 3\dataset"

OUTPUT_CSV_FILE = r"C:\urban development kaggle 3\all_satellite_data.csv"


# ============================================================
# SETTINGS
# ============================================================

# Keep ALL pixel values
# False = do NOT remove zero values
SKIP_ZERO_VALUES = False

# Keep ALL negative values
# False = do NOT remove negative values
SKIP_NEGATIVE_VALUES = False

# Number of CSV rows kept temporarily in RAM
# Smaller = less RAM usage
CHUNK_SIZE = 5000


# ============================================================
# CONVERTER
# ============================================================

def convert_all_tifs_to_one_csv():

    train_path = os.path.join(
        BASE_DATASET_PATH,
        "SN7_buildings_train",
        "train"
    )

    # --------------------------------------------------------
    # Check path
    # --------------------------------------------------------

    if not os.path.exists(train_path):

        print("ERROR: Train folder not found:")
        print(train_path)

        return

    # --------------------------------------------------------
    # Find all folders
    # --------------------------------------------------------

    tile_folders = sorted([
        folder
        for folder in os.listdir(train_path)
        if os.path.isdir(
            os.path.join(train_path, folder)
        )
    ])

    print("=" * 80)
    print("ALL TIFF FILES -> ONE SINGLE CSV")
    print("=" * 80)

    print(f"Dataset path : {BASE_DATASET_PATH}")
    print(f"Train path   : {train_path}")
    print(f"Folders found: {len(tile_folders)}")
    print(f"Output CSV   : {OUTPUT_CSV_FILE}")

    print(f"Started      : {datetime.now()}")

    print("=" * 80)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        OUTPUT_CSV_FILE
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Delete old CSV
    # --------------------------------------------------------

    if os.path.exists(OUTPUT_CSV_FILE):

        print("\nOld output CSV exists.")

        os.remove(
            OUTPUT_CSV_FILE
        )

        print("Old CSV deleted.")

    # ========================================================
    # OPEN ONE CSV
    # ========================================================

    with open(
        OUTPUT_CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        # ----------------------------------------------------
        # CSV HEADER
        # ----------------------------------------------------

        writer.writerow([
            "longitude",
            "latitude",
            "luminous_value",
            "folder",
            "filename"
        ])

        # ----------------------------------------------------
        # Counters
        # ----------------------------------------------------

        total_folders = len(
            tile_folders
        )

        total_tif_files = 0

        successful_files = 0

        failed_files = 0

        total_pixels_written = 0

        # ====================================================
        # PROCESS ALL FOLDERS
        # ====================================================

        for folder_index, tile_folder in enumerate(
            tile_folders,
            start=1
        ):

            tile_path = os.path.join(
                train_path,
                tile_folder
            )

            images_path = os.path.join(
                tile_path,
                "images"
            )

            # ------------------------------------------------
            # Check images folder
            # ------------------------------------------------

            if not os.path.exists(images_path):

                print(
                    f"\n[{folder_index}/{total_folders}] "
                    f"{tile_folder}"
                )

                print(
                    "    No images folder - skipped"
                )

                continue

            # ------------------------------------------------
            # Find TIFF files
            # ------------------------------------------------

            tif_files = sorted(
                glob.glob(
                    os.path.join(
                        images_path,
                        "*.tif"
                    )
                )
            )

            if not tif_files:

                print(
                    f"\n[{folder_index}/{total_folders}] "
                    f"{tile_folder}"
                )

                print(
                    "    No TIFF files - skipped"
                )

                continue

            print()
            print("=" * 80)

            print(
                f"FOLDER "
                f"{folder_index}/{total_folders}: "
                f"{tile_folder}"
            )

            print(
                f"TIFF files: {len(tif_files)}"
            )

            print("=" * 80)

            # =================================================
            # PROCESS EACH TIFF
            # =================================================

            for tif_index, tif_file in enumerate(
                tif_files,
                start=1
            ):

                total_tif_files += 1

                filename = os.path.basename(
                    tif_file
                )

                print(
                    f"\n[{tif_index}/{len(tif_files)}] "
                    f"{filename}"
                )

                try:

                    # =========================================
                    # OPEN TIFF
                    # =========================================

                    with rasterio.open(
                        tif_file
                    ) as src:

                        width = src.width

                        height = src.height

                        transform = src.transform

                        print(
                            f"    Dimensions: "
                            f"{width:,} x {height:,}"
                        )

                        print(
                            f"    CRS: {src.crs}"
                        )

                        # =====================================
                        # PROCESS ROW BY ROW
                        # =====================================

                        for row in range(height):

                            # ---------------------------------
                            # Read ONLY one row
                            # ---------------------------------

                            data = src.read(
                                1,
                                window=(
                                    (row, row + 1),
                                    (0, width)
                                )
                            )[0]

                            # ---------------------------------
                            # Small temporary buffer
                            # ---------------------------------

                            rows_to_write = []

                            for col in range(width):

                                pixel_value = float(
                                    data[col]
                                )

                                # --------------------------------
                                # DO NOT REMOVE ZERO
                                # --------------------------------

                                if (
                                    SKIP_ZERO_VALUES
                                    and
                                    pixel_value == 0
                                ):
                                    continue

                                # --------------------------------
                                # DO NOT REMOVE NEGATIVE
                                # --------------------------------

                                if (
                                    SKIP_NEGATIVE_VALUES
                                    and
                                    pixel_value < 0
                                ):
                                    continue

                                # --------------------------------
                                # Calculate coordinates
                                # --------------------------------

                                longitude, latitude = xy(
                                    transform,
                                    row,
                                    col,
                                    offset="center"
                                )

                                rows_to_write.append([
                                    longitude,
                                    latitude,
                                    pixel_value,
                                    tile_folder,
                                    filename
                                ])

                                # --------------------------------
                                # Write every 5000 rows
                                # --------------------------------

                                if (
                                    len(rows_to_write)
                                    >= CHUNK_SIZE
                                ):

                                    writer.writerows(
                                        rows_to_write
                                    )

                                    total_pixels_written += len(
                                        rows_to_write
                                    )

                                    rows_to_write = []

                            # ---------------------------------
                            # Write remaining rows
                            # ---------------------------------

                            if rows_to_write:

                                writer.writerows(
                                    rows_to_write
                                )

                                total_pixels_written += len(
                                    rows_to_write
                                )

                            # ---------------------------------
                            # Flush to disk periodically
                            # ---------------------------------

                            if (
                                row % 100 == 0
                                or
                                row == height - 1
                            ):

                                csv_file.flush()

                                print(
                                    f"\r    Progress: "
                                    f"{row + 1:,}/"
                                    f"{height:,} rows | "
                                    f"CSV records: "
                                    f"{total_pixels_written:,}",
                                    end=""
                                )

                    # =========================================
                    # TIFF SUCCESS
                    # =========================================

                    successful_files += 1

                    print(
                        "\n    ✓ TIFF completed"
                    )

                except Exception as error:

                    failed_files += 1

                    print(
                        "\n    ✗ ERROR:"
                    )

                    print(
                        f"      {error}"
                    )

                    # Continue with next TIFF
                    continue

        # ====================================================
        # FINAL FLUSH
        # ====================================================

        csv_file.flush()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("CONVERSION FINISHED")
    print("=" * 80)

    print(
        f"Folders found       : {total_folders}"
    )

    print(
        f"TIFF files found    : {total_tif_files}"
    )

    print(
        f"Successful TIFFs    : {successful_files}"
    )

    print(
        f"Failed TIFFs        : {failed_files}"
    )

    print(
        f"Total CSV rows      : "
        f"{total_pixels_written:,}"
    )

    print(
        f"Output file         : "
        f"{OUTPUT_CSV_FILE}"
    )

    print(
        f"Finished            : "
        f"{datetime.now()}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Check final file size
    # --------------------------------------------------------

    if os.path.exists(
        OUTPUT_CSV_FILE
    ):

        file_size = os.path.getsize(
            OUTPUT_CSV_FILE
        )

        file_size_gb = (
            file_size /
            (1024 ** 3)
        )

        print(
            f"\nCSV file size: "
            f"{file_size_gb:.2f} GB"
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    convert_all_tifs_to_one_csv()