import os
import glob
import re
import rasterio
import pandas as pd


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SN7_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "SN7_buildings_train",
    "train"
)


# ==========================================================
# FIND ALL SN7 TIFF FILES
# ==========================================================

tif_files = glob.glob(
    os.path.join(
        SN7_FOLDER,
        "**",
        "*.tif"
    ),
    recursive=True
)

print("=" * 70)
print("SN7 GEOGRAPHIC INFORMATION")
print("=" * 70)

print("Total TIFF files:", len(tif_files))


# ==========================================================
# FIND UNIQUE TILE FOLDERS
# ==========================================================

tile_folders = set()

for tif_file in tif_files:

    # Example:
    # .../train/L15-0331E-1257N_1327_3160_13/images/file.tif

    images_folder = os.path.dirname(tif_file)

    tile_folder = os.path.dirname(
        images_folder
    )

    tile_folders.add(tile_folder)


print(
    "Unique geographic tiles:",
    len(tile_folders)
)


# ==========================================================
# READ ONE IMAGE FROM EACH TILE
# ==========================================================

results = []


for count, tile_folder in enumerate(
    sorted(tile_folders),
    start=1
):

    image_files = glob.glob(
        os.path.join(
            tile_folder,
            "images",
            "*.tif"
        )
    )

    if not image_files:
        continue

    image_file = image_files[0]

    tile_name = os.path.basename(
        tile_folder
    )

    try:

        with rasterio.open(
            image_file
        ) as src:

            bounds = src.bounds

            results.append({

                "tile": tile_name,

                "image": os.path.basename(
                    image_file
                ),

                "crs": str(src.crs),

                "width": src.width,

                "height": src.height,

                "min_x": bounds.left,

                "min_y": bounds.bottom,

                "max_x": bounds.right,

                "max_y": bounds.top

            })

        print(
            f"[{count}/{len(tile_folders)}] "
            f"{tile_name}"
        )

    except Exception as e:

        print(
            "ERROR:",
            image_file
        )

        print(e)


# ==========================================================
# CREATE DATAFRAME
# ==========================================================

df = pd.DataFrame(
    results
)


# ==========================================================
# SAVE TILE INFORMATION
# ==========================================================

output_folder = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "viirs_csv"
)

os.makedirs(
    output_folder,
    exist_ok=True
)


output_file = os.path.join(
    output_folder,
    "sn7_tile_bounds.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ==========================================================
# DISPLAY INFORMATION
# ==========================================================

print("\n")
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    "Unique tiles processed:",
    len(df)
)

print(
    "Saved:",
    output_file
)

print("\nFirst 10 tiles:")

print(
    df.head(10).to_string(
        index=False
    )
)


# ==========================================================
# OVERALL GEOGRAPHIC EXTENT
# ==========================================================

if len(df) > 0:

    print("\n")
    print("=" * 70)
    print("OVERALL GEOGRAPHIC EXTENT")
    print("=" * 70)

    print(
        "Minimum X:",
        df["min_x"].min()
    )

    print(
        "Maximum X:",
        df["max_x"].max()
    )

    print(
        "Minimum Y:",
        df["min_y"].min()
    )

    print(
        "Maximum Y:",
        df["max_y"].max()
    )

    print(
        "\nCRS used by first tiles:"
    )

    print(
        df["crs"].value_counts()
    )