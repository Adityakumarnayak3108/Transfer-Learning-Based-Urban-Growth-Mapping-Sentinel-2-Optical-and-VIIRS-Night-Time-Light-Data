import os
import glob
import rasterio
import pandas as pd
import numpy as np

from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.transform import xy


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# FOLDERS
# ============================================================

VIIRS_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "VIIRS"
)

SN7_BOUNDS_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "viirs_csv",
    "sn7_tile_bounds.csv"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "viirs_csv"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_FOLDER,
    "viirs_sn7_all_years.csv"
)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# READ SN7 TILE BOUNDS
# ============================================================

print("=" * 70)
print("VIIRS EXTRACTION FOR SN7 TILES")
print("=" * 70)

print("\nReading SN7 tile information...")

tiles_df = pd.read_csv(
    SN7_BOUNDS_FILE
)

print(
    "SN7 tiles found:",
    len(tiles_df)
)


# ============================================================
# FIND ALL VIIRS TIFF FILES
# ============================================================

viirs_files = []

for year in ["2017", "2018", "2019", "2020"]:

    year_folder = os.path.join(
        VIIRS_FOLDER,
        year
    )

    files = glob.glob(
        os.path.join(
            year_folder,
            "*.tif"
        )
    )

    files.extend(
        glob.glob(
            os.path.join(
                year_folder,
                "*.tiff"
            )
        )
    )

    viirs_files.extend(files)


print(
    "VIIRS images found:",
    len(viirs_files)
)


# ============================================================
# DISPLAY VIIRS FILES
# ============================================================

print("\nVIIRS files:")

for file in viirs_files:

    print(
        " -",
        os.path.basename(file)
    )


# ============================================================
# REMOVE OLD OUTPUT FILE
# ============================================================

if os.path.exists(OUTPUT_CSV):

    os.remove(
        OUTPUT_CSV
    )

    print(
        "\nOld output CSV removed."
    )


# ============================================================
# PROCESS VIIRS IMAGES
# ============================================================

total_images = len(viirs_files)

total_records = 0

first_write = True


for image_number, viirs_file in enumerate(
    viirs_files,
    start=1
):

    image_name = os.path.basename(
        viirs_file
    )

    # Get year from folder
    year = os.path.basename(
        os.path.dirname(
            viirs_file
        )
    )


    print("\n")
    print("=" * 70)

    print(
        f"[VIIRS IMAGE {image_number}/{total_images}]"
    )

    print(
        "Year:",
        year
    )

    print(
        "Image:",
        image_name
    )

    print("=" * 70)


    # --------------------------------------------------------
    # OPEN VIIRS IMAGE
    # --------------------------------------------------------

    with rasterio.open(
        viirs_file
    ) as src:

        print(
            "CRS:",
            src.crs
        )

        print(
            "Size:",
            src.width,
            "x",
            src.height
        )

        print(
            "Resolution:",
            src.res
        )

        print(
            "NoData:",
            src.nodata
        )


        # ----------------------------------------------------
        # PROCESS EACH SN7 TILE
        # ----------------------------------------------------

        image_records = 0


        for tile_number, tile in tiles_df.iterrows():

            tile_name = tile["tile"]


            # ------------------------------------------------
            # SN7 BOUNDS ARE EPSG:3857
            # CONVERT TO EPSG:4326
            # ------------------------------------------------

            min_lon, min_lat, max_lon, max_lat = (
                transform_bounds(
                    "EPSG:3857",
                    "EPSG:4326",
                    tile["min_x"],
                    tile["min_y"],
                    tile["max_x"],
                    tile["max_y"],
                    densify_pts=21
                )
            )


            # ------------------------------------------------
            # CHECK WHETHER TILE INTERSECTS VIIRS
            # ------------------------------------------------

            viirs_bounds = src.bounds


            if (
                max_lon < viirs_bounds.left
                or
                min_lon > viirs_bounds.right
                or
                max_lat < viirs_bounds.bottom
                or
                min_lat > viirs_bounds.top
            ):

                continue


            # ------------------------------------------------
            # CREATE WINDOW
            # ------------------------------------------------

            try:

                window = from_bounds(
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                    transform=src.transform
                )

            except Exception:

                continue


            # ------------------------------------------------
            # CLIP WINDOW TO IMAGE
            # ------------------------------------------------

            window = window.intersection(
                rasterio.windows.Window(
                    0,
                    0,
                    src.width,
                    src.height
                )
            )


            # ------------------------------------------------
            # SKIP EMPTY WINDOW
            # ------------------------------------------------

            if (
                window.width <= 0
                or
                window.height <= 0
            ):

                continue


            # ------------------------------------------------
            # READ VIIRS DATA
            # ------------------------------------------------

            data = src.read(
                1,
                window=window
            )


            if data.size == 0:

                continue


            # ------------------------------------------------
            # GET ROW/COLUMN POSITIONS
            # ------------------------------------------------

            row_start = int(
                window.row_off
            )

            col_start = int(
                window.col_off
            )

            rows = np.arange(
                row_start,
                row_start + data.shape[0]
            )

            cols = np.arange(
                col_start,
                col_start + data.shape[1]
            )


            row_grid, col_grid = np.meshgrid(
                rows,
                cols,
                indexing="ij"
            )


            # ------------------------------------------------
            # GET LAT/LON OF PIXEL CENTERS
            # ------------------------------------------------

            longitude, latitude = xy(
                src.transform,
                row_grid,
                col_grid,
                offset="center"
            )


            longitude = np.asarray(
                longitude
            ).flatten()

            latitude = np.asarray(
                latitude
            ).flatten()

            luminosity = data.flatten()


            # ------------------------------------------------
            # REMOVE NODATA
            # ------------------------------------------------

            if src.nodata is not None:

                valid = (
                    luminosity != src.nodata
                )

                luminosity = luminosity[
                    valid
                ]

                latitude = latitude[
                    valid
                ]

                longitude = longitude[
                    valid
                ]


            # ------------------------------------------------
            # REMOVE NaN / INF
            # ------------------------------------------------

            valid = np.isfinite(
                luminosity
            )

            luminosity = luminosity[
                valid
            ]

            latitude = latitude[
                valid
            ]

            longitude = longitude[
                valid
            ]


            # ------------------------------------------------
            # CREATE DATAFRAME
            # ------------------------------------------------

            if len(luminosity) == 0:

                continue


            result = pd.DataFrame({

                "year": int(year),

                "image_name": image_name,

                "sn7_tile": tile_name,

                "latitude": latitude,

                "longitude": longitude,

                "luminosity": luminosity

            })


            # ------------------------------------------------
            # WRITE TO CSV
            # ------------------------------------------------

            result.to_csv(
                OUTPUT_CSV,
                mode="w" if first_write else "a",
                header=first_write,
                index=False
            )


            first_write = False

            records = len(result)

            image_records += records

            total_records += records


            # ------------------------------------------------
            # MEMORY CLEANUP
            # ------------------------------------------------

            del data
            del row_grid
            del col_grid
            del longitude
            del latitude
            del luminosity
            del result


            print(
                f"\rTiles processed: "
                f"{tile_number + 1}/{len(tiles_df)}",
                end=""
            )


        print(
            "\nRecords extracted from this image:",
            image_records
        )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print(
    "Total records:",
    total_records
)

print(
    "Output file:"
)

print(
    OUTPUT_CSV
)

print("=" * 70)