import os
import glob
import rasterio
import pandas as pd
import numpy as np
from rasterio.transform import xy
from rasterio.warp import transform


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

VIIRS_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "VIIRS"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "viirs_csv"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

OUTPUT_CSV = os.path.join(
    OUTPUT_FOLDER,
    "viirs_all_years.csv"
)


# ==========================================================
# FIND ALL TIFF FILES
# ==========================================================

tif_files = []

for year in ["2017", "2018", "2019", "2020"]:

    year_folder = os.path.join(
        VIIRS_FOLDER,
        year
    )

    tif_files.extend(
        glob.glob(
            os.path.join(year_folder, "*.tif")
        )
    )

    tif_files.extend(
        glob.glob(
            os.path.join(year_folder, "*.tiff")
        )
    )


print("=" * 60)
print("VIIRS TIFF TO CSV")
print("=" * 60)

print("Total TIFF files:", len(tif_files))


# ==========================================================
# PROCESS EACH IMAGE
# ==========================================================

first_file = True

for image_number, tif_file in enumerate(
    tif_files,
    start=1
):

    image_name = os.path.basename(tif_file)

    year = os.path.basename(
        os.path.dirname(tif_file)
    )

    print("\n" + "=" * 60)
    print(
        f"[{image_number}/{len(tif_files)}]"
    )
    print("Image:", image_name)
    print("Year:", year)


    try:

        with rasterio.open(tif_file) as src:

            print("Width:", src.width)
            print("Height:", src.height)
            print("Bands:", src.count)
            print("CRS:", src.crs)
            print("NoData:", src.nodata)

            # ------------------------------------------
            # WE ONLY USE BAND 1
            # ------------------------------------------

            band = 1

            total_rows = src.height

            # Process only a small number of rows at once
            CHUNK_ROWS = 100

            rows_written = 0

            for row_start in range(
                0,
                total_rows,
                CHUNK_ROWS
            ):

                row_end = min(
                    row_start + CHUNK_ROWS,
                    total_rows
                )

                height = row_end - row_start

                # --------------------------------------
                # READ SMALL WINDOW
                # --------------------------------------

                window = rasterio.windows.Window(
                    col_off=0,
                    row_off=row_start,
                    width=src.width,
                    height=height
                )

                data = src.read(
                    band,
                    window=window
                )

                # --------------------------------------
                # GET PIXEL COORDINATES
                # --------------------------------------

                rows = np.arange(
                    row_start,
                    row_end
                )

                cols = np.arange(
                    0,
                    src.width
                )

                # Create coordinates only for this chunk
                row_grid, col_grid = np.meshgrid(
                    rows,
                    cols,
                    indexing="ij"
                )

                x, y = xy(
                    src.transform,
                    row_grid,
                    col_grid,
                    offset="center"
                )

                longitude = np.asarray(
                    x
                ).flatten()

                latitude = np.asarray(
                    y
                ).flatten()

                # --------------------------------------
                # CONVERT TO LAT/LON
                # --------------------------------------

                if src.crs is not None:

                    if src.crs.to_string() != "EPSG:4326":

                        longitude, latitude = transform(
                            src.crs,
                            "EPSG:4326",
                            longitude,
                            latitude
                        )

                        longitude = np.asarray(
                            longitude
                        )

                        latitude = np.asarray(
                            latitude
                        )

                # --------------------------------------
                # FLATTEN LUMINOSITY
                # --------------------------------------

                luminosity = data.flatten()

                # --------------------------------------
                # REMOVE NODATA
                # --------------------------------------

                if src.nodata is not None:

                    valid = (
                        luminosity != src.nodata
                    )

                    luminosity = luminosity[valid]
                    latitude = latitude[valid]
                    longitude = longitude[valid]

                # --------------------------------------
                # REMOVE NaN
                # --------------------------------------

                valid = np.isfinite(
                    luminosity
                )

                luminosity = luminosity[valid]
                latitude = latitude[valid]
                longitude = longitude[valid]

                # --------------------------------------
                # OPTIONAL:
                # REMOVE ZERO LUMINOSITY
                # --------------------------------------

                # Uncomment this if you only want
                # locations with actual night-light values.

                # valid = luminosity > 0

                # luminosity = luminosity[valid]
                # latitude = latitude[valid]
                # longitude = longitude[valid]

                # --------------------------------------
                # CREATE DATAFRAME
                # --------------------------------------

                df = pd.DataFrame({

                    "year": year,

                    "image_name": image_name,

                    "latitude": latitude,

                    "longitude": longitude,

                    "luminosity": luminosity

                })

                # --------------------------------------
                # WRITE DIRECTLY TO CSV
                # --------------------------------------

                df.to_csv(
                    OUTPUT_CSV,
                    mode="w" if first_file else "a",
                    header=first_file,
                    index=False
                )

                first_file = False

                rows_written += len(df)

                # --------------------------------------
                # CLEAN MEMORY
                # --------------------------------------

                del data
                del row_grid
                del col_grid
                del latitude
                del longitude
                del luminosity
                del df

                print(
                    f"\rRows processed: {row_end}/{total_rows}",
                    end=""
                )

            print(
                f"\nRows written for image: {rows_written}"
            )

    except Exception as e:

        print("\nERROR:")
        print(tif_file)
        print(e)


# ==========================================================
# FINISHED
# ==========================================================

print("\n")
print("=" * 60)
print("CONVERSION FINISHED")
print("=" * 60)

print("CSV:")
print(OUTPUT_CSV)