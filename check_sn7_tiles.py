import os
import pandas as pd
import re


# ==========================================================
# FILE PATH
# ==========================================================

csv_file = r"C:\urban development kaggle 3\dataset\SN7_buildings_train_csvs\csvs\sn7_train_ground_truth_pix.csv"


# ==========================================================
# READ CSV
# ==========================================================

df = pd.read_csv(csv_file)

print("=" * 60)
print("SN7 TILE INFORMATION")
print("=" * 60)

print("Total building records:", len(df))


# ==========================================================
# SHOW UNIQUE FILENAMES
# ==========================================================

print("\nNumber of unique image/tile filenames:")

print(
    df["filename"].nunique()
)


print("\nFirst 20 unique filenames:")

for filename in df["filename"].unique()[:20]:

    print(filename)


# ==========================================================
# EXTRACT TILE LOCATION
# ==========================================================

pattern = r"L15-([0-9]+)E-([0-9]+)N"

tile_locations = []

for filename in df["filename"].unique():

    match = re.search(
        pattern,
        filename
    )

    if match:

        longitude_tile = int(
            match.group(1)
        )

        latitude_tile = int(
            match.group(2)
        )

        tile_locations.append({

            "filename": filename,

            "longitude_tile": longitude_tile,

            "latitude_tile": latitude_tile

        })


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

tile_df = pd.DataFrame(
    tile_locations
)

print("\n" + "=" * 60)
print("EXTRACTED TILE LOCATIONS")
print("=" * 60)

print(tile_df.to_string(index=False))


# ==========================================================
# SUMMARY
# ==========================================================

if len(tile_df) > 0:

    print("\n" + "=" * 60)
    print("GEOGRAPHIC RANGE")
    print("=" * 60)

    print(
        "Minimum tile longitude:",
        tile_df["longitude_tile"].min()
    )

    print(
        "Maximum tile longitude:",
        tile_df["longitude_tile"].max()
    )

    print(
        "Minimum tile latitude:",
        tile_df["latitude_tile"].min()
    )

    print(
        "Maximum tile latitude:",
        tile_df["latitude_tile"].max()
    )

else:

    print(
        "\nCould not extract tile coordinates."
    )