import os
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CSV_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "viirs_csv",
    "viirs_sn7_all_years.csv"
)


# ============================================================
# READ CSV
# ============================================================

print("=" * 70)
print("VIIRS CSV VALIDATION")
print("=" * 70)

print("\nReading:")
print(CSV_FILE)

df = pd.read_csv(CSV_FILE)

print("\nCSV loaded successfully.")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print(
    "Total records:",
    len(df)
)

print(
    "Total columns:",
    len(df.columns)
)

print("\nColumns:")

for column in df.columns:
    print(" -", column)


# ============================================================
# DATA TYPES
# ============================================================

print("\n")
print("=" * 70)
print("DATA TYPES")
print("=" * 70)

print(
    df.dtypes
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n")
print("=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)


# ============================================================
# YEAR SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("RECORDS BY YEAR")
print("=" * 70)

year_counts = (
    df["year"]
    .value_counts()
    .sort_index()
)

print(year_counts)


# ============================================================
# IMAGE SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("RECORDS BY IMAGE")
print("=" * 70)

image_counts = (
    df["image_name"]
    .value_counts()
    .sort_index()
)

print(image_counts)


# ============================================================
# TILE SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SN7 TILE SUMMARY")
print("=" * 70)

print(
    "Unique SN7 tiles:",
    df["sn7_tile"].nunique()
)


# ============================================================
# LATITUDE INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("LATITUDE")
print("=" * 70)

print(
    "Minimum latitude:",
    df["latitude"].min()
)

print(
    "Maximum latitude:",
    df["latitude"].max()
)

print(
    "Mean latitude:",
    df["latitude"].mean()
)


# ============================================================
# LONGITUDE INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("LONGITUDE")
print("=" * 70)

print(
    "Minimum longitude:",
    df["longitude"].min()
)

print(
    "Maximum longitude:",
    df["longitude"].max()
)

print(
    "Mean longitude:",
    df["longitude"].mean()
)


# ============================================================
# LUMINOSITY INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("LUMINOSITY")
print("=" * 70)

print(
    "Minimum luminosity:",
    df["luminosity"].min()
)

print(
    "Maximum luminosity:",
    df["luminosity"].max()
)

print(
    "Mean luminosity:",
    df["luminosity"].mean()
)

print(
    "Median luminosity:",
    df["luminosity"].median()
)

print(
    "Standard deviation:",
    df["luminosity"].std()
)


# ============================================================
# ZERO LUMINOSITY
# ============================================================

zero_count = (
    df["luminosity"] == 0
).sum()

zero_percentage = (
    zero_count / len(df)
) * 100

print("\n")
print("=" * 70)
print("ZERO LUMINOSITY")
print("=" * 70)

print(
    "Zero luminosity records:",
    zero_count
)

print(
    "Percentage:",
    round(zero_percentage, 2),
    "%"
)


# ============================================================
# NEGATIVE LUMINOSITY
# ============================================================

negative_count = (
    df["luminosity"] < 0
).sum()

print("\n")
print("=" * 70)
print("NEGATIVE LUMINOSITY")
print("=" * 70)

print(
    "Negative luminosity records:",
    negative_count
)


# ============================================================
# INFINITE VALUES
# ============================================================

numeric_columns = [
    "latitude",
    "longitude",
    "luminosity"
]

infinite_counts = np.isinf(
    df[numeric_columns]
).sum()

print("\n")
print("=" * 70)
print("INFINITE VALUES")
print("=" * 70)

print(infinite_counts)


# ============================================================
# DUPLICATE RECORDS
# ============================================================

duplicate_count = (
    df.duplicated().sum()
)

print("\n")
print("=" * 70)
print("DUPLICATES")
print("=" * 70)

print(
    "Duplicate rows:",
    duplicate_count
)


# ============================================================
# DUPLICATE LAT/LON/YEAR
# ============================================================

duplicate_location_count = (
    df.duplicated(
        subset=[
            "year",
            "latitude",
            "longitude"
        ]
    ).sum()
)

print(
    "Duplicate year + latitude + longitude:",
    duplicate_location_count
)


# ============================================================
# SAMPLE DATA
# ============================================================

print("\n")
print("=" * 70)
print("FIRST 10 RECORDS")
print("=" * 70)

print(
    df.head(10).to_string(
        index=False
    )
)


# ============================================================
# RANDOM SAMPLE
# ============================================================

print("\n")
print("=" * 70)
print("RANDOM SAMPLE")
print("=" * 70)

print(
    df.sample(
        min(10, len(df)),
        random_state=42
    ).to_string(
        index=False
    )
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(
    "File:",
    CSV_FILE
)

print(
    "Total records:",
    len(df)
)

print(
    "Unique years:",
    df["year"].nunique()
)

print(
    "Unique images:",
    df["image_name"].nunique()
)

print(
    "Unique SN7 tiles:",
    df["sn7_tile"].nunique()
)

print("=" * 70)