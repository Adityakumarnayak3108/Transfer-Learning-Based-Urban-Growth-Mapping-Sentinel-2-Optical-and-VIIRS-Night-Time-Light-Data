import os
import pandas as pd


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CSV_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "SN7_buildings_train_csvs",
    "csvs",
    "sn7_train_ground_truth_pix.csv"
)


df = pd.read_csv(CSV_FILE)


print("=" * 70)
print("SN7 GROUND TRUTH INSPECTION")
print("=" * 70)

print("\nFile:")
print(CSV_FILE)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nUnique filenames:")
print(df["filename"].nunique())

print("\nUnique building IDs:")
print(df["id"].nunique())

print("\nFirst 5 records:")
print(
    df.head().to_string(index=False)
)

print("\nGeometry examples:")

for i, geometry in enumerate(
    df["geometry"].head(5)
):
    print(f"\n{i + 1}:")
    print(geometry)

print("\nMissing values:")
print(df.isnull().sum())

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)