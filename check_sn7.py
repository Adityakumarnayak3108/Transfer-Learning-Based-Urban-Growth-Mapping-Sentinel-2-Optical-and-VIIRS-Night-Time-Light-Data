import os
import glob
import pandas as pd


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset"
)


# ==========================================================
# FIND CSV FILES
# ==========================================================

files = []

files.extend(
    glob.glob(
        os.path.join(
            DATASET_FOLDER,
            "**",
            "*.csv"
        ),
        recursive=True
    )
)

print("=" * 60)
print("CSV FILES FOUND")
print("=" * 60)

for i, file in enumerate(files, start=1):

    print(f"{i}. {file}")


# ==========================================================
# SHOW INFORMATION
# ==========================================================

for file in files:

    print("\n")
    print("=" * 60)
    print("FILE:")
    print(file)
    print("=" * 60)

    try:

        df = pd.read_csv(
            file,
            nrows=5
        )

        print("\nColumns:")

        for column in df.columns:
            print(" -", column)

        print("\nFirst 5 rows:")
        print(df.head())

    except Exception as e:

        print("Could not read file:")
        print(e)