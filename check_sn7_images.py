import os
import glob


# Project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# SN7 training image folder
SN7_FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "SN7_buildings_train"
)


print("=" * 60)
print("SN7 IMAGE FILES")
print("=" * 60)

print("Folder:")
print(SN7_FOLDER)


# Find images
extensions = [
    "*.tif",
    "*.tiff",
    "*.png",
    "*.jpg",
    "*.jpeg"
]

files = []

for extension in extensions:

    files.extend(
        glob.glob(
            os.path.join(
                SN7_FOLDER,
                "**",
                extension
            ),
            recursive=True
        )
    )


print("\nTotal image files found:", len(files))


print("\nFirst 20 files:")

for file in files[:20]:

    print(file)