import os
import pandas as pd
import rasterio

# ============================================================
# PATHS
# ============================================================

GROUND_TRUTH = r"C:\urban development kaggle 3\dataset\SN7_buildings_train_csvs\csvs\sn7_train_ground_truth_pix.csv"

# CHANGE THIS to the folder containing your SN7 .tif images
IMAGE_FOLDER = r"C:\urban development kaggle 3\dataset\SN7_buildings_train_csvs"

# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("=" * 70)
print("SN7 TIFF IMAGE VALIDATION")
print("=" * 70)

df = pd.read_csv(GROUND_TRUTH)

print("\nGround truth loaded.")
print("Rows:", len(df))
print("Unique filenames:", df["filename"].nunique())

# ============================================================
# FIND TIFF FILES
# ============================================================

tif_files = []

for root, dirs, files in os.walk(IMAGE_FOLDER):
    for file in files:
        if file.lower().endswith((".tif", ".tiff")):
            tif_files.append(os.path.join(root, file))

print("\nTotal TIFF files found:", len(tif_files))

# ============================================================
# TIFF FILENAMES
# ============================================================

tif_names = {
    os.path.splitext(os.path.basename(path))[0]
    for path in tif_files
}

gt_names = set(df["filename"].astype(str))

# ============================================================
# MATCHING
# ============================================================

matched = gt_names.intersection(tif_names)
missing_images = gt_names - tif_names
extra_images = tif_names - gt_names

print("\n" + "=" * 70)
print("FILENAME MATCHING")
print("=" * 70)

print("Ground-truth filenames :", len(gt_names))
print("TIFF filenames         :", len(tif_names))
print("Matched filenames      :", len(matched))
print("Missing TIFF images    :", len(missing_images))
print("Extra TIFF images      :", len(extra_images))

# ============================================================
# SHOW MISSING FILES
# ============================================================

if missing_images:
    print("\nFirst 20 missing TIFF images:")
    for name in list(missing_images)[:20]:
        print(" -", name)
else:
    print("\n✓ No ground-truth images are missing.")

# ============================================================
# INSPECT TIFF IMAGES
# ============================================================

print("\n" + "=" * 70)
print("TIFF IMAGE INSPECTION")
print("=" * 70)

if len(tif_files) == 0:
    print("ERROR: No TIFF files found.")
    raise SystemExit

sample_count = min(10, len(tif_files))

image_info = []

for path in tif_files[:sample_count]:

    try:
        with rasterio.open(path) as src:

            width = src.width
            height = src.height
            bands = src.count
            dtype = src.dtypes[0]

            image_info.append({
                "filename": os.path.splitext(os.path.basename(path))[0],
                "width": width,
                "height": height,
                "bands": bands,
                "dtype": dtype
            })

            print("\nFile:", os.path.basename(path))
            print("  Width  :", width)
            print("  Height :", height)
            print("  Bands  :", bands)
            print("  Dtype  :", dtype)

    except Exception as e:
        print("\nERROR:", os.path.basename(path))
        print(e)

# ============================================================
# CHECK POLYGON COORDINATES
# ============================================================

print("\n" + "=" * 70)
print("POLYGON COORDINATE VALIDATION")
print("=" * 70)

# Get image dimensions from first readable TIFF
first_image = None

for path in tif_files:
    try:
        with rasterio.open(path) as src:
            first_image = (src.width, src.height)
        break
    except:
        pass

if first_image:

    width, height = first_image

    print("\nUsing sample image dimensions:")
    print("Width :", width)
    print("Height:", height)

    # Extract coordinates from WKT polygon strings
    from shapely import wkt

    outside_count = 0
    invalid_count = 0

    sample_df = df.head(10000)

    for geometry in sample_df["geometry"]:

        try:
            polygon = wkt.loads(geometry)

            minx, miny, maxx, maxy = polygon.bounds

            if (
                minx < 0 or
                miny < 0 or
                maxx > width or
                maxy > height
            ):
                outside_count += 1

        except:
            invalid_count += 1

    print("\nChecked:", len(sample_df), "polygons")
    print("Polygons outside image:", outside_count)
    print("Invalid geometries:", invalid_count)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print("Ground truth rows       :", len(df))
print("Ground truth filenames  :", len(gt_names))
print("TIFF files              :", len(tif_files))
print("Matched filenames       :", len(matched))
print("Missing TIFF images     :", len(missing_images))
print("Extra TIFF images       :", len(extra_images))

print("\nINSPECTION COMPLETE")
print("=" * 70)