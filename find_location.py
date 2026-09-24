import os
import rasterio
from pyproj import Transformer

# Path to SN7_buildings_train
dataset_path = r"dataset\SN7_buildings_train\train"

transformer = Transformer.from_crs(
    "EPSG:3857",
    "EPSG:4326",
    always_xy=True
)

print("="*100)

for aoi in sorted(os.listdir(dataset_path)):

    aoi_path = os.path.join(dataset_path, aoi)

    if not os.path.isdir(aoi_path):
        continue

    image_folder = os.path.join(aoi_path, "images")

    if not os.path.exists(image_folder):
        continue

    tif_files = sorted([
        f for f in os.listdir(image_folder)
        if f.endswith(".tif")
    ])

    if len(tif_files) == 0:
        continue

    # Only use the first monthly image because all months cover the same area
    img = os.path.join(image_folder, tif_files[0])

    with rasterio.open(img) as src:
        bounds = src.bounds

    lon1, lat1 = transformer.transform(bounds.left, bounds.bottom)
    lon2, lat2 = transformer.transform(bounds.right, bounds.top)

    center_lat = (lat1 + lat2) / 2
    center_lon = (lon1 + lon2) / 2

    print(f"AOI : {aoi}")
    print(f"Center : {center_lat:.6f}, {center_lon:.6f}")
    print("-"*100)