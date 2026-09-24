import os

ROOT = r"C:\urban development kaggle 3"

print("Searching for TIFF files...")
print("=" * 70)

count = 0

for root, dirs, files in os.walk(ROOT):
    for file in files:
        if file.lower().endswith((".tif", ".tiff")):
            full_path = os.path.join(root, file)

            print(full_path)

            count += 1

            if count >= 20:
                break

    if count >= 20:
        break

print("\n" + "=" * 70)
print("First TIFF files found:", count)
print("=" * 70)