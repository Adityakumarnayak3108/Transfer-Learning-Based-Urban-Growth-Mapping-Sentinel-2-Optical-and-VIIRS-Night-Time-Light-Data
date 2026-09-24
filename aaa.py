#!/usr/bin/env python3
"""
VIIRS TIF to CSV Converter
Converts satellite imagery to CSV with latitude, longitude, and luminous values
"""

import os
import pandas as pd
import rasterio
import glob
import sys

# ============================================================
# ⚠️ IMPORTANT: UPDATE THIS PATH WITH YOUR DATASET LOCATION
# ============================================================
BASE_PATH = r'C:\urban development kaggle 3\dataset'

OUTPUT_CSV = 'viirs_combined_data.csv'


def extract_tif_data(tif_file_path):
    """Extract latitude, longitude, and luminous values from one TIF file"""

    filename = os.path.basename(tif_file_path)
    print(f"    • {filename[:55]:<55}", end=" ", flush=True)

    try:
        with rasterio.open(tif_file_path) as src:

            # Read pixel values from first band
            data = src.read(1)

            # Get geographic transformation
            transform = src.transform

            # Get image size
            height, width = data.shape

            # List to collect all pixel data
            pixels_data = []

            # Loop through each pixel
            for row in range(height):
                for col in range(width):

                    # Get brightness value
                    luminous_value = float(data[row, col])

                    # Skip empty pixels
                    if luminous_value <= 0:
                        continue

                    # Calculate latitude and longitude
                    longitude = transform.c + (col + 0.5) * transform.a
                    latitude = transform.f + (row + 0.5) * transform.e

                    pixels_data.append({
                        'filename': filename,
                        'latitude': latitude,
                        'longitude': longitude,
                        'luminous_value': luminous_value
                    })

            pixels_count = len(pixels_data)

            print(f"✓ {pixels_count:>7,} pixels")

            return pixels_data

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return []


def main():
    """Main function"""

    print("\n" + "=" * 80)
    print("VIIRS GEOTIFF TO CSV CONVERTER - 2017-2020 Data")
    print("=" * 80)

    # Check if path exists
    if not os.path.exists(BASE_PATH):
        print("\n❌ ERROR: Path does not exist!")
        print(f"   {BASE_PATH}")
        print("\n⚠️ Please update BASE_PATH at the top of the script!")
        sys.exit(1)

    print(f"\n📁 Loading data from: {BASE_PATH}\n")

    years = ['2017', '2018', '2019', '2020']

    all_pixels = []

    # Process each year
    for year in years:

        year_path = os.path.join(BASE_PATH, 'VIIRS', year)

        # Check if year folder exists
        if not os.path.exists(year_path):
            print(f"⚠️ {year}: Folder not found")
            continue

        # Find all TIF files
        tif_files = glob.glob(
            os.path.join(year_path, '*.tif')
        )

        if not tif_files:
            print(f"⚠️ {year}: No TIF files found")
            continue

        # Print year information
        print(f"📅 {year} - Processing {len(tif_files)} file(s):")

        # Process each TIF file
        for tif_file in sorted(tif_files):

            pixels = extract_tif_data(tif_file)

            if pixels:
                all_pixels.extend(pixels)

        print()

    print("-" * 80)

    # Check if we got data
    if not all_pixels:
        print("\n❌ FAILED: No data was extracted!")
        print("   Check if your TIF files exist and contain valid data")
        sys.exit(1)

    # Create DataFrame
    print(
        f"\n⏳ Converting {len(all_pixels):,} pixels to CSV...",
        end=" ",
        flush=True
    )

    df = pd.DataFrame(all_pixels)

    df = df[
        [
            'filename',
            'latitude',
            'longitude',
            'luminous_value'
        ]
    ]

    print("✓")

    # Save CSV
    print(
        f"⏳ Saving to {OUTPUT_CSV}...",
        end=" ",
        flush=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("✓")

    # Calculate file size
    file_size_mb = os.path.getsize(
        OUTPUT_CSV
    ) / (1024 * 1024)

    print("\n" + "=" * 80)
    print("✅ SUCCESS! Conversion Complete")
    print("=" * 80)

    print("\n📊 Results:")

    print(f"   Total pixels extracted: {len(df):,}")
    print(f"   CSV file: {OUTPUT_CSV}")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Unique files: {df['filename'].nunique()}")

    print("\n📍 Geographic bounds:")

    print(
        f"   Latitude:  "
        f"{df['latitude'].min():.6f} → "
        f"{df['latitude'].max():.6f}"
    )

    print(
        f"   Longitude: "
        f"{df['longitude'].min():.6f} → "
        f"{df['longitude'].max():.6f}"
    )

    print("\n💡 Luminous values:")

    print(
        f"   Min:  "
        f"{df['luminous_value'].min():.6f}"
    )

    print(
        f"   Max:  "
        f"{df['luminous_value'].max():.6f}"
    )

    print(
        f"   Mean: "
        f"{df['luminous_value'].mean():.6f}"
    )

    print(
        f"   Std:  "
        f"{df['luminous_value'].std():.6f}"
    )

    print("\n📋 First 10 rows:")
    print("-" * 80)

    print(
        df.head(10).to_string(index=False)
    )

    print("\n" + "=" * 80)
    print(
        f"✓ Your CSV file '{OUTPUT_CSV}' is ready to use!"
    )
    print("=" * 80 + "\n")

    return df


if __name__ == "__main__":
    main()