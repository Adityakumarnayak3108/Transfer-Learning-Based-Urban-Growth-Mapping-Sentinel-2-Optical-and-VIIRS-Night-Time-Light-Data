import os
import rasterio
from rasterio.transform import Affine
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def extract_geospatial_data_from_tif(tif_file_path):
    """
    Extract longitude, latitude, and pixel values from a GeoTIFF file.
    
    Args:
        tif_file_path: Path to the .tif file
    
    Returns:
        DataFrame with columns: longitude, latitude, luminous_value
    """
    try:
        with rasterio.open(tif_file_path) as src:
            # Read the raster data
            data = src.read(1)  # Read first band (for single-band images)
            
            # Get the geospatial transform
            transform = src.transform
            
            # Get image dimensions
            height, width = data.shape
            
            # Create lists to store coordinates and values
            longitudes = []
            latitudes = []
            luminous_values = []
            
            # Iterate through all pixels
            for row in range(height):
                for col in range(width):
                    # Get pixel value (luminous value)
                    pixel_value = data[row, col]
                    
                    # Skip NoData values (typically 0 or negative for satellite data)
                    if pixel_value <= 0:
                        continue
                    
                    # Calculate longitude and latitude from pixel coordinates
                    # Using the affine transform
                    lon = transform.c + (col * transform.a)  # x coordinate
                    lat = transform.f + (row * transform.e)  # y coordinate
                    
                    longitudes.append(lon)
                    latitudes.append(lat)
                    luminous_values.append(pixel_value)
            
            # Create DataFrame
            df = pd.DataFrame({
                'longitude': longitudes,
                'latitude': latitudes,
                'luminous_value': luminous_values
            })
            
            return df
    
    except Exception as e:
        print(f"Error processing {tif_file_path}: {str(e)}")
        return None


def process_all_folders(base_dataset_path, output_base_path):
    """
    Process all folders in the dataset and convert TIF files to CSV.
    
    Args:
        base_dataset_path: Path to the dataset root folder
        output_base_path: Path where CSV files will be saved
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base_path, exist_ok=True)
    
    # Get all tile folders (assuming structure: base_dataset_path/SN7_buildings_train/train/[tile_folders])
    train_path = os.path.join(base_dataset_path, 'SN7_buildings_train', 'train')
    
    if not os.path.exists(train_path):
        print(f"Train path not found: {train_path}")
        return
    
    # Get all tile folders
    tile_folders = [f for f in os.listdir(train_path) 
                    if os.path.isdir(os.path.join(train_path, f))]
    
    print(f"Found {len(tile_folders)} folders to process")
    
    total_files_processed = 0
    
    # Process each tile folder
    for idx, tile_folder in enumerate(sorted(tile_folders), 1):
        tile_path = os.path.join(train_path, tile_folder)
        images_path = os.path.join(tile_path, 'images')
        
        if not os.path.exists(images_path):
            print(f"[{idx}/{len(tile_folders)}] Skipping {tile_folder} - no images folder")
            continue
        
        # Get all TIF files in this folder
        tif_files = sorted(glob.glob(os.path.join(images_path, '*.tif')))
        
        if not tif_files:
            print(f"[{idx}/{len(tile_folders)}] No TIF files in {tile_folder}")
            continue
        
        print(f"[{idx}/{len(tile_folders)}] Processing {tile_folder} - Found {len(tif_files)} TIF files")
        
        # Create output folder for this tile
        tile_output_path = os.path.join(output_base_path, tile_folder)
        os.makedirs(tile_output_path, exist_ok=True)
        
        # Process each TIF file
        for tif_file in tif_files:
            tif_filename = os.path.basename(tif_file)
            csv_filename = tif_filename.replace('.tif', '.csv')
            csv_output_path = os.path.join(tile_output_path, csv_filename)
            
            print(f"  Converting: {tif_filename}...", end=' ')
            
            # Extract geospatial data
            df = extract_geospatial_data_from_tif(tif_file)
            
            if df is not None and len(df) > 0:
                # Save to CSV
                df.to_csv(csv_output_path, index=False)
                print(f"✓ Saved ({len(df)} records)")
                total_files_processed += 1
            else:
                print("✗ Failed or empty")
    
    print(f"\n{'='*60}")
    print(f"Completed! Total files processed: {total_files_processed}")
    print(f"Output saved to: {output_base_path}")


def process_single_tile(tile_path, output_path=None):
    """
    Process a single tile folder and convert all TIF files to CSV.
    
    Args:
        tile_path: Path to the specific tile folder
        output_path: Optional output path (defaults to tile_path/csv_output)
    """
    
    if output_path is None:
        output_path = os.path.join(tile_path, 'csv_output')
    
    os.makedirs(output_path, exist_ok=True)
    
    images_path = os.path.join(tile_path, 'images')
    
    if not os.path.exists(images_path):
        print(f"Images folder not found: {images_path}")
        return
    
    # Get all TIF files
    tif_files = sorted(glob.glob(os.path.join(images_path, '*.tif')))
    
    if not tif_files:
        print(f"No TIF files found in {images_path}")
        return
    
    print(f"Processing {len(tif_files)} TIF files...")
    
    # Process each TIF file
    for tif_file in tif_files:
        tif_filename = os.path.basename(tif_file)
        csv_filename = tif_filename.replace('.tif', '.csv')
        csv_output_path = os.path.join(output_path, csv_filename)
        
        print(f"Converting: {tif_filename}...", end=' ')
        
        # Extract geospatial data
        df = extract_geospatial_data_from_tif(tif_file)
        
        if df is not None and len(df) > 0:
            df.to_csv(csv_output_path, index=False)
            print(f"✓ Saved ({len(df)} records)")
        else:
            print("✗ Failed or empty")
    
    print(f"\nCSV files saved to: {output_path}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # OPTION 1: Process all 60 folders at once
    # Modify these paths to match your system
    base_dataset_path = r"C:\urban development kaggle 3\dataset"  # Change this path
    output_base_path = r"C:\urban development kaggle 3\csv_output"  # Change this path
    
    # Uncomment to process all folders:
    # process_all_folders(base_dataset_path, output_base_path)
    
    # OPTION 2: Process a single tile folder
    # single_tile_path = r"C:\urban development kaggle 3\dataset\SN7_buildings_train\train\L15-0331E-1257N_1327_3160_13"
    # process_single_tile(single_tile_path)
    
    print("Script loaded. Use one of the options above to process your data.")
    print("\nFor all folders:")
    print("  process_all_folders(base_dataset_path, output_base_path)")
    print("\nFor single tile:")
    print("  process_single_tile(tile_path)")
