"""
Convert ALL TIF files from 60 folders into ONE SINGLE CSV file
Author: Geospatial Data Processor
"""

import os
import rasterio
import pandas as pd
import numpy as np
import glob
from datetime import datetime

class SingleCSVConverter:
    """
    Converts all TIF files from 60 folders into a single combined CSV file.
    """
    
    def __init__(self, skip_zero_values=True, skip_negative_values=True):
        """
        Initialize converter.
        
        Args:
            skip_zero_values: Skip pixels with value 0 (often NoData)
            skip_negative_values: Skip negative pixel values
        """
        self.skip_zero_values = skip_zero_values
        self.skip_negative_values = skip_negative_values
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'total_records': 0
        }
    
    def extract_geospatial_data_from_tif(self, tif_file_path, folder_name=None, file_name=None):
        """
        Extract longitude, latitude, and pixel values from a GeoTIFF file.
        
        Args:
            tif_file_path: Path to the .tif file
            folder_name: Name of the folder (for identification)
            file_name: Name of the file (for identification)
        
        Returns:
            DataFrame with columns: longitude, latitude, luminous_value, folder, filename
        """
        try:
            with rasterio.open(tif_file_path) as src:
                # Read the raster data
                data = src.read(1)  # Read first band
                
                # Get the geospatial transform
                transform = src.transform
                
                # Get image dimensions
                height, width = data.shape
                
                # Create lists to store coordinates and values
                records = []
                
                # Iterate through all pixels
                for row in range(height):
                    for col in range(width):
                        # Get pixel value (luminous value)
                        pixel_value = float(data[row, col])
                        
                        # Apply filters
                        if self.skip_zero_values and pixel_value == 0:
                            continue
                        if self.skip_negative_values and pixel_value < 0:
                            continue
                        
                        # Calculate longitude and latitude from pixel coordinates
                        lon = transform.c + (col * transform.a)  # x coordinate
                        lat = transform.f + (row * transform.e)  # y coordinate
                        
                        records.append({
                            'longitude': lon,
                            'latitude': lat,
                            'luminous_value': pixel_value,
                            'folder': folder_name,
                            'filename': file_name
                        })
                
                # Create DataFrame
                if len(records) > 0:
                    df = pd.DataFrame(records)
                    return df
                else:
                    return None
        
        except Exception as e:
            print(f"Error processing {tif_file_path}: {str(e)}")
            return None
    
    def process_all_to_single_csv(self, base_dataset_path, output_csv_path):
        """
        Process all 60 folders and save everything to ONE single CSV file.
        
        Args:
            base_dataset_path: Path to the dataset root folder
            output_csv_path: Path to the output CSV file (e.g., "all_data.csv")
        """
        
        print(f"{'='*80}")
        print(f"Converting ALL TIF files from 60 folders into ONE CSV")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Find the train directory
        train_path = os.path.join(base_dataset_path, 'SN7_buildings_train', 'train')
        
        if not os.path.exists(train_path):
            print(f"❌ Train path not found: {train_path}")
            return
        
        # Get all tile folders
        tile_folders = sorted([f for f in os.listdir(train_path) 
                             if os.path.isdir(os.path.join(train_path, f))])
        
        print(f"📁 Found {len(tile_folders)} folders to process")
        print(f"📊 Collecting data...\n")
        
        # List to store all DataFrames
        all_dataframes = []
        
        # Process each tile folder
        for idx, tile_folder in enumerate(tile_folders, 1):
            tile_path = os.path.join(train_path, tile_folder)
            images_path = os.path.join(tile_path, 'images')
            
            if not os.path.exists(images_path):
                print(f"[{idx:3d}/{len(tile_folders)}] ⏭️  {tile_folder:<50} - No images folder")
                continue
            
            # Get all TIF files in this folder
            tif_files = sorted(glob.glob(os.path.join(images_path, '*.tif')))
            
            if not tif_files:
                print(f"[{idx:3d}/{len(tile_folders)}] ⏭️  {tile_folder:<50} - No TIF files")
                continue
            
            print(f"[{idx:3d}/{len(tile_folders)}] 📍 {tile_folder:<50} - {len(tif_files)} TIF files", end='')
            
            folder_records = 0
            
            # Process each TIF file
            for tif_file in tif_files:
                tif_filename = os.path.basename(tif_file)
                
                # Extract geospatial data
                df = self.extract_geospatial_data_from_tif(
                    tif_file, 
                    folder_name=tile_folder,
                    file_name=tif_filename
                )
                
                if df is not None and len(df) > 0:
                    all_dataframes.append(df)
                    self.stats['successful'] += 1
                    folder_records += len(df)
                    self.stats['total_records'] += len(df)
                else:
                    self.stats['failed'] += 1
            
            print(f" ✓ Extracted {folder_records:,} records")
        
        print(f"\n{'='*80}")
        print(f"Combining data from {self.stats['successful']} files...")
        print(f"Total records to write: {self.stats['total_records']:,}")
        print(f"{'='*80}\n")
        
        # Combine all DataFrames into one
        if all_dataframes:
            print("🔗 Merging all dataframes...")
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            print(f"📝 Writing to CSV file: {output_csv_path}")
            combined_df.to_csv(output_csv_path, index=False)
            
            print(f"\n{'='*80}")
            print(f"✅ SUCCESS!")
            print(f"{'='*80}")
            print(f"Output file: {output_csv_path}")
            print(f"Total rows: {len(combined_df):,}")
            print(f"Total columns: {len(combined_df.columns)}")
            print(f"\nColumn names: {list(combined_df.columns)}")
            print(f"\nData preview:")
            print(combined_df.head(10))
            print(f"\nData statistics:")
            print(combined_df.describe())
            print(f"{'='*80}\n")
            
            return combined_df
        else:
            print("❌ No data was extracted from any TIF files!")
            return None
    
    def process_with_progress_save(self, base_dataset_path, output_csv_path, batch_size=1000):
        """
        Process all 60 folders with periodic saving (good for memory management with large datasets).
        Saves progress every batch_size records.
        
        Args:
            base_dataset_path: Path to the dataset root folder
            output_csv_path: Path to the output CSV file
            batch_size: Number of records to accumulate before saving
        """
        
        print(f"{'='*80}")
        print(f"Converting ALL TIF files with BATCH SAVING")
        print(f"Batch size: {batch_size:,} records")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Find the train directory
        train_path = os.path.join(base_dataset_path, 'SN7_buildings_train', 'train')
        
        if not os.path.exists(train_path):
            print(f"❌ Train path not found: {train_path}")
            return
        
        # Get all tile folders
        tile_folders = sorted([f for f in os.listdir(train_path) 
                             if os.path.isdir(os.path.join(train_path, f))])
        
        print(f"📁 Found {len(tile_folders)} folders to process\n")
        
        # Buffer to store records
        buffer = []
        file_exists = False
        
        # Process each tile folder
        for idx, tile_folder in enumerate(tile_folders, 1):
            tile_path = os.path.join(train_path, tile_folder)
            images_path = os.path.join(tile_path, 'images')
            
            if not os.path.exists(images_path):
                continue
            
            # Get all TIF files in this folder
            tif_files = sorted(glob.glob(os.path.join(images_path, '*.tif')))
            
            if not tif_files:
                continue
            
            print(f"[{idx:3d}/{len(tile_folders)}] Processing {tile_folder:<50}", end='')
            
            # Process each TIF file
            for tif_file in tif_files:
                tif_filename = os.path.basename(tif_file)
                
                # Extract geospatial data
                df = self.extract_geospatial_data_from_tif(
                    tif_file, 
                    folder_name=tile_folder,
                    file_name=tif_filename
                )
                
                if df is not None and len(df) > 0:
                    buffer.extend(df.to_dict('records'))
                    self.stats['successful'] += 1
                    self.stats['total_records'] += len(df)
                else:
                    self.stats['failed'] += 1
            
            # Save batch if buffer reaches batch_size
            if len(buffer) >= batch_size:
                print(f" - Saving {len(buffer):,} records...", end=' ')
                batch_df = pd.DataFrame(buffer)
                
                if file_exists:
                    batch_df.to_csv(output_csv_path, mode='a', header=False, index=False)
                else:
                    batch_df.to_csv(output_csv_path, mode='w', header=True, index=False)
                    file_exists = True
                
                buffer = []
                print(f"✓")
            else:
                print()
        
        # Save remaining records
        if buffer:
            print(f"\nSaving final {len(buffer):,} records...")
            batch_df = pd.DataFrame(buffer)
            
            if file_exists:
                batch_df.to_csv(output_csv_path, mode='a', header=False, index=False)
            else:
                batch_df.to_csv(output_csv_path, mode='w', header=True, index=False)
        
        print(f"\n{'='*80}")
        print(f"✅ SUCCESS!")
        print(f"{'='*80}")
        print(f"Output file: {output_csv_path}")
        print(f"Total records written: {self.stats['total_records']:,}")
        print(f"Files processed successfully: {self.stats['successful']}")
        print(f"Files failed: {self.stats['failed']}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
    
    def print_summary(self):
        """Print processing summary statistics."""
        print(f"{'='*80}")
        print(f"Processing Summary")
        print(f"{'='*80}")
        print(f"Total files processed: {self.stats['successful'] + self.stats['failed']}")
        print(f"✓ Successful: {self.stats['successful']}")
        print(f"✗ Failed: {self.stats['failed']}")
        print(f"📊 Total records extracted: {self.stats['total_records']:,}")
        print(f"{'='*80}\n")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # ✅ MODIFY THESE PATHS
    BASE_DATASET_PATH = r"C:\urban development kaggle 3\dataset"
    OUTPUT_CSV_FILE = r"C:\urban development kaggle 3\all_satellite_data.csv"
    
    # ✅ Initialize converter
    converter = SingleCSVConverter(skip_zero_values=True, skip_negative_values=True)
    
    # ✅ OPTION 1: Process all 60 folders into ONE CSV (Recommended for smaller datasets)
    print("\n" + "="*80)
    print("OPTION 1: Load all data into memory and save (Faster, needs more RAM)")
    print("="*80 + "\n")
    # Uncomment below to run:
    # converter.process_all_to_single_csv(BASE_DATASET_PATH, OUTPUT_CSV_FILE)
    
    # ✅ OPTION 2: Process with batch saving (Better for large datasets with limited RAM)
    print("\n" + "="*80)
    print("OPTION 2: Batch saving (Memory efficient)")
    print("="*80 + "\n")
    # Uncomment below to run:
    converter.process_with_progress_save(BASE_DATASET_PATH, OUTPUT_CSV_FILE, batch_size=500000)
    
    # Print summary
    converter.print_summary()
