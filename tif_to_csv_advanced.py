import os
import rasterio
from rasterio.transform import Affine
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from datetime import datetime
import json

class TifToCsvConverter:
    """
    Advanced converter for GeoTIFF satellite imagery to CSV with geospatial data.
    Supports single and multi-band imagery with flexible output options.
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
    
    def extract_metadata(self, tif_file_path):
        """Extract metadata from GeoTIFF file."""
        try:
            with rasterio.open(tif_file_path) as src:
                return {
                    'crs': src.crs,
                    'transform': src.transform,
                    'bands': src.count,
                    'width': src.width,
                    'height': src.height,
                    'bounds': src.bounds,
                    'dtype': src.dtypes[0]
                }
        except Exception as e:
            print(f"Error reading metadata from {tif_file_path}: {str(e)}")
            return None
    
    def extract_geospatial_data_single_band(self, tif_file_path, band=1):
        """
        Extract data from single-band GeoTIFF.
        
        Args:
            tif_file_path: Path to the .tif file
            band: Band number to read (default: 1)
        
        Returns:
            DataFrame with columns: longitude, latitude, luminous_value
        """
        try:
            with rasterio.open(tif_file_path) as src:
                # Read the specified band
                data = src.read(band)
                transform = src.transform
                height, width = data.shape
                
                # Create lists to store data
                records = []
                
                # Iterate through all pixels
                for row in range(height):
                    for col in range(width):
                        pixel_value = float(data[row, col])
                        
                        # Apply filters
                        if self.skip_zero_values and pixel_value == 0:
                            continue
                        if self.skip_negative_values and pixel_value < 0:
                            continue
                        
                        # Calculate coordinates using affine transform
                        lon = transform.c + (col * transform.a)
                        lat = transform.f + (row * transform.e)
                        
                        records.append({
                            'longitude': lon,
                            'latitude': lat,
                            'luminous_value': pixel_value
                        })
                
                df = pd.DataFrame(records)
                return df if len(df) > 0 else None
        
        except Exception as e:
            print(f"Error processing {tif_file_path}: {str(e)}")
            return None
    
    def extract_geospatial_data_multi_band(self, tif_file_path):
        """
        Extract data from multi-band GeoTIFF.
        Creates separate values for each band.
        
        Args:
            tif_file_path: Path to the .tif file
        
        Returns:
            DataFrame with columns: longitude, latitude, band1, band2, ...
        """
        try:
            with rasterio.open(tif_file_path) as src:
                transform = src.transform
                num_bands = src.count
                height, width = src.read(1).shape
                
                records = []
                
                # Read all bands
                bands_data = [src.read(i+1) for i in range(num_bands)]
                
                # Iterate through all pixels
                for row in range(height):
                    for col in range(width):
                        # Get values from all bands
                        band_values = [float(bands_data[b][row, col]) for b in range(num_bands)]
                        
                        # Skip if all values are zero or negative
                        if all(v == 0 for v in band_values) if self.skip_zero_values else False:
                            continue
                        
                        # Calculate coordinates
                        lon = transform.c + (col * transform.a)
                        lat = transform.f + (row * transform.e)
                        
                        record = {
                            'longitude': lon,
                            'latitude': lat
                        }
                        
                        # Add band values
                        for b_idx, val in enumerate(band_values, 1):
                            record[f'band_{b_idx}'] = val
                        
                        records.append(record)
                
                df = pd.DataFrame(records)
                return df if len(df) > 0 else None
        
        except Exception as e:
            print(f"Error processing multi-band {tif_file_path}: {str(e)}")
            return None
    
    def process_tif_file(self, tif_file_path, output_csv_path, multi_band=False):
        """
        Process a single TIF file and save to CSV.
        
        Args:
            tif_file_path: Path to input .tif file
            output_csv_path: Path to output .csv file
            multi_band: If True, process all bands; if False, process first band only
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Get appropriate converter
        if multi_band:
            df = self.extract_geospatial_data_multi_band(tif_file_path)
        else:
            df = self.extract_geospatial_data_single_band(tif_file_path)
        
        if df is not None and len(df) > 0:
            df.to_csv(output_csv_path, index=False)
            self.stats['successful'] += 1
            self.stats['total_records'] += len(df)
            return True
        else:
            self.stats['failed'] += 1
            return False
    
    def process_all_folders(self, base_dataset_path, output_base_path, multi_band=False):
        """
        Process all 60 tile folders in the dataset.
        
        Args:
            base_dataset_path: Root path to the dataset
            output_base_path: Root path for output CSV files
            multi_band: If True, process all bands; if False, process first band only
        """
        print(f"{'='*70}")
        print(f"Starting batch processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        os.makedirs(output_base_path, exist_ok=True)
        
        # Find the train directory
        train_path = os.path.join(base_dataset_path, 'SN7_buildings_train', 'train')
        
        if not os.path.exists(train_path):
            print(f"❌ Train path not found: {train_path}")
            return
        
        # Get all tile folders
        tile_folders = sorted([f for f in os.listdir(train_path) 
                             if os.path.isdir(os.path.join(train_path, f))])
        
        print(f"📁 Found {len(tile_folders)} tile folders to process\n")
        
        # Process each tile folder
        for idx, tile_folder in enumerate(tile_folders, 1):
            tile_path = os.path.join(train_path, tile_folder)
            images_path = os.path.join(tile_path, 'images')
            
            if not os.path.exists(images_path):
                print(f"[{idx:2d}/{len(tile_folders)}] ⏭️  {tile_folder:<50} - No images folder")
                continue
            
            # Get all TIF files
            tif_files = sorted(glob.glob(os.path.join(images_path, '*.tif')))
            
            if not tif_files:
                print(f"[{idx:2d}/{len(tile_folders)}] ⏭️  {tile_folder:<50} - No TIF files")
                continue
            
            # Create output folder
            tile_output_path = os.path.join(output_base_path, tile_folder)
            os.makedirs(tile_output_path, exist_ok=True)
            
            print(f"[{idx:2d}/{len(tile_folders)}] 📍 {tile_folder}")
            print(f"               Found {len(tif_files)} TIF files")
            
            # Process each TIF file
            for tif_file in tif_files:
                tif_filename = os.path.basename(tif_file)
                csv_filename = tif_filename.replace('.tif', '.csv')
                csv_output_path = os.path.join(tile_output_path, csv_filename)
                
                success = self.process_tif_file(tif_file, csv_output_path, multi_band)
                
                status = "✓" if success else "✗"
                print(f"               {status} {csv_filename}")
            
            print()
        
        # Print summary statistics
        self._print_summary()
    
    def process_single_tile(self, tile_path, output_path=None, multi_band=False):
        """
        Process a single tile folder.
        
        Args:
            tile_path: Path to the tile folder
            output_path: Optional custom output path
            multi_band: If True, process all bands; if False, process first band only
        """
        if output_path is None:
            output_path = os.path.join(tile_path, 'csv_output')
        
        os.makedirs(output_path, exist_ok=True)
        
        images_path = os.path.join(tile_path, 'images')
        
        if not os.path.exists(images_path):
            print(f"❌ Images folder not found: {images_path}")
            return
        
        tif_files = sorted(glob.glob(os.path.join(images_path, '*.tif')))
        
        if not tif_files:
            print(f"❌ No TIF files found in {images_path}")
            return
        
        print(f"Processing {len(tif_files)} TIF files from {os.path.basename(tile_path)}\n")
        
        for tif_file in tif_files:
            tif_filename = os.path.basename(tif_file)
            csv_filename = tif_filename.replace('.tif', '.csv')
            csv_output_path = os.path.join(output_path, csv_filename)
            
            success = self.process_tif_file(tif_file, csv_output_path, multi_band)
            status = "✓" if success else "✗"
            print(f"{status} {tif_filename} -> {csv_filename}")
        
        print(f"\n✅ CSV files saved to: {output_path}")
    
    def _print_summary(self):
        """Print processing summary statistics."""
        print(f"{'='*70}")
        print(f"Processing Summary")
        print(f"{'='*70}")
        print(f"Total files processed: {self.stats['successful'] + self.stats['failed']}")
        print(f"✓ Successful: {self.stats['successful']}")
        print(f"✗ Failed: {self.stats['failed']}")
        print(f"📊 Total records extracted: {self.stats['total_records']:,}")
        print(f"{'='*70}\n")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Initialize converter
    converter = TifToCsvConverter(skip_zero_values=True, skip_negative_values=True)
    
    # OPTION 1: Process all 60 folders
    base_path = r"C:\urban development kaggle 3\dataset"  # Modify this path
    output_path = r"C:\urban development kaggle 3\csv_output"  # Modify this path
    
    # Uncomment to run:
    # converter.process_all_folders(base_path, output_path, multi_band=False)
    
    # OPTION 2: Process a single tile
    # single_tile = r"C:\urban development kaggle 3\dataset\SN7_buildings_train\train\L15-0331E-1257N_1327_3160_13"
    # converter.process_single_tile(single_tile, multi_band=False)
    
    print("Converter initialized. Use:")
    print("  converter.process_all_folders(base_path, output_path)")
    print("  converter.process_single_tile(tile_path)")
