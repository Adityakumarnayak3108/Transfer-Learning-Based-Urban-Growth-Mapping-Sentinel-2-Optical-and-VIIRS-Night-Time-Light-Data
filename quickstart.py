"""
QUICKSTART - Convert ALL 60 folders into ONE SINGLE CSV file
Just modify the paths below and run!
"""

import os
import sys

# ============================================================================
# ⭐ CONFIGURATION - MODIFY THESE 2 PATHS
# ============================================================================

# Path to your dataset root folder
BASE_DATASET_PATH = r"C:\urban development kaggle 3\dataset"

# Where to save the SINGLE CSV file (all 60 folders combined)
OUTPUT_CSV_FILE = r"C:\urban development kaggle 3\all_satellite_data.csv"

# ============================================================================
# DON'T MODIFY BELOW THIS LINE
# ============================================================================

def check_dependencies():
    """Check if required libraries are installed."""
    required = ['rasterio', 'pandas', 'numpy']
    missing = []
    
    for lib in required:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    
    if missing:
        print(f"❌ Missing required libraries: {', '.join(missing)}")
        print(f"\nInstall them using:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)
    
    print("✓ All dependencies installed")


def main():
    """Run the converter."""
    
    print("\n" + "="*80)
    print("🚀 TIF to SINGLE CSV Converter - Quick Start")
    print("="*80 + "\n")
    
    # Check dependencies
    print("Checking dependencies...")
    check_dependencies()
    
    # Import after dependency check
    from tif_to_single_csv import SingleCSVConverter
    
    # Validate paths
    print("\nValidating paths...")
    
    if not os.path.exists(BASE_DATASET_PATH):
        print(f"❌ Dataset path not found: {BASE_DATASET_PATH}")
        sys.exit(1)
    print(f"✓ Dataset path found: {BASE_DATASET_PATH}")
    
    output_dir = os.path.dirname(OUTPUT_CSV_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}")
    else:
        print(f"✓ Output directory: {output_dir if output_dir else 'Current directory'}")
    
    # Initialize converter
    print("\nInitializing converter...")
    converter = SingleCSVConverter(skip_zero_values=True, skip_negative_values=True)
    
    # Process
    print("\n⏳ Starting conversion...\n")
    
    try:
        # Use batch saving for memory efficiency
        converter.process_with_progress_save(
            BASE_DATASET_PATH, 
            OUTPUT_CSV_FILE, 
            batch_size=500000
        )
    
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✅ CONVERSION COMPLETE!")
    print("="*80)
    print(f"\n📁 Output file created: {OUTPUT_CSV_FILE}")
    print(f"📊 All data from 60 folders combined into ONE CSV file")
    
    # Try to show file size
    try:
        file_size_mb = os.path.getsize(OUTPUT_CSV_FILE) / (1024 * 1024)
        print(f"💾 File size: {file_size_mb:.2f} MB")
    except:
        pass
    
    print()


if __name__ == "__main__":
    main()