# Copyright 2025 Dahlem Center for Machine Learning and Robotics, Freie Universität Berlin
# CC BY-NC-SA 4.0
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import argparse

def merge_geotiffs(input_tiff1, input_tiff2, output_tiff):
    with rasterio.open(input_tiff1) as src1, rasterio.open(input_tiff2) as src2:
        # Calculate the transform and dimensions for the merged output
        transform, width, height = calculate_default_transform(
            src1.crs, src2.crs, 
            src1.width, src1.height, 
            *src1.bounds
        )

        # Update metadata for the output
        meta = {
            'driver': 'GTiff',
            'count': 3,  # Three bands in the output
            'dtype': np.uint8,
            'width': width,
            'height': height,
            'crs': src1.crs,
            'transform': transform,
            'compress': 'ZSTD',
            'nodata': 0,
        }

        # Create an empty array for the merged bands
        merged_bands = np.zeros((3, height, width), dtype=np.float32)

        # Reproject and read the first GeoTIFF
        for i in range(3):
            reproject(
                src1.read(i + 1),
                merged_bands[i],
                src_transform=src1.transform,
                dst_transform=transform,
                dst_shape=(height, width),
                src_crs=src1.crs,
                dst_crs=src1.crs,
                resampling=Resampling.nearest
            )

        # Reproject and read the second GeoTIFF
        for i in range(3):
            band2 = np.zeros((height, width), dtype=np.float32)
            reproject(
                src2.read(i + 1),
                band2,
                src_transform=src2.transform,
                dst_transform=transform,
                dst_shape=(height, width),
                src_crs=src2.crs,
                dst_crs=src2.crs,
                resampling=Resampling.nearest
            )

            # Fill the merged bands: use band1 where it's not zero, otherwise use band2
            merged_bands[i] = np.where(merged_bands[i] == 0, band2, merged_bands[i])

        # Define the output GeoTIFF and write the merged bands
        with rasterio.open(output_tiff, 'w', **meta) as dst:
            for i in range(3):
                dst.write(merged_bands[i], i + 1)  # Write each band

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Merge two GeoTIFF files with three bands, filling zeros in the first with values from the second.')
    parser.add_argument('input_tiff1', type=str, help='Path to the first input GeoTIFF file.')
    parser.add_argument('input_tiff2', type=str, help='Path to the second input GeoTIFF file.')
    parser.add_argument('output_tiff', type=str, help='Path to the output merged GeoTIFF file.')

    args = parser.parse_args()

    # Call the merge function with the provided arguments
    merge_geotiffs(args.input_tiff1, args.input_tiff2, args.output_tiff)

if __name__ == '__main__':
    main()
