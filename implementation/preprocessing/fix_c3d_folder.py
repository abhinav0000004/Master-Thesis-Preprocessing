import sys
sys.path.append("..//implementation")

from pathlib import Path
from os import listdir
from os.path import isfile, join
import ezc3d
import numpy as np
from experiments.plot import get_marker_names, get_keypoints, plot_multi, plot_single
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import argparse
from helper.interpolate import interpolate_missing

# Main function to process all .c3d files in a folder
def main(c3ds_dir, out_dir, do_plot=False):
    for f in tqdm(listdir(c3ds_dir)):
        fullpath = Path(join(c3ds_dir, f))
        assert isfile(fullpath)
        fix_file(fullpath, out_dir, do_plot)

# Function to process a single .c3d file
def fix_file(c3d_file_path: Path, out_dir: Path, do_plot=False):
    outpath = Path(f'{out_dir}/{c3d_file_path.stem}{c3d_file_path.suffix}')

    marker_names = get_marker_names(str(c3d_file_path))
    c3d = ezc3d.c3d(str(c3d_file_path))
    point_data_old = c3d['data']['points']
    point_data = np.empty_like(point_data_old)
    marker_count = len(c3d['parameters']['POINT']['LABELS']['value'])

    for i, marker_name in enumerate(marker_names):
        x, y, z = get_keypoints(str(c3d_file_path), i)

        smooth_fact = 3
        x = gaussian_filter1d(x, smooth_fact)
        y = gaussian_filter1d(y, smooth_fact)
        z = gaussian_filter1d(z, smooth_fact)

        if marker_name in ['RASI', 'LASI']:
            x, y, z, missing_indices = interpolate_missing(x, y, z, 'linear')
        else:
            x, y, z, missing_indices = interpolate_missing(x, y, z, 'gpr', 1, 5)

        point_data[0, i, :] = x
        point_data[1, i, :] = y
        point_data[2, i, :] = z

    if do_plot:
        plot_multi(str(c3d_file_path), marker_names, 'none')
    del c3d['data']['points']
    c3d['data']['points'] = point_data
    del c3d['data']['meta_points']['residuals']
    del c3d['data']['meta_points']['camera_masks']
    c3d['parameters']['POINT']['LABELS']['value'] = c3d['parameters']['POINT']['LABELS']['value'][:marker_count]

    # Write the data
    c3d.write(str(outpath))
    if do_plot:
        plot_multi(str(outpath), marker_names)

# Argument parser configuration
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix C3D folder script")
    parser.add_argument('c3ds_dir', type=str, help="Path to the directory containing .c3d files")
    parser.add_argument('out_dir', type=str, help="Path to the output directory")
    #parser.add_argument('--do_plot', action='store_true', help="Enable plotting (default: OFF)")
    
    args = parser.parse_args()

    # Call the main function with parsed arguments
    #main(args.c3ds_dir, args.out_dir, False)
    main(args.c3ds_dir, args.out_dir, args.do_plot)
# if __name__ == "__main__":
#     # Hardcoded values for input directory and output directory
#     c3ds_dir = "preprocessing\c3d"  # Replace with the actual path to your C3D files
#     out_dir = "preprocessing\processedc3d"  # Replace with the actual output directory path
#     do_plot = True  # Set to True if you want to enable plotting

#     # Call the main function directly with these values
#     main(c3ds_dir, out_dir, do_plot)