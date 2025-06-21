import sys
sys.path.append("..//implementation")

from tqdm import tqdm
from os.path import isfile
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import ezc3d

from helper.util import get_marker_names, get_keypoints
from helper.plot_markers import plot_2d

plt.rcParams.update({'font.size': 6})

# The c3d data we captured for S3 performing jumpingjacks was corrupted.
# Use the this script to remove the corrupt data (sets it to NaN, so it will be imputed in ./fix_c3d_folder.py).

AFFECTED_FILES = {
    'preprocessing\c3d\S3_jumpingjacks_lighting.c3d': [
        (49, 66), (166, 173), (274, 281), (392, 400), (496, 503)
    ],
    # 'F:/MPC/S3/c3ds/s3_jumpingjacks_normal.c3d': [], # missing way to much markers
    'preprocessing\c3d\S3_jumpingjacks_object.c3d': [
        (26, 33), (120, 142), (245, 253), (353, 360), (363, 372), (454, 490), (569, 576)
    ],
    'preprocessing\c3d\S3_jumpingjacks_person.c3d': [
        (22, 40), (133, 150), (245, 253), (352, 360), (363, 372), (407, 413)
    ],
}

def main():
    for f, corrupted_start_end in tqdm(AFFECTED_FILES.items()):
        print(f)
        assert isfile(f)
        plot_raw_c3d(f)
        c3d = ezc3d.c3d(f)
        point_data_old = c3d['data']['points']
        point_data = np.empty_like(point_data_old)

        index_lists = [list(range(start, end + 1)) for start, end in corrupted_start_end]
        corrupt_indices = list(itertools.chain.from_iterable(index_lists))

        # Plot (indices that will be deleted are shown in yellow)
        _, axis = plt.subplots(8, 5)
        for i in range(39):
            marker_names = get_marker_names(f)
            ax = axis[math.floor(i/5), i % 5]
            x, y, z = get_keypoints(f, i)
            title = f'{marker_names[i]} ({i})'
            plot_2d(ax, title, x, y, z, [], corrupt_indices)
        plt.legend(loc='lower right')
        plt.show(block=True)

        # Remove corrupt indices for all keypoints
        for i in range(39):
            x, y, z = get_keypoints(f, i)
            x, y, z = remove_corrupt_data(x, y, z, corrupt_indices) 
            point_data[0, i, :] = x
            point_data[1, i, :] = y
            point_data[2, i, :] = z

        del c3d['data']['points']
        c3d['data']['points'] = point_data
        c3d.write(f) # overwrite file

# Plot the raw data (before removing corrupted indices)
def plot_raw_c3d(f):
    # Load the marker names and C3D data
    marker_names = get_marker_names(f)
    c3d = ezc3d.c3d(f)

    # Create subplots for 39 markers
    _, axis = plt.subplots(8, 5, figsize=(15, 10))  # Adjust figure size
    for i in range(39):
        # Get marker coordinates (x, y, z) for each marker
        x, y, z = get_keypoints(f, i)
        title = f'{marker_names[i]} ({i})'

        # Plot the raw data (no corrupted indices highlighted yet)
        ax = axis[math.floor(i / 5), i % 5]
        plot_2d(ax, title, x, y, z, [], [])  # Pass empty corrupt indices
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.show(block=True)

def remove_corrupt_data(x, y, z, corrupt_indices):
    x[corrupt_indices] = np.NaN
    y[corrupt_indices] = np.NaN
    z[corrupt_indices] = np.NaN
    return x, y, z


def find_corrupt_indices_zscore(x, y, z, combined=False):
    # Here we experimented with automatic detection of corrupt data
    # As it only affected 3 files, we decided to manualy look for the corrupt indices instead (using plot.py)
    combined_zscore_theshold = 5
    single_zscore_threshold = 2.5

    # search for corrupt indices
    zscore_x = stats.zscore(x, axis=None)
    zscore_y = stats.zscore(y, axis=None)
    zscore_z = stats.zscore(z, axis=None)
    if combined:
        zscore = abs(zscore_x) + abs(zscore_y) + abs(zscore_z)
        corrupt_indices = np.where(zscore > combined_zscore_theshold)[0]
    else:
        zscore = np.max(np.transpose([abs(zscore_x), abs(zscore_y), abs(zscore_z)]), axis=1)
        corrupt_indices = np.where(zscore > single_zscore_threshold)[0]
    missing_indices = np.argwhere(np.isnan(x)).flatten()
    return list(set(corrupt_indices) | set(missing_indices)) # return corrupt and missing indices


if __name__ == '__main__':
    main()