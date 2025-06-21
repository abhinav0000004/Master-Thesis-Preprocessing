import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from pathlib import Path
import os
from ..helper.interpolate import interpolate_missing
from ..helper.util import get_keypoints, get_marker_names, group_intervals

# Check whether linear interpolation or GPR performs better on complete c3d data (we skip any markers with gaps)
# We delete 'num_tests_interval' of length 'test_len_interval' from a complete marker, by setting the x, y, and z value at corresponding frames to NaN
# We then calculate the average error from the ground truth for the linear and GPR interpolation

DATA_DIR = Path('F:', 'MPC') # Location of the MPC dataset on your machine
CSV_PATH = Path('..', 'output', 'lerp_vs_gpr.csv') # Path to csv file containing the errors 
DO_PLOT = False # If True: Show a plot of every marker interpolation, comparing ground truth to linear and GPR interpolation

num_tests_interval = [1, 8]
test_len_interval = [10, 100]
take = 1
use = 5

random.seed(555) # set random seed to ensure consistency across multiple executions

def main():
    # take 24 files (3 in each subject, 4 for each action, 6 for each variation) and compute avg_error_lin, avg_error_gpr for every one
    df = pd.DataFrame(columns=['avg_error_lin', 'avg_error_gpr'])
    df.index.name = 'c3d_path'
    for i in range(8):
        subj = f'S{i+1}'
        c3d_folder = os.path.join(DATA_DIR, subj, 'c3ds')
        files = os.listdir(c3d_folder)
        files.sort()
        for j in range(3):
            file_idx = (8*j) + i
            c3d_path = os.path.join(c3d_folder, files[file_idx])
            avg_error_lin, avg_error_gpr = test_file(c3d_path)
            df.loc[c3d_path] = [avg_error_lin, avg_error_gpr]
            df.to_csv(CSV_PATH) # save to .csv after every iteration


def test_file(c3d_file_path):
    absolute_error_lin = 0
    absolute_error_gpr = 0

    marker_names = get_marker_names(c3d_file_path)

    num_complete_keypoints = 0
    for kp_idx, marker_name in enumerate(tqdm(marker_names)):
        if DO_PLOT: print(marker_name)
        x, y, z = get_keypoints(c3d_file_path, kp_idx)
        point_data_3d = np.array([x, y, z])
        if np.isnan(point_data_3d).any(): continue # skip incomplete keypoints

        error_lin, error_gpr = test_keypoint(point_data_3d)
        absolute_error_lin += error_lin
        absolute_error_gpr += error_gpr
        num_complete_keypoints += 1

    if num_complete_keypoints == 0:
        avg_error_lin = np.NaN
        avg_error_gpr = np.NaN
    else:
        avg_error_lin = absolute_error_lin / num_complete_keypoints
        avg_error_gpr = absolute_error_gpr / num_complete_keypoints

    print('==========')
    print(c3d_file_path)
    print('AVG_ERROR_LIN', avg_error_lin)
    print('AVG_ERROR_GPR', avg_error_gpr)
    print('==========')

    return avg_error_lin, avg_error_gpr


def test_keypoint(point_data_3d: np.array):
    assert not np.isnan(point_data_3d).any() # only work with completely captured keypoints    
    assert point_data_3d.shape[0] == 3 # sanity check
    assert point_data_3d.shape[1] > 500 # check that we have enough points
    
    smooth_fact = 3
    point_data_3d = np.apply_along_axis(lambda dim: gaussian_filter1d(dim, smooth_fact), axis=1, arr=point_data_3d)

    x_gt, y_gt, z_gt = point_data_3d.copy()
    x, y, z = point_data_3d

    num_tests = random.randint(*num_tests_interval)
    for _ in range(num_tests):
        # we deliberately do not check if intervals overlap
        length = random.randint(*test_len_interval)
        start = random.randint(0, point_data_3d.shape[1] - length)
        point_data_3d[:, start:start + length] = np.nan

    x, y, z, missing_indices = interpolate_missing(x, y, z, 'none')

    x_lin, y_lin, z_lin, _ = interpolate_missing(x, y, z, 'linear')
    x_gpr, y_gpr, z_gpr, _ = interpolate_missing(x, y, z, 'gpr', take, use)

    # gt, lin and gpr points at the deleted indices
    gt_points_miss = np.array([x_gt, y_gt, z_gt])[:, missing_indices]
    lin_points_miss = np.array([x_lin, y_lin, z_lin])[:, missing_indices]
    gpr_points_miss = np.array([x_gpr, y_gpr, z_gpr])[:, missing_indices]

    absolute_error_lin = np.absolute(gt_points_miss - lin_points_miss).sum()
    absolute_error_gpr = np.absolute(gt_points_miss - gpr_points_miss).sum()

    avg_error_lin = absolute_error_lin / len(missing_indices)
    avg_error_gpr = absolute_error_gpr / len(missing_indices)

    if DO_PLOT:
        print('avg_error_lin', avg_error_lin)
        print('avg_error_gpr', avg_error_gpr)

        ax = plt.axes()

        for p in group_intervals(missing_indices):
            ax.axvspan(p[0], p[1], color='#ff8080', alpha=0.2)
        framecount = len(x)
        ax.plot(range(framecount), x_gt, linewidth=2.0, label='x_gt', color='tab:blue', alpha=0.5)
        ax.plot(range(framecount), y_gt, linewidth=2.0, label='y_gt', color='tab:blue', alpha=0.5)
        ax.plot(range(framecount), z_gt, linewidth=2.0, label='z_gt', color='tab:blue', alpha=0.5)

        ax.plot(range(framecount), x_lin, linewidth=2.0, label='x_lin', color='tab:orange', alpha=0.5)
        ax.plot(range(framecount), y_lin, linewidth=2.0, label='y_lin', color='tab:orange', alpha=0.5)
        ax.plot(range(framecount), z_lin, linewidth=2.0, label='z_lin', color='tab:orange', alpha=0.5)

        ax.plot(range(framecount), x_gpr, linewidth=2.0, label='x_gpr', color='tab:green', alpha=0.5)
        ax.plot(range(framecount), y_gpr, linewidth=2.0, label='y_gpr', color='tab:green', alpha=0.5)
        ax.plot(range(framecount), z_gpr, linewidth=2.0, label='z_gpr', color='tab:green', alpha=0.5)

        # Label the axes
        ax.set_xlabel('Frames')
        ax.set_ylabel('Deflection in mm')

        ax.set_title('Interpolation Comparison')
        plt.legend(loc='upper right')
        plt.show()

    return avg_error_lin, avg_error_gpr


if __name__ == "__main__":
    main()