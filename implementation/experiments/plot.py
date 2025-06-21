import sys
sys.path.append("..//implementation")

import math
import numpy as np
import matplotlib.pyplot as plt
from helper.plot_markers import plot_2d
from helper.util import group_intervals, get_keypoints, get_marker_names
from helper.interpolate import interpolate_missing, interpolate_nan_gpr_uncertainty
from scipy.ndimage import gaussian_filter1d

# Contains methods for plotting c3d files in different ways

def main():
    c3d_file_path = 'preprocessing\c3d\S3_drinking_normal.c3d'
    #c3d_file_path = 'preprocessing\c3d\with\S3_jumpingjacks_lighting.c3d'
    keypoint_index = 1 # not relevant if diagram = 'multi'
    method = 'none'
    diagram = 'multi'
    take = 1 # every nth index that will be used
    use = 1 # every nth index that will be the actual value, rest will be NaN

    marker_names = get_marker_names(c3d_file_path)

    match diagram:
        case 'single':
            plot_single(c3d_file_path, marker_names, keypoint_index, method, take, use)
        case 'multi':
            plot_multi(c3d_file_path, marker_names, method)
        case 'compare':
            plot_compare(c3d_file_path, marker_names, keypoint_index, take, use)
        case 'uncertainty':
            plot_uncertainty(c3d_file_path, marker_names, keypoint_index, take, use)
        case 'smoothing':
            plot_smoothing(c3d_file_path, marker_names, keypoint_index)


def plot_single(c3d_file_path, marker_names, keypoint_idx, method='linear', take=1, use=1):
    title = f'{marker_names[keypoint_idx]} ({keypoint_idx})'
    x, y, z = get_keypoints(c3d_file_path, keypoint_idx)
    ax2 = plt.axes()
    x, y, z, missing_indices = interpolate_missing(x, y, z, method, take, use)
    plot_2d(ax2, title, x, y, z, missing_indices, [])

    plt.legend(loc='upper right', fontsize=12)
    plt.show(block=True)


def plot_multi(c3d_file_path, marker_names, method='linear'):
    _, axis = plt.subplots(8, 5)
    for i in range(39):
        ax = axis[math.floor(i/5), i % 5]
        x, y, z = get_keypoints(c3d_file_path, i)
        x, y, z, missing_indices = interpolate_missing(x, y, z, method)
        title = f'{marker_names[i]} ({i})'
        plot_2d(ax, title, x, y, z, missing_indices, [])
    plt.legend(loc='lower right')
    plt.show(block=True)


def plot_compare(c3d_file_path, marker_names, keypoint_idx, take=1, use=1):
    ax = plt.axes()
    x, y, z = get_keypoints(c3d_file_path, keypoint_idx)
    length = len(x)
    missing_indices = np.where(np.isnan(x))[0]

    for p in group_intervals(missing_indices):
        ax.axvspan(p[0], p[1], color='#ff8080', alpha=0.2)
    ax.plot(range(length), y, linewidth=2, label='y captured', alpha=1, color='tab:orange', zorder=10)

    method = 'linear'
    x, y, z = get_keypoints(c3d_file_path, keypoint_idx)
    x, y, z, _ = interpolate_missing(x, y, z, method, take, use)
    # ax.plot(range(length), x, linewidth=2.0, label='x linear interp.')
    ax.plot(range(length), y, linewidth=2.0, label='y linear interp.', color='tab:brown', alpha=1)
    # ax.plot(range(length), z, linewidth=2.0, label='z linear interp.')

    method = 'gpr'
    x, y, z = get_keypoints(c3d_file_path, keypoint_idx)
    x, y, z, _ = interpolate_missing(x, y, z, method, take, use)
    # ax.plot(range(length), x, linewidth=2, label='x gpr interp.')
    ax.plot(range(length), y, linewidth=2, label='y gpr interp.', color='tab:pink', alpha=1)
    # ax.plot(range(length), z, linewidth=2, label='z gpr interp.')

    # Label the axes
    ax.set_xlabel('Frames')
    ax.set_ylabel('Y [mm]')

    plt.legend(loc='upper right')
    plt.show(block=True)


def plot_uncertainty(c3d_file_path, marker_names, keypoint_idx, take, use):
    ax = plt.axes()

    x, y, z = get_keypoints(c3d_file_path, keypoint_idx)
    length = len(x)
    missing_indices = np.where(np.isnan(x))[0]
    # for p in range(0, length, use):
    #     ax.axvspan(p, p+1, color='#a0a0a0', alpha=0.1)
    for p in group_intervals(missing_indices):
        ax.axvspan(p[0], p[1], color='#ff8080', alpha=0.2)

    ax.plot(range(length), y, linewidth=2, label='y captured', alpha=1, color='tab:orange', zorder=10)

    x, y, z = get_keypoints(c3d_file_path, keypoint_idx)

    missing_indices = np.where(np.isnan(x))[0]
    x = np.array([elem if i % use == 0 else np.nan for (i, elem) in enumerate(x)])[::take]
    y = np.array([elem if i % use == 0 else np.nan for (i, elem) in enumerate(y)])[::take]
    z = np.array([elem if i % use == 0 else np.nan for (i, elem) in enumerate(z)])[::take]

    x, y, z, x_std, y_std, z_std = interpolate_nan_gpr_uncertainty(x, y, z)

    framecount = len(x)
    # ax.plot(range(framecount), x, linewidth=2.0, label='x', color='tab:blue')
    ax.plot(range(framecount), y, linewidth=2.0, label='y gpr interp.', color='tab:pink')
    # ax.plot(range(framecount), z, linewidth=2.0, label='z', color='tab:green')

    # plt.fill_between(
    #     np.array(range(framecount)).ravel(),
    #     x - x_std,
    #     x + x_std,
    #     alpha=0.5,
    #     label=r"x $\pm$ 1 std. dev.",
    #     color='tab:blue',
    # )

    plt.fill_between(
        np.array(range(framecount)).ravel(),
        y - y_std,
        y + y_std,
        alpha=0.5,
        label=r"y gpr $\pm$ 1 std. dev.",
        color='tab:pink',
    )

    # plt.fill_between(
    #     np.array(range(framecount)).ravel(),
    #     z - z_std,
    #     z + z_std,
    #     alpha=0.5,
    #     label=r"z $\pm$ 1 std. dev.",
    #     color='tab:green',
    # )

    # Label the axes
    ax.set_xlabel('Frames')
    ax.set_ylabel('X, Y, Z in mm')

    plt.legend(loc='upper right')
    plt.show(block=True)


def plot_smoothing(c3d_file_path, marker_names, keypoint_idx):
    ax = plt.axes()
    title = f'{marker_names[keypoint_idx]} ({keypoint_idx})'
    x, y, z = get_keypoints(c3d_file_path, keypoint_idx)

    length = len(x)
    ax.plot(range(length), y, linewidth=2, label='y captured', alpha=0.5, color='tab:orange')

    smooth_fact = 3
    x = gaussian_filter1d(x, smooth_fact)
    y = gaussian_filter1d(y, smooth_fact)
    z = gaussian_filter1d(z, smooth_fact)

    ax.plot(range(length), y, linewidth=2, label='y smoothed', alpha=1, color='tab:orange', linestyle='dashed')

    plt.xlim((115, 145))
    plt.ylim((240, 260))

    title = f'{marker_names[keypoint_idx]} ({keypoint_idx})'
    ax.set_title(title)
    plt.legend(loc='upper right')
    plt.show(block=True)


if __name__ == '__main__':
    main()
