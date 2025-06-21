from ezc3d import c3d
import numpy as np

# Some common utility methods

def get_marker_names(c3d_file_path):
    return c3d(c3d_file_path)['parameters']['POINT']['LABELS']['value'][0:39]


def get_keypoints(c3d_file_path, keypoint_idx):
    # get relevant keypoints
    c = c3d(c3d_file_path)
    point_data = c['data']['points']
    point_data_relevant = np.squeeze(point_data[:3, keypoint_idx, :])
    frames = np.swapaxes(point_data_relevant, 0, 1)

    # extract x, y, and z coordinates
    x = frames[:, 0]
    y = frames[:, 1]
    z = frames[:, 2]

    return x, y, z


def group_intervals(data):
    if len(data) == 0: return []
    intervals = []
    start = data[0]
    end = data[0]
    for num in data[1:]:
        if num == end + 1:
            end = num
        else:
            intervals.append((start, end))
            start = num
            end = num
    intervals.append((start, end))
    return intervals