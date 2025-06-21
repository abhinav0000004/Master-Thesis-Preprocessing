import opensim as osim
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from itertools import product
from pathlib import Path

SUBJECT_ID = 3
ADDB_DIR_PATH = f'F:/MPC/S{SUBJECT_ID}/addb_results'
JSON_DIR_PATH = f'F:/MPC/S{SUBJECT_ID}/joints_3d'


# Convert captured .osim and (mutliple) .mot files to one .json file 

def main():
    addb_to_json(ADDB_DIR_PATH, JSON_DIR_PATH, SUBJECT_ID)


def addb_to_json(addb_dir, json_dir, subject_id):
    actions = ['conversation', 'drinking', 'freestyle', 'jumpingjacks', 'shoelaces', 'walking']
    variations = ['normal', 'object', 'person', 'lighting']
    trials = [(a, v) for a, v in product(actions, variations)]

    osim_file = Path(addb_dir, 'Models', 'match_markers_but_ignore_physics.osim')
    for action, variation in tqdm(trials):
        if SUBJECT_ID == 3 and action == 'jumpingjacks' and variation == 'normal': continue # skip poor quality capture
        file_basename = f's{subject_id}_{action}_{variation}'
        mot_file1 = Path(addb_dir, 'IK', file_basename + '_segment_0_ik.mot')
        mot_file2 = Path(addb_dir, 'IK', file_basename + '_segment_1_ik.mot') # potentially does not exist
        json_file = Path(json_dir, file_basename + '.json')

        parse_mot_osim(mot_file1, mot_file2, osim_file, json_file)


def parse_mot_osim(mot_file1: Path, mot_file2: Path, osim_file: Path, json_out_file: Path):
    # Create dataframe with marker positions
    model = osim.Model(str(osim_file))
    
    in_degrees = check_in_degrees(str(mot_file1))
    
    motion_data = osim.TimeSeriesTable(str(mot_file1))
    pos_df, marker_set_names = get_marker_positions(motion_data, model, in_degrees=in_degrees) #, marker_list=marker_list)

    if mot_file2.is_file(): # handle second mot file, when motion data was segmented
        motion_data2 = osim.TimeSeriesTable(str(mot_file2))
        pos_df2, _ = get_marker_positions(motion_data2, model, in_degrees=in_degrees)
        pos_df2['frame'] += len(pos_df) # start frame after last frame of first segment (usually frame 2000)
        pos_df = pd.concat([pos_df, pos_df2])

    # delete every second frame, as marker data was captured with 100FPS, and videos at 50FPS
    pos_df = pos_df.iloc[::2].reset_index(drop=True)
    pos_df['frame'] = (pos_df['frame'] / 2).astype(int) # adjust the frames accordingly

    pos_df.set_index('frame', inplace=True)
    pos_df.drop('time', axis=1, inplace=True)

    for marker_name in marker_set_names:
        pos_df[marker_name] = pos_df[[f'{marker_name}_x', f'{marker_name}_y', f'{marker_name}_z']].apply(lambda row: list(row), axis=1)
        pos_df.drop([f'{marker_name}_x', f'{marker_name}_y', f'{marker_name}_z'], axis=1, inplace=True)

    df_melted = pos_df.reset_index().melt(id_vars="frame", var_name="keypoint", value_name="position")

    # Convert MultiIndex DataFrame to a nested dictionary
    nested_dict = (
        df_melted
        .groupby('frame')
        .apply(lambda group: group.set_index('keypoint').to_dict(orient='index'), include_groups=False)
        .to_dict()
    )

    # Remove superfluous attribute
    transformed_dict = {}
    for frame, markers in nested_dict.items():
        transformed_dict[frame] = {
            marker: details["position"] for marker, details in markers.items()
        }

    # Write Json to file
    with open(json_out_file, "w") as file:
        json.dump(transformed_dict, file, indent=4)


# adjusted from: https://github.com/perfanalytics/pose2sim/blob/main/Pose2Sim/Utilities/trc_from_mot_osim.py
def get_marker_positions(motion_data, model, in_degrees=True, marker_list=[]):
    '''
    Get dataframe of marker positions
    
    INPUTS: 
    - motion_data: .mot file opened with osim.TimeSeriesTable
    - model: .osim file opened with osim.Model 
    - in_degrees: True if the motion data is in degrees, False if in radians
    - marker_list: list of marker names to include in the trc file. All if not specified
    
    OUTPUT:
    - marker_positions_pd: DataFrame of marker positions 
    '''
    
    # Markerset
    marker_set = model.getMarkerSet()
    marker_set_names = [mk.getName() for mk in list(marker_set)]
    if len(marker_list)>0:
        marker_set_names = [marker for marker in marker_list if marker in marker_set_names]
        absent_markers = [marker for marker in marker_list if marker not in marker_set_names]
        if len(absent_markers)>0:
            print(f'The following markers were not found in the model: {absent_markers}')
    marker_set_names_xyz = np.array([[m+'_x', m+'_y', m+'_z'] for m in marker_set_names]).flatten()

    # Data
    times = motion_data.getIndependentColumn()
    joint_angle_set_names = motion_data.getColumnLabels() # or [c.getName() for c in model.getCoordinateSet()]
    joint_angle_set_names = [j for j in joint_angle_set_names if not j.endswith('activation')]
    motion_data_pd = pd.DataFrame(motion_data.getMatrix().to_numpy()[:,:len(joint_angle_set_names)], columns=joint_angle_set_names)

    # Get marker positions at each state
    state = model.initSystem()
    marker_positions = []
    for n,t in enumerate(times):
        # put the model in the right position
        for coord in joint_angle_set_names:
            if in_degrees and not coord.endswith('_tx') and not coord.endswith('_ty') and not coord.endswith('_tz'):
                value = motion_data_pd.loc[n,coord]*np.pi/180
            else:
                value = motion_data_pd.loc[n,coord]
            model.getCoordinateSet().get(coord).setValue(state,value, enforceContraints=False)
        # model.assemble(state)
        model.realizePosition(state) # much faster (IK already done, no need to compute it again)
        # get marker positions
        marker_positions += [np.array([marker_set.get(mk_name).findLocationInFrame(state, model.getGround()).to_numpy() for mk_name in marker_set_names]).flatten()]
    marker_positions_pd = pd.DataFrame(marker_positions, columns=marker_set_names_xyz)
    marker_positions_pd.insert(0, 'time', times)
    marker_positions_pd.insert(0, 'frame', np.arange(len(times)))
    
    return marker_positions_pd, marker_set_names


def check_in_degrees(mot_path) -> bool:
    # In degrees or radians
    with open(mot_path) as m_p:
        while True:
            line =  m_p.readline()
            if 'inDegrees' in line:
                break
    if 'yes' in line:
        return True
    else:
        return False


if __name__ == '__main__':
    main()