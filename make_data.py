import os.path as osp
from tqdm import tqdm
import os
import json
import numpy as np

from misc import constants, utils, data_io


def make_pc_prox_recording(save_dir, proxio, fit_type):
    """make pc data for one recording

    Args:
        save_dir (str): _description_
        proxio (DataIO): instantiated PROXIO object
        fit_type (str): _description_ 
    """
    
    for idx in tqdm(range(len(proxio))):
        frame_id = proxio.frame_list[idx]
        save_path = osp.join(save_dir, frame_id + '.npz')
        if osp.exists(save_path):
            continue

        depth_data = proxio.get_depth(frame_id)
        depth_points = depth_data['points']
        if depth_points is None:
            continue
        
        body_pc_data = proxio.get_body_pc(frame_id, fit_type)
        if body_pc_data is None:
            continue
        body_vertices = body_pc_data['vertices']
        body_points = body_pc_data['points']
        body_parts = body_pc_data['parts']
        body_center = body_pc_data['joints'][0]
        
        scene_points = proxio.get_scene(body_center)
        if scene_points is None:
            continue
        
        body_vertices = body_vertices - body_center
        body_points = body_points - body_center
        depth_points = depth_points - body_center
        scene_points = scene_points - body_center
        np.savez(save_path, body_v=body_vertices, body=body_points, body_p=body_parts, depth=depth_points, scene=scene_points, center=body_center)

def make_pc_prox(fit_type='proxd'):
    """make pc data for PROX dataset
    """
    
    prox_save_dir = osp.join(constants.fznet_root, 'prox')
    if fit_type == 'proxd':
        recordings = constants.proxd_recordings
    elif fit_type == 'lemo':
        recordings = constants.lemo_recordings
        
    proxio = data_io.PROXIO()
    for recording in tqdm(recordings):
        save_dir = osp.join(prox_save_dir, recording)
        os.makedirs(save_dir, exist_ok=True)
        proxio.instantiate(recording)
        make_pc_prox_recording(save_dir, proxio, fit_type)


def make_info_prox_recording(save_dir, proxio):
    """make information data (valid + non collision) for one recording

    Args:
        save_dir (str): _description_
        proxio (DataIO): instantiated PROXIO object
    """
    
    sdf_data = utils.load_sdf(proxio.sdf_path, proxio.sdf_info_path)
    vertice_pairs = utils.load_vertice_pairs()
    if not hasattr(proxio, 'info_dict'):
        proxio.info_dict = {}
    for frame in tqdm(sorted(os.listdir(save_dir))):
        save_path = osp.join(save_dir, frame)
        info = proxio.recording + '-' + frame[:-4]
        
        data = np.load(save_path)
        body_points = data['body'] + data['center']
        body_vertices = data['body_v'] + data['center']
        scene_points = data['scene'] + data['center']
        
        valid = utils.get_valid(body_points, scene_points)
        non_col = utils.get_non_col(body_points, sdf_data)
        vol_non_col = utils.get_vol_non_col(body_vertices, sdf_data, vertice_pairs)
        proxio.info_dict[info] = [valid, non_col, vol_non_col]

def make_info_prox():
    """make information data for PROX dataset
    """
    
    prox_save_dir = osp.join(constants.fznet_root, 'prox') 
    prox_info_path = prox_save_dir + '_info.json'
    prox_split_info_path = prox_save_dir + '_split_info.json'
    
    proxio = data_io.PROXIO()  
    for recording in tqdm(sorted(os.listdir(prox_save_dir))):
        recording_save_dir = osp.join(prox_save_dir, recording)
        proxio.instantiate(recording)
        make_info_prox_recording(recording_save_dir, proxio)
    info_json = json.dumps(proxio.info_dict)
    with open(prox_info_path, 'w') as f:
        f.write(info_json)
    
    split_info_dict = utils.get_prox_split_info(proxio.info_dict)
    split_info_json = json.dumps(split_info_dict)
    with open(prox_split_info_path, 'w') as f:
        f.write(split_info_json)


def make_vertice_pairs():
    utils.get_volume_points()


if __name__ == '__main__':
    # make_pc_prox()
    # make_info_prox()
    
    # make_vertice_pairs()
    
    pass