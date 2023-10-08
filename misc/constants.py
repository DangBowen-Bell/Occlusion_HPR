import os
import os.path as osp
import json


#! ######################################################################
#! paths

#* prox
proxq_root = '/data/PROX/quantitative'
prox_root = '/data/PROX/qualitative'
prox_kpt_root = '/data/PROX/keypoints'
proxd_fit_root = '/data/PROX/proxd_fittings'
posa_root = '/data/POSA'

#* smplx
smpl_root = '/data/SMPL-X/models'
vposer_path = '/data/SMPL-X/vposerDecoderWeights.npz'
part_segm_path = '/data/SMPL-X/other/smplx_parts_segm.pkl'

#* volume matching
vertice_pairs_path = '/data/SMPL-X/other/vertice_pairs.npy'

#* free zone
fznet_root = '/data/FZNet/'
udf_kpt_root = 'data/FZNet_ckpt/model.pt'

#* result
result_root = '/data/Result/'
prox_fit_root = result_root + 'fittings'
proxq_fit_root = prox_fit_root + '_proxq'
prox_img_root = result_root + 'images'
proxq_img_root = prox_img_root + '_proxq'
prox_mesh_root = result_root + 'mesh'
proxq_mesh_root = prox_mesh_root + '_proxq'

#* other
lemo_fit_root = '/data/LEMO/PROXD_temp'
proxe_root = '/data/PROX-E'


#! ######################################################################
#! recordings (prox)
proxq_recording_dir = osp.join(proxq_root, 'recordings')
proxq_recordings = sorted(os.listdir(proxq_recording_dir))

prox_recording_dir = osp.join(prox_root, 'recordings')
prox_recordings = sorted(os.listdir(prox_recording_dir))

proxd_recordings = sorted(os.listdir(proxd_fit_root))

lemo_recordings = sorted(os.listdir(lemo_fit_root))

prox_recordings_test = proxd_recordings


#! ######################################################################
#! frames (prox)

step = 30

prox_split_info_path = osp.join(fznet_root, 'prox_split_info.json')
if osp.exists(prox_split_info_path):
    with open(prox_split_info_path, 'r') as f:
        prox_split_info = json.load(f)
    prox_frames_test = {recording: [] for recording in prox_recordings_test}
    for info in prox_split_info['test']:
        recording, frame_id =info.split('-')
        prox_frames_test[recording].append(frame_id)
    
    total_frame = len(prox_split_info['test'])


#! ######################################################################
#! hsiviewer

prox_fittings = ['smplifyd', 'proxd', 'ours']
proxq_fittings = ['mosh', 'smplifyd', 'proxd', 'ours']


#! ######################################################################
#! other

H, W = 1080, 1920
fx, fy = 1060.53, 1060.38
cx, cy = 951.30, 536.77

volume_den_fit = 8
