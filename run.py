import os
import os.path as osp
from glob import glob

from misc import constants


def check_recording(recording, output_root):
    cur_cnt = 0
    cur_results = glob(output_root + '/' + recording + '/results/*/000.pkl')
    cur_results = [result.split('/')[-2] for result in cur_results]
    for frame_id in constants.prox_frames_test[recording]:
        if frame_id in cur_results:
            cur_cnt += 1
    is_finished = (cur_cnt == len(constants.prox_frames_test[recording]))
    return is_finished


def run(fit_cfg, use_quan=False, viz=False, debug=False):
    if 'ours' in fit_cfg:
        cfg_path = 'cfg_files/OURS.yaml'
        udf_ckpt_path = constants.udf_ckpt_path
    elif 'smplifyd' in fit_cfg:
        cfg_path = 'cfg_files/SMPLifyD.yaml'
    elif 'proxd' in fit_cfg:
        cfg_path = 'cfg_files/PROXD.yaml'

    if use_quan:
        recordings = constants.proxq_recordings
        recording_root = constants.proxq_root
        output_root = osp.join(constants.proxq_fit_root, fit_cfg)
        step = str(1)
    else:
        recordings = constants.prox_recordings_test
        recording_root = constants.prox_root
        output_root = osp.join(constants.prox_fit_root, fit_cfg)
        step = str(constants.step)

    for recording in sorted(recordings):
        recording_dir = osp.join(recording_root, 'recordings', recording)
        if not use_quan and check_recording(recording, output_root):
            print(recording, 'already finished.')
            continue 
        cmd = 'python prox/main.py'
        cmd = cmd + ' --config ' + cfg_path
        cmd = cmd + ' --recording_dir ' + recording_dir
        cmd = cmd + ' --output_folder ' + output_root
        cmd = cmd + ' --visualize=' + ('True' if viz else 'False')
        cmd = cmd + ' --step=' + step
        if 'ours' in fit_cfg:
            cmd = cmd + ' --udf_ckpt_path ' + udf_ckpt_path
        os.system(cmd)
        if viz or debug:
            break


if __name__ == '__main__':
    # run('smplifyd')
    # run('proxd')
    # run('ours')
    
    # run('smplifyd', viz=True)
    # run('proxd', viz=True)
    # run('ours', viz=True)     
    
    # run('smplifyd', debug=True)
    # run('proxd', debug=True)
    # run('ours', debug=True)
    
    pass