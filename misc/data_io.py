import numpy as np
import cv2
import os.path as osp
import open3d as o3d
import smplx
import os
import json
import random
import sys

sys.path.append('..')
from misc import constants, utils


proxq_root = constants.proxq_root
prox_root = constants.prox_root
prox_kpt_root = constants.prox_kpt_root

proxd_fit_root = constants.proxd_fit_root
lemo_fit_root = constants.lemo_fit_root
ours_fit_root = constants.prox_fit_root
ours_fit_root_q = constants.proxq_fit_root

proxe_root = constants.proxe_root
posa_root = constants.posa_root


class DataIO:
    def __init__(self):
        """initialize some base class variables
        """
        
        #* frame list
        self.start = 0
        self.step = 30
        
        #* coordinate
        self.coord = 'world'        
        
        #* pc data
        self.body_num = 4096
        self.scene_num = self.body_num
        self.depth_num = 1024

        #* depth data
        self.depth_body = True
        self.use_color_mask = True
        self.sample_depth = 'random'
        
        #* scene data
        self.scene_type = 'posa'
        self.crop_type = 'ball'
        self.crop_size = 1.0
        self.sample_scene = 'random'
        
        #* smplx data
        self.model_type = 'smplx'

    def __len__(self):
        return len(self.frame_list)

    def get_scene(self, center):
        """crop scene points

        Args:
            center (np): [3], crop center

        Returns:
            scene_points: [n, 3] 
        """
        
        vertices_idxs = utils.get_points_idxs(points=self.scene_vertices,
                                              center=center, 
                                              crop_type=self.crop_type,
                                              crop_size=self.crop_size)
        scene_points = self.scene_vertices[vertices_idxs]
        if len(scene_points) > 0:
            if self.sample_scene == 'random':
                points_idxs = np.random.randint(len(scene_points), size=self.scene_num)
                scene_points = scene_points[points_idxs]
            else:
                scene_points = utils.fps(points=scene_points, num=self.scene_num)
        else:
            scene_points = None
        
        return scene_points


class PROXIO(DataIO):
    def instantiate(self, recording, load_model=True):
        """instantiate some class variables

        Args:
            recording (str): _description_
            load_model (bool): _description_
        """
        
        scene = recording.split('_')[0]
        #* qualitative or quantitative
        if scene == 'vicon':            
            recording_root = osp.join(proxq_root,'recordings', recording)
            self.color_kpt_dir = osp.join(proxq_root, 'keypoints_overlay', recording)
            self.calib_dir = osp.join(proxq_root, 'calibration')
            self.step = 1
            
            scene_path = osp.join(proxq_root, 'scenes', 'vicon.ply')
            c2w_trans_path = osp.join(proxq_root, 'cam2world', scene + '.json')
            m2w_trans_path = osp.join(proxq_root, 'vicon2scene.json')
            self.trans_m2w = utils.load_trans(m2w_trans_path)
            self.sdf_path = osp.join(proxq_root, 'sdf', scene + '_sdf.npy')
            self.sdf_info_path = osp.join(proxq_root, 'sdf', scene + '.json')
            
            scene_bb_min = [-3.48, -3.62, -0.28]
            scene_bb_max = [1.55, 1.69, 1.78]
            self.scene_bb_vs, self.scene_bb_fs = utils.get_scene_bb(scene_bb_min, scene_bb_max)
        else:            
            recording_root = osp.join(prox_root,'recordings', recording)
            self.color_kpt_dir = osp.join(prox_kpt_root, recording, 'keypoint_images')
            self.calib_dir = osp.join(prox_root, 'calibration')
            
            #* PROX & PROX-E have the same cooridinate & different tropology
            #* PROX & POSA have different cooridinate & tropology
            
            if self.scene_type == 'prox':
                scene_path = osp.join(prox_root, 'scenes', scene + '.ply')
                #* PROX-E scene
                # scene_path = osp.join(proxe_root, 'scenes_semantics', scene + '.ply')
                # scene_seg_path = osp.join(proxe_root, 'scenes_semantics', scene + '_withlabels.ply')
                c2w_trans_path = osp.join(prox_root, 'cam2world', scene + '.json')
                self.sdf_path = osp.join(prox_root, 'sdf', scene + '_sdf.npy')
                self.sdf_info_path = osp.join(prox_root, 'sdf', scene + '.json')
                scene_bb_path = osp.join(proxe_root, 'PROXE_box_verts.json')
                self.scene_bb_vs, self.scene_bb_fs = utils.load_scene_bb(scene_bb_path, scene)
            elif self.scene_type == 'posa':
                scene_path = osp.join(posa_root, 'scenes', scene + '.ply')
                c2w_trans_path = osp.join(posa_root, 'cam2world', scene + '.json')
                #* voxel semantics
                # scene_seg_path = osp.join(posa_root, 'sdf', scene + '_semantics.npy')
                self.sdf_path = osp.join(posa_root, 'sdf', scene + '_sdf.npy')
                self.sdf_info_path = osp.join(posa_root, 'sdf', scene + '.json')            
                scene_bb_min, scene_bb_max = utils.load_scene_bb_posa(self.sdf_info_path)
                self.scene_bb_vs, self.scene_bb_fs = utils.get_scene_bb(scene_bb_min, scene_bb_max)     
            
        self.color_dir = osp.join(recording_root, 'Color')
        self.depth_dir = osp.join(recording_root, 'Depth')
        self.mask_dir = osp.join(recording_root, 'BodyIndex')
        self.mask_color_dir = osp.join(recording_root, 'BodyIndexColor')
        self.frame_list = [frame[:-4] for frame in os.listdir(self.color_dir)]
        self.frame_list = sorted(self.frame_list)[self.start::self.step]

        #* load scene
        with open(c2w_trans_path, 'r') as f:
            self.trans_c2w = np.array(json.load(f))
            self.trans_w2c = np.linalg.inv(self.trans_c2w)
        
        scene_o3d = o3d.io.read_triangle_mesh(scene_path)
        scene_o3d.compute_vertex_normals()
        if self.coord == 'camera':
            scene_o3d.transform(self.trans_w2c)
        self.scene_o3d = scene_o3d
        self.scene_vertices = np.asarray(scene_o3d.vertices)
        
        #* load camera
        self.projection = utils.Projection('PROX', self.calib_dir, self.use_color_mask)
        self.camera_pos = np.zeros((1, 3))
        if self.coord == 'world':
            self.camera_pos = utils.transform_points(self.camera_pos, self.trans_c2w)
            
        self.camera_k = utils.get_camera_k()
        self.camera_o3d = o3d.geometry.LineSet().create_camera_visualization(
            view_width_px=constants.W, view_height_px=constants.H, 
            intrinsic=self.camera_k, extrinsic=self.trans_w2c)    
        
        #* load smplx model
        if load_model:
            self.model = utils.load_smplx_model(recording)
        
        #* save parameters
        self.recording = recording
        self.scene = scene
        self.scene_path = scene_path

    def get_depth(self, frame_id, do_sample=True):
        """get depth points & images

        Args:
            frame_id (str): _description_

        Returns:
            points: [n, 3], body depth points in world/camera space
            depth: [m, n], depth image
            color: [m, n, 3], color image
            mask: [m, n], body mask image
        """
        
        color_path = osp.join(self.color_dir, frame_id + '.jpg')
        color_img = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
        color_img = cv2.flip(color_img, 1)
        
        if 'vicon' in self.recording:
            color_kpt_path = osp.join(self.color_kpt_dir, frame_id + '.jpg')
        else:
            color_kpt_path = osp.join(self.color_kpt_dir, frame_id + '_rendered.png')
        color_kpt_img = color_img
        if osp.exists(color_kpt_path):            
            color_kpt_img = cv2.imread(color_kpt_path, cv2.IMREAD_UNCHANGED)
            
        depth_path = osp.join(self.depth_dir, frame_id + '.png')
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(float) / 8000.0
        depth_img = cv2.flip(depth_img, 1)

        if self.use_color_mask:
            mask_path = osp.join(self.mask_color_dir, frame_id + '.png')
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask_path = osp.join(self.mask_dir, frame_id + '.png')
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.threshold(mask_img, 254, 255, cv2.THRESH_BINARY)[1]
        mask_img = cv2.flip(mask_img, 1)

        #* use copy() in case it changed in the function
        if self.depth_body:
            depth_points = self.projection.get_depth_points(depth_img.copy(), color_img, mask_img)
        else:
            depth_points = self.projection.get_depth_points(depth_img.copy(), color_img)
                
        depth_points = utils.remove_noise(depth_points)
        if len(depth_points) > 0:
            if do_sample:
                if self.sample_depth == 'random':
                    points_idxs = np.random.randint(len(depth_points), size=self.depth_num)
                    depth_points = depth_points[points_idxs]
                else:
                    depth_points = utils.fps(points=depth_points, num=self.depth_num)    
            if self.coord == 'world':
                depth_points = utils.transform_points(depth_points, self.trans_c2w)
        else:
            depth_points = None

        data = {
            'points': depth_points,
            'color': color_img,
            'color_kpt': color_kpt_img,
            'depth': depth_img,
            'mask': mask_img,
            'color_path': color_path,
            'color_kpt_path': color_kpt_path,
            'depth_path': depth_path,
            'mask_path': mask_path
        }
        
        return data

    def get_body(self, frame_id, fit_type, stage_idx=0):
        """get body vertices

        Args:
            frame_id (str): _description_
            fit_type (str): _description_
            stage_idx (int): _description_

        Returns:
            vertices (np): [n, 3], body vertices in world/camera space
            joints (np): [n, 3], body joints in world/camera space
        """
        
        if 'vicon' in self.recording:
            if fit_type == 'mosh':
                fit_root = osp.join(proxq_root, 'fittings/mosh') 
            else:
                fit_root = osp.join(ours_fit_root_q, fit_type)
        else:
            if fit_type == 'lemo':
                fit_root = lemo_fit_root
            elif fit_type == 'proxd': 
                fit_root = proxd_fit_root
            else:
                fit_root = osp.join(ours_fit_root, fit_type)
        
        fit_dir = osp.join(fit_root, self.recording, 'results')
        if 'ours' in fit_type:
            fit_path = osp.join(fit_dir, frame_id, str(int(stage_idx)) + '00.pkl')
        else:
            fit_path = osp.join(fit_dir, frame_id, '000.pkl')
        if not osp.exists(fit_path):
            return None

        torch_params = utils.load_smplx_params(fit_path)
        output = self.model(return_verts=True, **torch_params)

        #* camera space
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()[:]

        #* world space
        if self.coord == 'world':
            if 'vicon' in self.recording and fit_type == 'mosh':
                vertices = utils.transform_points(vertices, self.trans_m2w)
                joints = utils.transform_points(joints, self.trans_m2w)
            else:
                vertices = utils.transform_points(vertices, self.trans_c2w)
                joints = utils.transform_points(joints, self.trans_c2w)

        data = {
            'vertices': vertices, 
            'joints': joints
        }
        
        return data
    
    def get_body_pc(self, frame_id, fit_type):
        """get body point cloud

        Args:
            frame_id (str): _description_
            fit_type (str): _description_

        Returns:
            points (np): [n, 3], body points in world/camera space
            parts (np): [n], parts of points
        """
            
        body_data = self.get_body(frame_id, fit_type)
        if body_data is None:
            return None
        
        points, idxs = utils.fps(points=body_data['vertices'], num=self.body_num, return_idxs=True)
        parts = np.zeros(self.body_num)

        data = {
            'points': points, 
            'parts': parts
        }
        data.update(body_data)
        
        return data
    
    def load_cast_scene(self):
        self.cast_scene = utils.get_cast_scene(self.scene, self.scene_o3d)

