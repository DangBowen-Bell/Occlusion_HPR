import numpy as np
import os.path as osp
import os
import json
import open3d as o3d
import smplx
import torch
import torch.nn.functional as F
import pickle
import cv2
import sys
import trimesh
from psbody.mesh.visibility import visibility_compute
from psbody.mesh import Mesh
from scipy.spatial import KDTree

sys.path.append('..')
from misc import constants
from misc import fps as fps_lib
from prox.dist_chamfer import chamferDist
chamfer_dist = chamferDist()


#! Opreate data
def transform_points(points, trans):
    """transform points

    Args:
        points (np): [n, 3]
        trans (np): [4, 4]

    Returns:
        new_points (np): [n, 3]
    """
    
    paddings = np.array([[1]] * points.shape[0])
    new_points = np.concatenate([points, paddings], axis=1)
    new_points = np.dot(new_points, trans.T)[:, :3]
    
    return new_points

def fps(points, num, return_idxs=False):
    """fps sampling

    Args:
        points (np): [n, 3]
        num (int): _description_
        return_idxs (bool, optional): _description_. Defaults to False.

    Returns:
        fps_points (np): [n, 3]
    """
    
    _, dtype = get_tensor_type()
    
    points = torch.tensor(points, dtype=dtype)
    points = points.view(1, -1, 3)
    points_cenrtoids = fps_lib.farthest_point_sample(points, num)
    fps_points = points[:, points_cenrtoids[0, :], :]
    fps_points = fps_points.view(-1, 3).numpy()
    
    if return_idxs:
        return fps_points, points_cenrtoids[0, :]
    else:
        return fps_points

def remove_noise(points, radius=1.0):
    """remove noise points out of a ball

    Args:
        points (np): [n, 3]
        radius (float, optional): _description_. Defaults to 1.0.
        
    Returns:
        points (np): [n, 3]
    """
    
    center = np.mean(points, axis=0)
    valid_idxs = np.where(
        (points[:, 0] - center[0])**2 +
        (points[:, 1] - center[1])**2 +
        (points[:, 2] - center[2])**2 < radius**2)
    points = points[valid_idxs]
    
    return points

def get_points_idxs(points, center=np.zeros(3), crop_type='ball', crop_size=1.0):
    """crop points using ball or cube

    Args:
        points (np): [n, 3]
        center (np, optional): [3]. Defaults to np.zeros(3).
        crop_type (str): _description_ 
        crop_size (float): _description_ 

    Returns:
        idxs (np): [n]
    """
    
    size = crop_size
    if crop_type == 'cube':
        idxs = np.where(
            (points[:, 0] >= center[0] - crop_size) &
            (points[:, 0] <= center[0] + crop_size) &
            (points[:, 1] >= center[1] - crop_size) &
            (points[:, 1] <= center[1] + crop_size) & 
            (points[:, 2] >= center[2] - crop_size) &
            (points[:, 2] <= center[2] + crop_size))
    elif crop_type == 'ball':
        idxs = np.where(
            (points[:, 0] - center[0])**2 +
            (points[:, 1] - center[1])**2 +
            (points[:, 2] - center[2])**2 < crop_size**2)
    
    return idxs


#! Generate data
def get_tensor_type():
    """get tensor type

    Returns:
        _type_: _description_
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    return device, dtype

def get_grid_points(res, bb_min=-1.0, bb_max=1.0):
    """generate grid points 

    Args:
        res (int): _description_
        bb_min (float, optional): _description_. Defaults to -1.0.
        bb_max (float, optional): _description_. Defaults to 1.0.
        
    returns:
        points (tensor): [n, 3]
    """
    
    voxel_size = (bb_max - bb_min) / (res - 1)        
    overall_index = torch.arange(0, res**3, 1, out=torch.LongTensor())

    points = torch.zeros(res**3, 3)
    points[:, 0] = ((overall_index.long() / res) / res) % res
    points[:, 1] = (overall_index.long() / res) % res
    points[:, 2] = overall_index % res
    points[:, 0] = bb_min + points[:, 0] * voxel_size
    points[:, 1] = bb_min + points[:, 1] * voxel_size
    points[:, 2] = bb_min + points[:, 2] * voxel_size
    
    return points
    
def get_scene_bb(bb_min, bb_max):
    """get scene bounding box data

    Args:
        bb_min (list): [3]
        bb_max (list): [3]

    Returns:
        bb_vs (list): [[[3], ...], ...]
        bb_fs (list): [[[3], ...], ...]
    """
    
    bb_vs = []
    for x in [bb_min[0], bb_max[0]]:
        for y in [bb_min[1], bb_max[1]]:
            for z in [bb_min[2], bb_max[2]]:
                bb_vs.append([x, y, z])
    bb_vs = [bb_vs]
    bb_fs = [[
        [0, 1, 2], [1, 2, 3],
        [2, 3, 6], [3 ,6, 7],
        [4, 6, 7], [4, 5, 7],
        [0, 4, 5], [0, 1, 5],
        # [0, 2, 4], [2, 4, 6],
        [1, 3, 5], [3, 5, 7]]] 

    return bb_vs, bb_fs

def get_camera_k():
    """get camera intrinsic matrix for prox dataset

    Returns:
        camera_k (np): [3, 3]
    """
    
    camera_k = np.array(
        [[constants.fx, 0,            constants.cx],
         [0,            constants.fy, constants.cy],
         [0,            0,            1]]
    )
    
    return camera_k


#! Load data 
def load_smplx_model(recording):
    """load smplx body model
    
    Returns:
        recording (str): _description_
    """
    
    device, dtype = get_tensor_type()
    
    female_subs = [162, 3452, 159, 3403]
    sub = int(recording.split('_')[1])
    gender = 'female' if sub in female_subs else 'male'
        
    model = smplx.create(model_path=constants.smpl_root, 
                         model_type='smplx', 
                         gender=gender,
                         num_pca_comps=12,
                         create_global_orient=True,
                         create_body_pose=True,
                         create_betas=True,
                         create_left_hand_pose=True,
                         create_right_hand_pose=True,
                         create_expression=True,
                         create_jaw_pose=True,
                         create_leye_pose=True,
                         create_reye_pose=True,
                         create_transl=True).to(device).to(dtype)
    
    return model

def load_smplx_params(path):
    """load smplx params

    Args:
        path (str): _description_

    Returns:
        torch_params (dict): _description_
    """
    
    device, dtype = get_tensor_type()
    
    with open(path, 'rb') as f:
        if 'mosh' in path:  
            params = pickle.load(f, encoding='latin1')
        else:
            params = pickle.load(f)
    
    torch_params = {}
    for key in params.keys():
        if key in ['pose_embedding', 'camera_rotation', 'camera_translation']:
            continue
        else:
            torch_params[key] = torch.tensor(params[key]).to(device).to(dtype)
 
    return torch_params  

def load_trans(path):
    """load transformation matrix

    Args:
        path (str): _description_

    Returns:
        trans (np): [4, 4]
    """
    
    with open(path, 'r') as f:
        trans = np.array(json.load(f))
    return trans

def load_sdf(sdf_path, sdf_info_path, return_tensor=True):
    """load sdf data

    Args:
        sdf_path (str): _description_
        sdf_info_path (str): _description_
        return_tensor (bool): _description_

    Returns:
        sdf (tensor/np): [n, n, n]
        grid_min (tensor/np): _description_
        grid_max (tensor/np): _description_
        grid_dim (int): _description_
    """

    with open(sdf_info_path, 'r') as f:
        sdf_info = json.load(f)
    grid_min = np.array(sdf_info['min'])
    grid_max = np.array(sdf_info['max'])
    grid_dim = sdf_info['dim']
    sdf = np.load(sdf_path).reshape(grid_dim, grid_dim, grid_dim)
    
    if return_tensor:
        device, dtype = get_tensor_type()
        grid_min = torch.tensor(grid_min).to(device).to(dtype)
        grid_max = torch.tensor(grid_max).to(device).to(dtype)
        sdf = torch.tensor(sdf).to(device).to(dtype)
        
    data = {
        'sdf': sdf,
        'min': grid_min,
        'max': grid_max,
        'dim': grid_dim,
    }
    
    return data

def load_scene_bb(scene_bb_path, scene):
    """load scene bounding box data

    Args:
        scene_bb_path (str): _description_
        scene (str): scene name

    Returns:
        bb_vs (list): [[[3], ...], ...]
        bb_fs (list): [[[3], ...], ...]
    """
    
    with open(scene_bb_path, 'r') as f:
        scene_bb = json.load(f)[scene]
    bb_vs = []
    bb_fs = []
    for key in scene_bb.keys():
        v1 = np.array(scene_bb[key]['v1'])
        v2 = np.array(scene_bb[key]['v2'])
        v3 = np.array(scene_bb[key]['v3'])
        v4 = np.array(scene_bb[key]['v4'])
        bb_vs.append([v1, v2, v3, v4])
        bb_fs.append([[0, 1, 2], [1, 2, 3]])
        
    return bb_vs, bb_fs

def load_scene_bb_posa(sdf_info_path):
    """load scene bounding box from posa dataset

    Args:
        sdf_info_path (str): _description_

    Returns:
        bb_min (np): _description_
        bb_max (np): _description_
    """
    
    with open(sdf_info_path, 'r') as f:
        sdf_info = json.load(f)
        
    bb_min = np.array(sdf_info['bbox'][1])
    bb_max = np.array(sdf_info['bbox'][0])
    
    return bb_min, bb_max

def load_vertice_pairs():
    """load vertice pairs

    Returns:
        vertice_pairs (tensor): [n, 2]
    """
    
    device, _ = get_tensor_type()
    
    vertice_pairs = np.load(constants.vertice_pairs_path)
    vertice_pairs = torch.tensor(vertice_pairs, device=device)
    
    return vertice_pairs


#! Projection 
class Projection():
    def __init__(self, dataset, calib_dir, use_color_mask=False, z_thre=1e-2):
        with open(osp.join(calib_dir, 'IR.json'), 'r') as f:
            self.depth_cam = json.load(f)
        with open(osp.join(calib_dir, 'Color.json'), 'r') as f:
            self.color_cam = json.load(f)
        self.dataset = dataset
        self.use_color_mask = use_color_mask
        self.z_thre = z_thre

    def row(self, A):
        return A.reshape((1, -1))

    def col(self, A):
        return A.reshape((-1, 1))

    def unproject_depth_image(self, depth_image):
        # [0,1,...,w-1, ..., 0,1,...,w-1], [w*h]
        us = np.arange(depth_image.size) % depth_image.shape[1]
        # [0,...,0, ..., h-1,...,h-1], [w*h]
        vs = np.arange(depth_image.size) // depth_image.shape[1]
        # [w*h]
        ds = depth_image.ravel()

        # [w*h, 3]
        uvd = np.array(np.vstack((us.ravel(), vs.ravel(), ds.ravel())).T)
        # [w*h, 1, 2]
        xy_undistorted_camspace = cv2.undistortPoints(np.asarray(uvd[:, :2].reshape((1, -1, 2)).copy()),
                                                      np.asarray(self.depth_cam['camera_mtx']),
                                                      np.asarray(self.depth_cam['k']))

        # [w*h, 3]
        xyz_camera_space = np.hstack((xy_undistorted_camspace.squeeze(), self.col(uvd[:, 2])))
        xyz_camera_space[:, :2] *= self.col(xyz_camera_space[:, 2])

        # VISUALIZE
        # viz_pc(xyz_camera_space)

        if self.dataset == 'PROX':
            other_answer = xyz_camera_space - self.row(np.asarray(self.depth_cam['view_mtx'])[:, 3])
            xyz = other_answer.dot(np.asarray(self.depth_cam['view_mtx'])[:, :3])
        else:
            trans_depth2color = np.asarray(self.depth_cam['ext_depth2color'])
            xyz = transform_points(xyz_camera_space, trans_depth2color)
        xyz = xyz.reshape((depth_image.shape[0], depth_image.shape[1], -1))

        # VISUALIZE
        # viz_pc(xyz.reshape(-1, 3))

        return xyz

    def project_points(self, points, cam=None):        
        points = points.reshape((-1, 3)).copy()
        if cam is None or self.dataset == 'EgoBody':
            cam = self.color_cam if cam is None else cam
            return cv2.projectPoints(points, np.asarray([0.0, 0.0, 0.0]), 
                                             np.asarray([0.0, 0.0, 0.0]), 
                                             np.asarray(cam['camera_mtx']),
                                             np.asarray(cam['k']))[0].squeeze()  
        else:
            return cv2.projectPoints(points, np.asarray(cam['R']),
                                             np.asarray(cam['T']),
                                             np.asarray(cam['camera_mtx']),
                                             np.asarray(cam['k']))[0].squeeze() 

    def get_depth_points(self, depth_img, color_img, mask_img=None):
        if not self.use_color_mask and mask_img is not None:
            depth_img[mask_img!=0] = 0

        color_h, color_w = color_img.shape[0], color_img.shape[1]

        # 1.depth image -> color camera
        # [w*h, 3]
        points = self.unproject_depth_image(depth_img).reshape(-1, 3)

        # 2.color camera -> color screen
        # [w*h, 2]
        uvs = self.project_points(points, self.color_cam)
        uvs = np.round(uvs).astype(int)
        # [w*h]
        valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < color_h)
        valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < color_w)
        valid_idx = np.logical_and(valid_x, valid_y)
        if self.use_color_mask and mask_img is not None:        
            valid_mask_idx = valid_idx.copy()
            valid_mask_idx[valid_mask_idx] = mask_img[uvs[valid_idx][:, 1], uvs[valid_idx][:, 0]]==0
            valid_idx = valid_mask_idx.copy()
        points = points[valid_idx]

        if self.dataset == 'PROX':
            T = np.concatenate([np.asarray(self.color_cam['view_mtx']),
                                np.array([0, 0, 0, 1]).reshape(1, -1)])
            stacked = np.column_stack((points, np.ones(len(points))))
            points = np.dot(T, stacked.T).T[:, :3]
            points = np.ascontiguousarray(points)

        valid_idx = points[:, 2] > self.z_thre
        # drop noise using distance of axis z
        # valid_idx = np.logical_and(points[:, 2] > 0.0, points[:, 2] < 4.0)
        points = points[valid_idx]

        # viz_pc(points)

        return points


#! Evaluation
def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re

def get_mpjpe(joints, gt_joints):
    """get mpjpe & mpjpe pa

    Args:
        joints (tensor): [1, N, 3]
        gt_joints (tensor): [1, N, 3]

    Returns:
        mpjpe (float): _description_
        mpjpe_pa (float): _description_
    """
    
    mpjpe = torch.sqrt(((joints - gt_joints)**2).sum(dim=-1)).mean().detach().cpu().numpy() * 1e3
    mpjpe_pa = reconstruction_error(joints.detach().cpu().numpy(), gt_joints.detach().cpu().numpy(), reduction=None)[0] * 1e3
    
    return mpjpe, mpjpe_pa

def get_v2v(vertices, gt_vertices):
    """get v2v & v2v pa

    Args:
        vertices (tensor): [1, N, 3]
        gt_vertices (tensor): [1, N, 3]

    Returns:
        v2v (float): _description_
        v2v_pa (float): _description_
    """
    
    v2v = torch.sqrt(((vertices - gt_vertices)**2).sum(dim=-1)).mean().detach().cpu().numpy() * 1e3
    v2v_pa = reconstruction_error(vertices.detach().cpu().numpy(), gt_vertices.detach().cpu().numpy(), reduction=None)[0] * 1e3
    
    return v2v, v2v_pa

def get_non_col(vertices, sdf_data):
    """compute non-collision score

    Args:
        vertices (np/tensor): [n, 3], vertices in world space
        sdf (tensor): [n, n, n]
        grid_min (tensor): _description_
        grid_max (tensor): _description_
        grid_dim (int): _description_

    Returns:
        non-col (float): _description_
    """
    
    device, dtype = get_tensor_type()
    
    vertices = torch.tensor(vertices).to(device).to(dtype)
    
    sdf = sdf_data['sdf']
    grid_min = sdf_data['min']
    grid_max = sdf_data['max']
    grid_dim = sdf_data['dim']
    
    v_num = len(vertices)
    vertices_norm = (vertices.unsqueeze(0)-grid_min) / (grid_max-grid_min)*2 - 1
    body_sdf = F.grid_sample(sdf.view(1, 1, grid_dim, grid_dim, grid_dim), 
                             vertices_norm[:, :, [2, 1, 0]].view(1, v_num, 1, 1, 3),
                             padding_mode='border')
    pene_num = body_sdf.lt(0).sum().item()
    non_col = 1 - pene_num / v_num
    return non_col

def get_vol_non_col(vertices, sdf_data, vertice_pairs):
    """compute volume non-collision score

    Args:
        vertices (np/tensor): [n, 3], vertices in world space
        sdf_data (dict): _description_
        vertice_pairs (tensor): [n, 2]

    Returns:
        vol_non-col (float): _description_
    """
    
    device, dtype = get_tensor_type()
    
    vertices = torch.tensor(vertices).to(device).to(dtype).unsqueeze(0)
    vol_points = get_volume_points_tensor(vertices, vertice_pairs, use_fit=False)
    vol_points = vol_points.squeeze(0)
    
    # vol_points = vol_points.detach().cpu().numpy()
    # viz_pc(vol_points)
    
    vol_non_col = get_non_col(vol_points, sdf_data)
    
    return vol_non_col
    
def get_valid(body_points, scene_points, val_min_dis=0.1):
    """get close points ratio

    Args:
        body_points (np): [n, 3], body points in world/camera space
        scene_points (np):  [n, 3], scene points in world/camera space

    Returns:
        valid (float): close points ratio
    """
    
    device, dtype = get_tensor_type()
    
    body_points_tensor = torch.tensor(body_points).to(dtype).to(device).unsqueeze(dim=0)
    scene_points_tensor = torch.tensor(scene_points).to(dtype).to(device).unsqueeze(dim=0)
    dist, _, idx, _ = chamfer_dist(body_points_tensor, scene_points_tensor)
    idx = idx.detach().cpu().numpy().squeeze()
    dist = dist.detach().cpu().numpy().squeeze()
    valid = len(idx[dist < val_min_dis]) / len(dist)
    return valid

def get_pm(vertices, depth_points):
    """get partial matching score

    Args:
        vertices (np/tensor): [n, 3], body vertices in world/camera space
        depth_points (np):  [n, 3], depth points in world/camera space

    Returns:
        pm (float): partial matching score
    """
    
    device, dtype = get_tensor_type()
    
    vertices_tensor = torch.tensor(vertices).to(dtype).to(device).unsqueeze(dim=0)
    depth_points_tensor = torch.tensor(depth_points).to(dtype).to(device).unsqueeze(dim=0)
    _, dist, _, _ = chamfer_dist(vertices_tensor, depth_points_tensor)
    pm = dist.mean().item() * 1e3
    
    return pm


#! Other
def viz_pc(points, colors=None):
    """visualize point cloud

    Args:
        points (np): [n, 3]
        colors (np): [n, 3]
    """
    
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pc.colors = o3d.utility.Vector3dVector(colors)
    axis = o3d.geometry.TriangleMesh().create_coordinate_frame()
    
    o3d.visualization.draw_geometries([pc, axis])
    
def viz_mesh(vertices, faces, colors=None):
    """visualize mesh

    Args:
        vertices (np): [n, 3]
        faces (np): [n, 3]
        colors (np): [n, 3]
    """
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    
    #*test crop
    # axis = o3d.geometry.TriangleMesh().create_coordinate_frame()
    # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.05, -2, -1), 
    #                                            max_bound=(0.05, 1, 1))
    # bbox.color = np.zeros(3)
    # mesh = mesh.crop(bbox)
    # o3d.visualization.draw_geometries([mesh, bbox, axis])

def get_frame_ids(recording, 
                  use_quan=False, use_fixed=True):
    """get frame id list

    Args:
        recording (str): _description_
        use_quan (bool, optional): _description_. Defaults to False.
        use_fixed (bool, optional): _description_. Defaults to True.

    Returns:
        frame_ids (list): _description_
    """
    
    if use_quan:
        data_root = constants.proxq_root
        step = 1
    else:
        data_root = constants.prox_root
        step = constants.step
    
    if use_quan or use_fixed:
        recording_dir = osp.join(data_root, 'recordings', recording, 'Color')
        frame_ids = [frame[:-4] for frame in os.listdir(recording_dir)]
        frame_ids = sorted(frame_ids)[::step]
    else:
        frame_ids = constants.prox_frames_test[recording]
        
    return frame_ids

def get_cast_scene(scene_name, scene_o3d):
    """get cast scene

    Args:
        scene_name (str): _description_
        scene_o3d (o3d): _description_

    Returns:
        cast_scene (o3d): _description_
    """
    
    cast_scene = o3d.t.geometry.RaycastingScene()
    scene_o3d_t = o3d.t.geometry.TriangleMesh.from_legacy(scene_o3d)
    cast_scene.add_triangles(scene_o3d_t)
    
    #* get bounding box of the scene
    if scene_name == 'vicon':
        bb_min = [-3.48, -3.62, -0.28]
        bb_max = [1.55, 1.69, 1.78]
    else:
        scene_sdf_info_path = osp.join(constants.posa_root, 'sdf', scene_name + '.json')
        bb_min, bb_max = load_scene_bb_posa(scene_sdf_info_path)
    scene_bb_vs, scene_bb_fs = get_scene_bb(bb_min, bb_max)
    
    for i in range(len(scene_bb_vs)):
        scene_bb_o3d = o3d.geometry.TriangleMesh()
        scene_bb_o3d.vertices = o3d.utility.Vector3dVector(scene_bb_vs[i])
        scene_bb_o3d.triangles = o3d.utility.Vector3iVector(scene_bb_fs[i]) 
        scene_bb_o3d_t = o3d.t.geometry.TriangleMesh.from_legacy(scene_bb_o3d)
        cast_scene.add_triangles(scene_bb_o3d_t)
        
    return cast_scene

def get_sv_points(cast_scene, camera_pos, depth_points, 
                  ray_max_depth=0.1, ray_interval=0.01):
    """get shadow volume points

    Args:
        cast_scene (o3d): cast scene with bounding box in [world] space
        camera_pos (np): [3], camera position in [world] space
        depth_points (np): [n, 3]. depth points in [world] space
        ray_max_depth (float): _description_. Defaults to 0.1.
        ray_interval (float): _description_. Defaults to 0.01.

    Returns:
        sv_points (np): [n, 3]. shadow volume points in [world] space
    """
    
    #* Get ray information
    ray_directions = depth_points - np.repeat(camera_pos, len(depth_points), 0)
    ray_origins = depth_points
    for i in range(len(depth_points)):
        ray_directions[i] = ray_directions[i] / np.linalg.norm(ray_directions[i])

    #* Cast rays
    rays = np.concatenate((ray_origins, ray_directions), 1)
    rays = rays.astype(np.float32)
    result = cast_scene.cast_rays(rays)
    hit_dist = result['t_hit'].numpy()
    
    #* Get shadow volume points
    sv_points = []
    for i in range(len(hit_dist)):
        if hit_dist[i] > 999.0:
            continue
        max_depth = min(hit_dist[i], ray_max_depth)
        # max_depth = hit_dist[i]
        #* (1) all points
        d = 0
        while d < max_depth:
            sv_points.append(ray_origins[i] + ray_directions[i] * d)
            d += ray_interval
        #* (2) end points
        # sv_points.append(ray_origins[i] + ray_directions[i] * max_depth)
        
    sv_points = np.array(sv_points)
    
    # sv_pc = trimesh.PointCloud(vertices=sv_points)
    # sv_pc.export('sv.ply')
    
    return sv_points

def get_volume_points():
    """_summary_

    Returns:
        volume_points (np): [n, 3]
    """
    
    model = load_smplx_model('vicon_03301_01')
    output = model(return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    m = Mesh(v=vertices, f=model.faces)
    (vis, n_dot) = visibility_compute(v=m.v, f=m.f, cams=np.array([[0.0, 0.0, 1.0]]))
    vis = vis.squeeze()

    #* calculate visible vertices
    vis_idxs = np.argwhere(vis > 0).squeeze(-1)
    _, idx_idxs = fps(vertices[vis_idxs], num=1024, return_idxs=True)
    vis_idxs = vis_idxs[idx_idxs]
    
    vis_vertices = vertices[vis_idxs]
    not_vis_idxs = np.argwhere(vis == 0).squeeze(-1)
    not_vis_vertices = vertices[not_vis_idxs]
    
    print('Visible vertices: ', vis_idxs.shape)
    print('Not visible vertices: ', not_vis_idxs.shape)
    
    #* establish cast scene
    body_o3d = o3d.geometry.TriangleMesh()
    body_o3d.vertices = o3d.utility.Vector3dVector(vertices)            
    body_o3d.triangles = o3d.utility.Vector3iVector(model.faces) 
    body_o3d.compute_vertex_normals()
    cast_scene = o3d.t.geometry.RaycastingScene()
    body_o3d_t = o3d.t.geometry.TriangleMesh.from_legacy(body_o3d)
    cast_scene.add_triangles(body_o3d_t)

    #* cast rays
    ray_origins = vis_vertices.copy()
    ray_origins[:, 2] -= 0.001
    ray_directions = np.zeros((len(ray_origins), 3))
    ray_directions[:, 2] = -1
    rays = np.concatenate((ray_origins, ray_directions), 1).astype(np.float32)
    result = cast_scene.cast_rays(rays)
    hit_dist = result['t_hit'].numpy()
    
    #* query nearest vertices
    kdtree = KDTree(not_vis_vertices)
    vertice_pairs = []
    for i in range(len(hit_dist)):
        if hit_dist[i] < 10:
            location = ray_origins[i] + ray_directions[i] * hit_dist[i]
            dists, neighbors = kdtree.query(location)
            vertice_pairs.append([vis_idxs[i], not_vis_idxs[neighbors]])
    vertice_pairs = np.array(vertice_pairs)
    if not osp.exists(constants.vertice_pairs_path):
        np.save(constants.vertice_pairs_path, vertice_pairs)
    print('paired vertices: ', vertice_pairs.shape)
    
    #* calculate volume points
    volume_den = constants.volume_den_fit
    volume_points_list = []
    for a_i in range(volume_den):
        a = a_i / (volume_den-1)        
        points = a*vertices[vertice_pairs[:, 0]] + (1-a)*vertices[vertice_pairs[:, 1]]
        volume_points_list.append(points)
    volume_points = np.concatenate(volume_points_list, axis=0)
    print('Volume points: ', volume_points.shape)
    # viz_pc(volume_points)
    
    #* ------------------------------------------------------------
    #* find point to illustrate
    min_distance = float('inf')
    nearest_idx = -1
    for i, point in enumerate(volume_points):
        distance = np.linalg.norm(point)
        if distance < min_distance:
            min_distance = distance
            nearest_idx = i
    idx = nearest_idx % len(vertice_pairs)
    
    for i in range(volume_den):
        volume_points_list.append(np.array([volume_points_list[i][idx]]))
        volume_points_list[i] = np.delete(volume_points_list[i], idx, axis=0)
    pc_o3ds = []
    for i in range(volume_den*2):
        if i < volume_den:
            color = [[1, 0, 0]]
        else:
            color = [[0, 1, 0]]
        
        points = volume_points_list[i]
        colors = color * len(volume_points_list[i])
        
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)
        pc_o3ds.append(pc)
             
        # pc = trimesh.PointCloud(vertices=points)
        # pc.export(str(i) + '.ply')
    # o3d.visualization.draw_geometries(pc_o3ds)
    #* ------------------------------------------------------------

def get_volume_points_tensor(vertices, vertice_pairs, use_fit=True):
    """_summary_

    Args:
        vertices (tensor): [1, n, 3]
        vertice_pairs (tensor): [n, 2]
        volume_den (int): _description_

    Returns:
        volume_points (tensor): [1, n, 3]
    """
    
    device, dtype = get_tensor_type()
    
    pair_num = len(vertice_pairs)
    volume_den = constants.volume_den_fit
    volume_points = torch.zeros((1, pair_num*volume_den, 3), dtype=dtype, device=device)
    for a_i in range(volume_den):
        a = a_i / (volume_den-1)
        volume_points[:, a_i*pair_num:(a_i+1)*pair_num, :] = a*vertices.index_select(1, vertice_pairs[:, 0]) + (1-a)*vertices.index_select(1, vertice_pairs[:, 1])
        
    return volume_points
        
def model_to_ext(model_mat):
    """convert model matrix to extrinsic matrix
 
    Args:
        model_mat (np): [4, 4]

    Returns:
        ext_mat (np): [4, 4]
    """
    
    to_gl_camera = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    from_gl_camera = np.linalg.inv(to_gl_camera)
    ext_mat = np.linalg.inv(model_mat @ from_gl_camera)
    
    return ext_mat
    
def get_prox_split_info(prox_info_dict):
    """_summary_

    Args:
        prox_info_dict (dict): _description_

    Returns:
        prox_split_info_dict (dict): {'train': [], 'test': []}
    """
        
    info_list = []
    min_valid_ratio = 1. / 4.
    for key, val in prox_info_dict.items():
        if val[0] >= min_valid_ratio:
            info_list.append(key)
    info_list = sorted(info_list)
    np.random.seed(2023)
    test_idxs = np.random.choice(len(info_list), size=len(info_list)//5, replace=False)
    
    prox_split_info_dict = {'train': [],
                            'test': []}
    for idx in range(len(info_list)):
        if idx not in test_idxs:
            prox_split_info_dict['train'].append(info_list[idx])
        else:
            prox_split_info_dict['test'].append(info_list[idx])
            
    return prox_split_info_dict
