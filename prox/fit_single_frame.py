# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp

import open3d as o3d

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img
import json
from optimizers import optim_factory

import fitting
from vposer import VPoserDecoder
from psbody.mesh import Mesh
import scipy.sparse as sparse

import dist_chamfer as ext
distChamfer = ext.chamferDist()
import trimesh

sys.path.append('.')
from misc import utils, model


def get_udf_data(udf_model, udf_data):
    """_summary_

    Args:
        udf_model (_type_): _description_
        udf_data (dict): _description_

    Returns:
        udf_data (dict): _description_
    """
    
    #* get body center in [world] space
    body_center = np.array([udf_data['body_center']])
    cam2world = np.array(udf_data['cam2world'])
    body_center = utils.transform_points(body_center, cam2world)[0]
    udf_data['body_center'] = body_center
    
    #* depth & scene in [origin] space
    depth_points = udf_data['depth_points'] - body_center
    scene_points = np.array(udf_data['scene_o3d'].vertices) - body_center
    #* min_z in [origin] space
    min_z = 0.1 - body_center[2]
    
    #* get scene points in a ball
    scene_idxs = utils.get_points_idxs(scene_points)
    scene_points = scene_points[scene_idxs]
    if len(scene_points) > 0:
        points_idxs = np.random.randint(len(scene_points), size=4096)
        scene_points = scene_points[points_idxs]
    else:
        #* UDF loss will be 0
        return udf_data
    udf_data['scene_points'] = scene_points + body_center

    #* run UDF model to get human UDF
    depth = torch.tensor(depth_points, dtype=torch.float32).cuda().unsqueeze(0)
    scene = torch.tensor(scene_points, dtype=torch.float32).cuda().unsqueeze(0)
    
    N = 64
    batch_N = 32
    num_samples = N**3
    max_batch = batch_N**3    

    samples = torch.zeros(N**3, 6)
    samples[:, :3] = utils.get_grid_points(N)

    head = 0
    while head < num_samples:
        tail = min(head+max_batch, num_samples)
        sample_subset = samples[head:tail, :3].cuda().unsqueeze(0)
        udf = udf_model(scene, depth, sample_subset)
        udf = udf.squeeze(-1)
        samples[head:tail, 3:5] = udf.detach().cpu()
        head += max_batch
    samples = samples.detach().cpu().numpy()
    
    #* Get area points in [world] space
    udf_idx = np.where(
        (samples[:, 3] < udf_data['udf_max_dist']) &
        (samples[:, 2] > min_z)
    )
    area_points = samples[:, :3][udf_idx]
    if len(area_points) > 0:
        area_points = area_points + body_center
        udf_data['area_points'] = torch.tensor(area_points, dtype=torch.float32).cuda().unsqueeze(0)
        udf_data['udf'] = samples[:, 3:5]
        udf_data['min_z'] = min_z
    else:
        #* UDF loss will be 0
        return udf_data
    
    return udf_data


def fit_single_frame(img,
                     keypoints,
                     init_trans,
                     scan,
                     scene_name,
                     body_model,
                     camera,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     body_scene_rendering_fn='body_scene.png',
                     body_scene_depth_rendering_fn='body_scene_depth.png',
                     udf_fn='udf.npy',
                     out_img_fn='overlay.png',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     coll_loss_weights=None,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length_x=5000.,
                     focal_length_y=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     ####################
                     ### PROX
                     render_results=True,
                     camera_mode='moving',
                     ## Depth
                     s2m=False,
                     s2m_weights=None,
                     m2s=False,
                     m2s_weights=None,
                     rho_s2m=1,
                     rho_m2s=1,
                     init_mode=None,
                     trans_opt_stages=None,
                     viz_mode='mv',
                     #penetration
                     sdf_penetration=False,
                     sdf_penetration_weights=0.0,
                     sdf_dir=None,
                     cam2world_dir=None,
                     #contact
                     contact=False,
                     rho_contact=1.0,
                     contact_loss_weights=None,
                     contact_angle=15,
                     contact_body_parts=None,
                     body_segments_dir=None,
                     load_scene=False,
                     scene_dir=None,
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'
    body_model.reset_params()
    body_model.transl.requires_grad = True

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # if visualize:
    #     pil_img.fromarray((img * 255).astype(np.uint8)).show()

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)

        vposer = VPoserDecoder(vposer_ckpt=vposer_ckpt, latent_dim=vposer_latent_dim,
                               dtype=dtype, **kwargs)
        vposer = vposer.to(device=device)
        vposer.eval()

    if use_vposer:
        latent_mean = torch.zeros([batch_size, vposer_latent_dim],device=device,
        requires_grad=True, dtype=dtype)
        body_mean_pose = vposer(latent_mean).detach().cpu()
    else:
        body_mean_pose = body_pose_prior.get_mean().detach().cpu()

    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :2]
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    #! Get depth points in [camera] space
    scan_tensor = None
    if scan is not None:
        scan_tensor = torch.tensor(scan.get('points'), device=device, dtype=dtype).unsqueeze(0)

    # load pre-computed signed distance field
    sdf = None
    sdf_normals = None
    grid_min = None
    grid_max = None
    voxel_size = None
    if sdf_penetration:
        with open(osp.join(sdf_dir, scene_name + '.json'), 'r') as f:
            sdf_data = json.load(f)
            grid_min = torch.tensor(np.array(sdf_data['min']), dtype=dtype, device=device)
            grid_max = torch.tensor(np.array(sdf_data['max']), dtype=dtype, device=device)
            grid_dim = sdf_data['dim']
        voxel_size = (grid_max - grid_min) / grid_dim
        sdf = np.load(osp.join(sdf_dir, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
        sdf = torch.tensor(sdf, dtype=dtype, device=device)
        # sdf_normals = np.load(osp.join(sdf_dir, scene_name + '_normals.npy')).reshape(grid_dim, grid_dim, grid_dim, 3)
        # sdf_normals = torch.tensor(sdf_normals, dtype=dtype, device=device)


    with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
        cam2world = np.array(json.load(f))
        world2cam = np.linalg.inv(cam2world)
        R = torch.tensor(cam2world[:3, :3].reshape(3, 3), dtype=dtype, device=device)
        t = torch.tensor(cam2world[:3, 3].reshape(1, 3), dtype=dtype, device=device)

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=point2plane,
                vectorized=True, penalize_outside=penalize_outside)

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)

    # load vertix ids of contact parts
    contact_verts_ids  = ftov = None
    if contact:
        contact_verts_ids = []
        for part in contact_body_parts:
            with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
                data = json.load(f)
                contact_verts_ids.append(list(set(data["verts_ind"])))
        contact_verts_ids = np.concatenate(contact_verts_ids)

        vertices = body_model(return_verts=True, body_pose= torch.zeros((batch_size, 63), dtype=dtype, device=device)).vertices
        vertices_np = vertices.detach().cpu().numpy().squeeze()
        body_faces_np = body_model.faces_tensor.detach().cpu().numpy().reshape(-1, 3)
        m = Mesh(v=vertices_np, f=body_faces_np)
        ftov = m.faces_by_vertex(as_sparse_matrix=True)

        ftov = sparse.coo_matrix(ftov)
        indices = torch.LongTensor(np.vstack((ftov.row, ftov.col))).to(device)
        values = torch.FloatTensor(ftov.data).to(device)
        shape = ftov.shape
        ftov = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

    # Read the scene scan if any
    scene_v = scene_vn = scene_f = None
    if scene_name is not None:
        if load_scene:
            scene = Mesh(filename=os.path.join(scene_dir, scene_name + '.ply'))

            scene.vn = scene.estimate_vertex_normals()

            scene_v = torch.tensor(scene.v[np.newaxis, :],
                                   dtype=dtype,
                                   device=device).contiguous()
            scene_vn = torch.tensor(scene.vn[np.newaxis, :],
                                    dtype=dtype,
                                    device=device)
            scene_f = torch.tensor(scene.f.astype(int)[np.newaxis, :],
                                   dtype=torch.long,
                                   device=device)

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights
    if s2m:
        opt_weights_dict['s2m_weight'] = s2m_weights
    if m2s:
        opt_weights_dict['m2s_weight'] = m2s_weights
    if sdf_penetration:
        opt_weights_dict['sdf_penetration_weight'] = sdf_penetration_weights
    if contact:
        opt_weights_dict['contact_loss_weight'] = contact_loss_weights
    
    #! ############################################################
    #! UDF & RAY loss
    
    if len(scan.get('points')) == 0:
        return
    
    #* get depth points in [world] space
    depth_points = utils.transform_points(scan.get('points'), cam2world)
    points_idxs = np.random.randint(len(depth_points), size=1024)
    depth_points = depth_points[points_idxs]
    
    #* get scene_o3d in [world] space
    scene_path = osp.join(scene_dir, scene_name + '.ply')
    scene_o3d = o3d.io.read_triangle_mesh(scene_path)
    
    #* load vertice pairs
    vertice_pairs = utils.load_vertice_pairs()
    
    other_data = {
        'vertice_pairs': vertice_pairs
    }
    
    #! ############################################################
    #! UDF loss

    udf_data = {
        'scene_o3d': scene_o3d, 
        'cam2world': cam2world,
        'depth_points': depth_points,
        'body_center': None,
        'udf_max_dist': 0,
        'scene_points': None,
        'area_points': None
    }
    
    use_udf = kwargs.get('use_udf', False)
    if use_udf:
        opt_weights_dict['udf_loss_weight'] = kwargs.get('udf_weights')
        udf_max_dists = kwargs.get('udf_max_dists')
        udf_ckpt_path = kwargs.get('udf_ckpt_path') 
        ckpt = torch.load(udf_ckpt_path)
        udf_model = model.FZNet().cuda()
        udf_model.load_state_dict(ckpt['model'])
        udf_model.eval()
        print('Load udf_model successfully')
        
    #! ############################################################
    #! RAY loss
    
    ray_data = {
        'cam_pos': None,
        'cam2scan_dist': None,
        'sv_points': None
    }

    use_ray = kwargs.get('use_ray', False)
    if use_ray:
        opt_weights_dict['ray_dist_loss_weight'] = kwargs.get('ray_dist_weights')
        opt_weights_dict['ray_sv2m_loss_weight'] = kwargs.get('ray_sv2m_weights')
        opt_weights_dict['ray_m2sv_loss_weight'] = kwargs.get('ray_m2sv_weights')
        
        #* camera position in [camera] space
        cam_pos = torch.zeros((1, 1, 3), dtype=dtype, device=device)
        cam2scan_dist = torch.norm(scan_tensor-cam_pos, dim=2).mean()
        ray_data['cam_pos'] = cam_pos
        ray_data['cam2scan_dist'] = cam2scan_dist
        
        #* camera position in [world] space
        cam_pos = utils.transform_points(np.zeros((1, 3)), cam2world)
        sv_start = time.time()
        cast_scene = utils.get_cast_scene(scene_name, scene_o3d)        
        ray_max_depth = kwargs.get('ray_max_depth')
        ray_interval = kwargs.get('ray_interval')
        #* shadown volume in [world] space
        sv_points_np = utils.get_sv_points(cast_scene, cam_pos, depth_points, ray_max_depth, ray_interval)
        sv_time = time.time() - sv_start
        # print('SV done after {:.4f} seconds'.format(sv_time))
        #* shadown volume in [camera] space
        sv_points_np = utils.transform_points(sv_points_np, world2cam)
        sv_points = torch.tensor(sv_points_np, dtype=dtype, device=device).unsqueeze(0)
        ray_data['sv_points'] = sv_points
        
        print('Get sv points successfully')

    #! ############################################################
    
    output_folder = kwargs.get('output_folder', None)
    time_path = osp.join(output_folder, 'time.json')
    cur_frame_name = result_fn.split('/')[-2]
    if osp.exists(time_path):
        with open(time_path, 'r') as f:
            prev_time_data = json.load(f)    
        prev_time_data[cur_frame_name] = []
    else:
        prev_time_data = {cur_frame_name: []}
    
    #! ############################################################

    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # load indices of the head of smpl-x model
    with open( osp.join(body_segments_dir, 'body_mask.json'), 'r') as fp:
        head_indx = np.array(json.load(fp))
    N = body_model.get_num_verts()
    body_indx = np.setdiff1d(np.arange(N), head_indx)
    head_mask = np.in1d(np.arange(N), head_indx)
    body_mask = np.in1d(np.arange(N), body_indx)

    # The indices of the joints used for the initialization of the camera
    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)

    edge_indices = kwargs.get('body_tri_idxs')

    # which initialization mode to choose: similar traingles, mean of the scan or the average of both
    if init_mode == 'scan':
        init_t = init_trans
    elif init_mode == 'both':
        init_t = (init_trans.to(device) + fitting.guess_init(body_model, gt_joints, edge_indices,
                                    use_vposer=use_vposer, vposer=vposer,
                                    pose_embedding=pose_embedding,
                                    model_type=kwargs.get('model_type', 'smpl'),
                                    focal_length=focal_length_x, dtype=dtype) ) /2.0
    else:
        init_t = fitting.guess_init(body_model, gt_joints, edge_indices,
                                    use_vposer=use_vposer, vposer=vposer,
                                    pose_embedding=pose_embedding,
                                    model_type=kwargs.get('model_type', 'smpl'),
                                    focal_length=focal_length_x, dtype=dtype)
    
    if use_ray:
        init_t = np.mean(sv_points_np, axis=0)
    init_t = torch.tensor(init_t, dtype=dtype).view(-1,3)

    camera_loss = fitting.create_loss('camera_init',
                                      trans_estimation=init_t,
                                      init_joints_idxs=init_joints_idxs,
                                      depth_loss_weight=depth_loss_weight,
                                      camera_mode=camera_mode,
                                      dtype=dtype).to(device=device)
    camera_loss.trans_estimation[:] = init_t

    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face, use_hands=use_hands,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               s2m=s2m,
                               m2s=m2s,
                               rho_s2m=rho_s2m,
                               rho_m2s=rho_m2s,
                               head_mask=head_mask,
                               body_mask=body_mask,
                               sdf_penetration=sdf_penetration,
                               voxel_size=voxel_size,
                               grid_min=grid_min,
                               grid_max=grid_max,
                               sdf=sdf,
                               sdf_normals=sdf_normals,
                               R=R,
                               t=t,
                               contact=contact,
                               contact_verts_ids=contact_verts_ids,
                               rho_contact=rho_contact,
                               contact_angle=contact_angle,
                               dtype=dtype,
                               **kwargs)
    loss = loss.to(device=device)

    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, viz_mode=viz_mode, **kwargs) as monitor:

        img = torch.tensor(img, dtype=dtype)

        H, W, _ = img.shape

        # Reset the parameters to estimate the initial translation of the
        # body model
        if camera_mode == 'moving':
            body_model.reset_params(body_pose=body_mean_pose)
            # Update the value of the translation of the camera as well as
            # the image center.
            with torch.no_grad():
                camera.translation[:] = init_t.view_as(camera.translation)
                camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

            # Re-enable gradient calculation for the camera translation
            camera.translation.requires_grad = True

            camera_opt_params = [camera.translation, body_model.global_orient]

        elif camera_mode == 'fixed':
            body_model.reset_params(body_pose=body_mean_pose, transl=init_t)
            camera_opt_params = [body_model.transl, body_model.global_orient]

        # If the distance between the 2D shoulders is smaller than a
        # predefined threshold then try 2 fits, the initial one and a 180
        # degree rotation
        shoulder_dist = torch.dist(gt_joints[:, left_shoulder_idx],
                                   gt_joints[:, right_shoulder_idx])
        try_both_orient = shoulder_dist.item() < side_view_thsh

        camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
            camera_opt_params,
            **kwargs)

        # The closure passed to the optimizer
        fit_camera = monitor.create_fitting_closure(
            camera_optimizer, body_model, camera, gt_joints,
            camera_loss, create_graph=camera_create_graph,
            use_vposer=use_vposer, vposer=vposer,
            pose_embedding=pose_embedding,
            scan_tensor=scan_tensor,
            return_full_pose=False, return_verts=False, 
            udf_data=udf_data, ray_data=ray_data, other_data=other_data)

        # Step 1: Optimize over the torso joints the camera translation
        # Initialize the computational graph by feeding the initial translation
        # of the camera and the initial pose of the body model.
        camera_init_start = time.time()
        cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                fit_camera,
                                                camera_opt_params, body_model,
                                                use_vposer=use_vposer,
                                                pose_embedding=pose_embedding,
                                                vposer=vposer)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            # tqdm.write('Camera initialization done after {:.4f}'.format(
            #     time.time() - camera_init_start))            
            cam_ini_time = time.time() - camera_init_start
            prev_time_data[cur_frame_name].append(cam_ini_time)
            tqdm.write('Camera initialization done after {:.4f}'.format(
                cam_ini_time))
            tqdm.write('Camera initialization final loss {:.4f}'.format(
                cam_init_loss_val))

        # If the 2D detections/positions of the shoulder joints are too
        # close the rotate the body by 180 degrees and also fit to that
        # orientation
        if try_both_orient:
            body_orient = body_model.global_orient.detach().cpu().numpy()
            flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
                cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
            flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

            flipped_orient = torch.tensor(flipped_orient,
                                          dtype=dtype,
                                          device=device).unsqueeze(dim=0)
            orientations = [body_orient, flipped_orient]
        else:
            orientations = [body_model.global_orient.detach().cpu().numpy()]

        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []
        body_transl = body_model.transl.clone().detach()
        # Step 2: Optimize the full model
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()
            
            new_params = defaultdict(transl=body_transl,
                                     global_orient=orient,
                                     body_pose=body_mean_pose)
            body_model.reset_params(**new_params)
            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
                print(opt_idx)
                if opt_idx not in trans_opt_stages:
                    body_model.transl.requires_grad = False
                else:
                    body_model.transl.requires_grad = True
                    
                # if opt_idx not in shape_opt_stages:
                #     body_model.betas.requires_grad = False
                # else:
                #     body_model.betas.requires_grad = True
                
                body_params = list(body_model.parameters())

                final_params = list(
                    filter(lambda x: x.requires_grad, body_params))

                if use_vposer:
                    final_params.append(pose_embedding)

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()

                curr_weights['bending_prior_weight'] = (
                    3.17 * curr_weights['body_pose_weight'])
                if use_hands:
                    joint_weights[:, 25:76] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 76:] = curr_weights['face_weight']
                loss.reset_loss_weights(curr_weights)

                #! ############################################################
                #! UDF loss
                
                udf_data['scene_points'] = None
                udf_data['area_points'] = None
                if use_udf and curr_weights['udf_loss_weight'] > 0.0:
                    udf_data['body_center'] = monitor.body_center
                    udf_data['udf_max_dist'] = udf_max_dists[opt_idx]
                    udf_start = time.time()
                    udf_data = get_udf_data(udf_model, udf_data)
                    udf_time = time.time() - udf_start
                    print('UDF done after {:.4f} seconds'.format(udf_time))
                    print('UDF points number: ', udf_data['area_points'].shape)
                #! ############################################################

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    scan_tensor=scan_tensor,
                    scene_v=scene_v, scene_vn=scene_vn, scene_f=scene_f,ftov=ftov,
                    return_verts=True, return_full_pose=True, 
                    udf_data=udf_data, ray_data=ray_data, other_data=other_data)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                            opt_idx, elapsed))
                    prev_time_data[cur_frame_name].append(elapsed)
                    
                #! save results in the process
                if use_ray and use_udf:
                    tmp_result = {'camera_' + str(key): val.detach().cpu().numpy()
                                  for key, val in camera.named_parameters()}
                    tmp_result.update({key: val.detach().cpu().numpy()
                                       for key, val in body_model.named_parameters()})
                    if use_vposer:
                        tmp_result['pose_embedding'] = pose_embedding.detach().cpu().numpy()
                        body_pose = vposer.forward(pose_embedding).view(1, -1) if use_vposer else None
                        tmp_result['body_pose'] = body_pose.detach().cpu().numpy()
                    tmp_result_fn = result_fn.replace('000.pkl', str(opt_idx+1)+'00.pkl')
                    with open(tmp_result_fn, 'wb') as result_file:
                        pickle.dump(tmp_result, result_file, protocol=2)

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                tqdm.write('Body final loss val = {:.5f}'.format(
                    final_loss_val))
                prev_time_data[cur_frame_name].append(elapsed)

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {'camera_' + str(key): val.detach().cpu().numpy()
                      for key, val in camera.named_parameters()}
            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})
            if use_vposer:
                result['pose_embedding'] = pose_embedding.detach().cpu().numpy()
                body_pose = vposer.forward(pose_embedding).view(1, -1) if use_vposer else None
                result['body_pose'] = body_pose.detach().cpu().numpy()

            results.append({'loss': final_loss_val,
                            'result': result})

        with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)
        
        if use_udf and udf_data['area_points'] is not None:
            np.savez(udf_fn, area=udf_data['area_points'].squeeze(0).detach().cpu().numpy(), 
                     udf=udf_data['udf'], min_z=udf_data['min_z'], center=udf_data['body_center'])

    time_json = json.dumps(prev_time_data)
    with open(time_path, 'w') as f:
        f.write(time_json)

    if save_meshes or visualize:
        body_pose = vposer.forward(pose_embedding).view(1, -1) if use_vposer else None

        model_type = kwargs.get('model_type', 'smpl')
        append_wrists = model_type == 'smpl' and use_vposer
        if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

        model_output = body_model(return_verts=True, body_pose=body_pose)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        import trimesh

        out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
        out_mesh.export(mesh_fn)

    if render_results:
        import pyrender

        #* common
        H, W = 1080, 1920
        camera_center = np.array([951.30, 536.77])
        camera_pose = np.eye(4)
        camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
        camera = pyrender.camera.IntrinsicsCamera(
            fx=1060.53, fy=1060.38,
            cx=camera_center[0], cy=camera_center[1])
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        
        #* rendering body
        img = img.detach().cpu().numpy()
        H, W, _ = img.shape
        body_mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)
        
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        scene.add(body_mesh, 'mesh')

        r = pyrender.OffscreenRenderer(viewport_width=W,
                                       viewport_height=H,
                                       point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        input_img = img
        output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img)
        img = pil_img.fromarray((output_img * 255).astype(np.uint8))
        img.save(out_img_fn)

        #* redering body+scene
        body_mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)
        static_scene = trimesh.load(osp.join(scene_dir, scene_name + '.ply'))
        trans = np.linalg.inv(cam2world)
        static_scene.apply_transform(trans)
        static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)

        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        scene.add(static_scene_mesh, 'mesh')
        scene.add(body_mesh, 'mesh')

        r = pyrender.OffscreenRenderer(viewport_width=W,
                                       viewport_height=H)
        color, _ = r.render(scene)
        color = color.astype(np.float32) / 255.0
        img = pil_img.fromarray((color * 255).astype(np.uint8))
        img.save(body_scene_rendering_fn)
        
        #* rendering body+scene+depth      
        depth_points = utils.transform_points(scan_points_world, trans)  
        
        sphere = trimesh.creation.uv_sphere(radius=0.01)
        sphere.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]
        tfs = np.tile(np.eye(4), (len(depth_points), 1, 1))
        tfs[:,:3,3] = depth_points
        depth_mesh = pyrender.Mesh.from_trimesh(sphere, poses=tfs)
        
        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        scene.add(static_scene_mesh, 'mesh')
        scene.add(body_mesh, 'mesh')
        scene.add(depth_mesh, 'mesh')
        
        r = pyrender.OffscreenRenderer(viewport_width=W,
                                       viewport_height=H)
        color, _ = r.render(scene)
        color = color.astype(np.float32) / 255.0
        img = pil_img.fromarray((color * 255).astype(np.uint8))
        img.save(body_scene_depth_rendering_fn)
        