from torch.utils.data import Dataset
import os
import numpy as np
import imp
import trimesh
import torch
import os.path as osp
import json

from lib.chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()

import utils

fz_data_root = '/data/FZNet'


class FZDataset(Dataset):
    def __init__(self, dataset, split, options):
        data_dir = osp.join(fz_data_root, dataset)
        split_info_path = data_dir + '_split_info.json'
        with open(split_info_path, 'r') as f:
            split_info_dict = json.load(f)      
        infos = split_info_dict[split]

        #* parameters
        self.surf_num = int(options.surf_ratio * options.sample_num)
        self.space_num = options.sample_num - self.surf_num
        self.sigmas = [options.sigma, options.sigma*0.1]
        
        self.infos = infos
        self.dataset = dataset
        self.split = split
        self.data_dir = data_dir
        self.options = options

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        info = self.infos[idx]
        return self.get_item_udf(info)
    
    
    def get_item_udf(self, info):
        pc_path = osp.join(self.data_dir, info.replace('-', '/', 1) + '.npz')
        
        pc_udf_data = self.get_pc_udf_data(pc_path)

        data = {'info': info}
        data.update(pc_udf_data)
        data['scene_input'] = data['scene'] 
        data['depth_input'] = data['depth']
    
        return data
    
    def get_pc_udf_data(self, pc_path):
        #* load pc data
        pc_data = np.load(pc_path)
        scene = pc_data['scene']
        body = pc_data['body']
        body_p = pc_data['body_p']
        depth = pc_data['depth']
        
        trans_o2c = pc_data['trans_o2c'] if 'trans_o2c' in pc_data.keys() else np.eye(4)
        trans_c2w = pc_data['trans_c2w'] if 'trans_c2w' in pc_data.keys() else np.eye(4)
        center = pc_data['center'] if 'center' in pc_data.keys() else np.zeros(3)
        orient = pc_data['orient'] if 'orient' in pc_data.keys() else np.zeros(3)
        rotmat = np.eye(4)
        
        #* do augmentation
        if self.split == 'train' and self.options.use_aug:
            angle = np.random.uniform(-np.pi, np.pi)
            rotvec = np.array([0, 0, angle])
            rotmat = utils.rotvec_to_rotmat(rotvec)
            scene = utils.transform_points(scene, rotmat)
            body = utils.transform_points(body, rotmat)
            depth = utils.transform_points(depth, rotmat)

        pc_data = {
            'scene': scene.astype(np.float32),
            'body': body.astype(np.float32),
            'body_p': body_p.astype(np.float32),
            'depth': depth.astype(np.float32),
            'rotmat': rotmat.astype(np.float32)
        }
        
        #* make udf data
        if self.split == 'train' or self.split == 'test':
            udf_data = self.make_udf(pc_data)
        else:
            udf_data = {}
        
        other_data = {
            'trans_o2c': trans_o2c.astype(np.float32),
            'trans_c2w': trans_c2w.astype(np.float32),
            'center': center.astype(np.float32),
            'orient': orient.astype(np.float32)
        }
        pc_data.update(udf_data)
        pc_data.update(other_data)
        return pc_data
    
    def make_udf(self, pc_data):
        body = pc_data['body']
        body_p = pc_data['body_p']
        scene = pc_data['scene']
        
        #* sample points on the body & scene
        body_pc_idx = np.random.randint(len(body), size=self.surf_num//4)
        body_pc = body[body_pc_idx]
        body_pc1 = body_pc + np.random.normal(0, self.sigmas[0], body_pc.shape)
        body_pc2 = body_pc + np.random.normal(0, self.sigmas[1], body_pc.shape)
        scene_pc_idx = np.random.randint(len(scene), size=self.surf_num//4)
        scene_pc = scene[scene_pc_idx]
        scene_pc1 = scene_pc + np.random.normal(0, self.sigmas[0], scene_pc.shape)
        scene_pc2 = scene_pc + np.random.normal(0, self.sigmas[1], scene_pc.shape)        
        space_pc = np.random.rand(self.space_num, 3) * 2 - [1, 1, 1]
        pc = np.concatenate((body_pc1, body_pc2, scene_pc1, scene_pc2, space_pc))
        
        #* calculate udf
        pc_tensor = torch.tensor(pc).to(torch.float32).unsqueeze(dim=0).cuda()
        body_tensor = torch.tensor(body).to(torch.float32).unsqueeze(dim=0).cuda()
        scene_tensor = torch.tensor(scene).to(torch.float32).unsqueeze(dim=0).cuda()
        body_dist, _, body_idx, _ = chamfer_dist(pc_tensor, body_tensor)
        scene_dist, _, _, _ = chamfer_dist(pc_tensor, scene_tensor)
        body_dist = np.transpose(body_dist.detach().cpu().numpy())
        body_idx = np.transpose(body_idx.detach().cpu().numpy())
        body_part = body_p[body_idx]
        scene_dist = np.transpose(scene_dist.detach().cpu().numpy())
        udf = np.concatenate((body_dist, scene_dist, body_part), axis=1)

        udf_data = {
            'xyz': pc.astype(np.float32),
            'udf': udf.astype(np.float32)
        }
        return udf_data


    def get_loader(self):
        data_loader = torch.utils.data.DataLoader(self, batch_size=self.options.batch_size, 
                                                        num_workers=self.options.num_workers, 
                                                        shuffle=(self.split=='train'))
        return data_loader
