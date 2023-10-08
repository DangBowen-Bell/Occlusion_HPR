import os
import json
import argparse
import numpy as np
from collections import namedtuple
import os.path as osp


class UDFOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', default='debug', help='')

        ################################################################################
        #* io
        io = self.parser.add_argument_group('IO Options')
        io.add_argument('--log_dir', default='logs', help='')
        io.add_argument('--ckpt', default=None, help='')
        io.add_argument('--from_json', default=None, help='')
        io.add_argument('--resume', dest='resume', default=False, action='store_true', help='')

        ################################################################################
        #* train
        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=200, help='')
        train.add_argument('--test_steps', type=int, default=500, help='')
        train.add_argument('--batch_size', type=int, default=8, help='')
        train.add_argument('--device_id', type=str, default='0', help='')
        train.add_argument('--usage', type=str, default='train', help='')

        #* optimizer
        train.add_argument('--opt', type=str, default='Adam', help='')
        train.add_argument('--lr', type=float, default=1e-4, help='')
        train.add_argument('--lr_start', type=int, default=100, help='')
   
        #* dataset
        train.add_argument('--train_ds', type=str, default='prox', help='')
        train.add_argument('--test_ds', type=str, default='prox', help='')
        train.add_argument('--sample_num', type=int, default=20000, help='')
        train.add_argument('--surf_ratio', type=float, default=0.95, help='')
        train.add_argument('--sigma', type=float, default=0.02, help='')
        train.add_argument('--num_workers', type=int, default=0, help='')
        train.add_argument('--point_num', type=int, default=4096, help='')
        train.add_argument('--use_aug', default=True, action='store_false', help='')
 
        #* model      
        train.add_argument('--min_max', default=True, action='store_false', help='')
        train.add_argument('--clamp_dis', type=float, default=0.1, help='')

        #* generate        
        train.add_argument('--sel_setting', type=str, default='all', help='')
        train.add_argument('--gen_step', type=int, default=10, help='')
        train.add_argument('--bb_min', type=float, default=-1.0, help='')
        train.add_argument('--bb_max', type=float, default=1.0, help='')        
        train.add_argument('--gen_res', type=int, default=64, help='')
        train.add_argument('--batch_res', type=int, default=32, help='')

    def parse_args(self):
        self.args = self.parser.parse_args()
        if self.args.from_json is not None:
            path_to_json = osp.abspath(self.args.from_json)
            with open(path_to_json, 'r') as f:
                json_args = json.load(f)
                json_args = namedtuple('json_args', json_args.keys())(**json_args)
                return json_args
        else:
            self.args.log_dir = osp.join(osp.abspath(self.args.log_dir), self.args.name)
            if not osp.exists(self.args.log_dir):
                os.makedirs(self.args.log_dir)
            self.args.ckpt_dir = osp.join(self.args.log_dir, 'checkpoints')
            if not osp.exists(self.args.ckpt_dir):
                os.makedirs(self.args.ckpt_dir)
            self.args.summary_dir = osp.join(self.args.log_dir, 'tensorboard')
            self.save_dump()
            return self.args

    def save_dump(self):
        if not osp.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(osp.join(self.args.log_dir, 'config.json'), 'w') as f:
            json.dump(vars(self.args), f, indent=4)
