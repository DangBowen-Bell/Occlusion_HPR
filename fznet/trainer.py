import torch
import torch.optim as optim
import os
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import sys
from tqdm import tqdm
tqdm.monitor_interval = 0

from model import *
from dataset import *
import utils


class BaseTrainer(object):
    def __init__(self, options):
        self.options = options
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ckpt_dir = self.options.ckpt_dir
    
        self.init_fn()
    
        self.saver = utils.CheckpointSaver(save_dir=self.ckpt_dir)
        self.ckpt = None            
        self.epoch_count = 0
        self.step_count = 0
        if self.options.resume and self.saver.exists_checkpoint():
            self.ckpt = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict, checkpoint_file=self.options.ckpt)
            self.epoch_count = self.ckpt['epoch']
            self.step_count = self.ckpt['total_step_count']

        self.min_test_loss = None

        if self.options.usage == 'train':
            self.summary_writer = SummaryWriter(self.options.summary_dir)
        
        if self.options.usage == 'test':
            self.kl_weight = 0.0 
            self.test()
            sys.exit(0)
    
        if self.options.usage == 'generate':
            self.generate()
            sys.exit(0)

    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')


    def train(self):
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), 
                          total=self.options.num_epochs, initial=self.epoch_count):
            train_data_loader = self.train_ds.get_loader()

            lr = self.lr_schedule.get_learning_rate(epoch)
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = lr

            losses_ave = None
            for step, batch in enumerate(tqdm(train_data_loader, 
                                              desc='Epoch '+str(epoch), 
                                              total=len(self.train_ds) // self.options.batch_size)):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

                output, losses, _ = self.train_step(batch)
                if losses_ave is None:
                    losses_ave = losses
                else:
                    for loss_name, val in losses.items():
                        losses_ave[loss_name] += val
     
                self.step_count += 1
                if self.step_count % self.options.test_steps == 0:
                    save_ckpt = self.test()
                    if save_ckpt:
                        for path in glob(self.ckpt_dir + '/*.pt'):
                            os.remove(path)
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step+1, self.options.batch_size, None, self.step_count)
                        tqdm.write('Checkpoint saved')

            for loss_name, val in losses_ave.items():
                losses_ave[loss_name] /= len(train_data_loader)
            self.train_summaries(losses_ave)

    def train_step(self, input):
        raise NotImplementedError('You need to provide a _train_step method')

    def train_summaries(self, losses):
        raise NotImplementedError('You need to provide a _train_summaries method')


    def test(self):
        test_data_loader = self.test_ds.get_loader()

        losses_ave = None
        info_sum = []
        loss_sum = []
        for step, batch in enumerate(tqdm(test_data_loader, 
                                     desc='Test', 
                                     total=len(test_data_loader))):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            output, losses, losses_s = self.test_step(batch)
            if losses_ave is None:
                losses_ave = losses
            else:
                for loss_name, val in losses.items():
                    losses_ave[loss_name] += val
            info_sum.extend(output['info'])
            loss_sum.extend(losses_s['loss'])
    
        for loss_name, val in losses_ave.items():
            losses_ave[loss_name] /= len(test_data_loader)
        self.test_summaries(losses_ave)

        save_ckpt = False
        test_loss = losses_ave['loss']
        if self.min_test_loss is None or test_loss < self.min_test_loss:
            self.min_test_loss = test_loss
            output_path = osp.join(self.ckpt_dir, 'best_pred.npz')
            if osp.exists(output_path):
                os.remove(output_path)
            np.savez(output_path, info=info_sum, loss=loss_sum)
            save_ckpt = True
        
        return save_ckpt

    def test_step(self, input):
        raise NotImplementedError('You need to provide a _test_step method')


    def test_summaries(self, losses):
        raise NotImplementedError('You need to provide a _test_summaries method')

    def generate(self):
        self.mesh_dir = osp.join(self.ckpt_dir, 'mesh_' + self.options.test_ds + '_' + self.options.sel_setting)
        os.makedirs(self.mesh_dir, exist_ok=True)
        self.udf_dir = osp.join(self.ckpt_dir, 'udf_' + self.options.test_ds + '_' + self.options.sel_setting)
        os.makedirs(self.udf_dir, exist_ok=True)
        self.sample_path = osp.join(self.ckpt_dir, 'sample.npy')

        self.options.batch_size = 1
        test_data_loader = self.test_ds.get_loader()
        for step, batch in enumerate(tqdm(test_data_loader, 
                                          desc='Generate', 
                                          total=len(test_data_loader))):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            if step % self.options.gen_step == 0:
                self.generate_step(batch)   
            
    def generate_step(self, input):
        raise NotImplementedError('You need to provide a _generate_step method')


class Trainer(BaseTrainer):
    def init_fn(self):
        self.train_ds = FZDataset(self.options.train_ds, 'train', self.options)
        self.test_ds = FZDataset(self.options.test_ds, 'test', self.options)
        print('train: ', len(self.train_ds))
        print('test: ', len(self.test_ds))

        self.model = FZNet().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters())
        self.lr_schedule = utils.StepLearningRateSchedule(self.options.lr, self.options.lr_start, 0.5)

        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}

        self.l1_none = torch.nn.L1Loss(reduction='none')
        self.l1_sum = torch.nn.L1Loss(reduction='sum')
        self.l1_mean = torch.nn.L1Loss(reduction='mean')

        self.ce_mean = torch.nn.CrossEntropyLoss(reduction='mean')


    def forward_step(self, input):
        clamp_max = self.options.clamp_dis
        
        info = input['info']
        scene = input['scene_input'].to(torch.float32)
        depth = input['depth_input'].to(torch.float32)
        xyz = input['xyz'].to(torch.float32)
        udf = input['udf'].to(torch.float32)
        udf = udf[:, :, :2]
        batch_size = xyz.shape[0]
        batch_sample_num = batch_size * xyz.shape[1]

        pred_udf = self.model(scene, depth, xyz)            
        
        if self.options.min_max: 
            udf = torch.clamp(udf, max=clamp_max)
            pred_udf = torch.clamp(pred_udf, max=clamp_max)

        loss_rec_body = self.l1_sum(pred_udf[:, :, 0], udf[:, :, 0]) / batch_sample_num
        loss_rec_scene = self.l1_sum(pred_udf[:, :, 1], udf[:, :, 1]) / batch_sample_num
        loss = loss_rec_body + loss_rec_scene

        losses = {'loss': loss,
                  'loss_rec_body': loss_rec_body,
                  'loss_rec_scene': loss_rec_scene}

        losses_s = {'loss': np.zeros(batch_size)}
        output = {'info': info}

        return output, losses, losses_s


    def train_step(self, input):
        self.model.train()

        output, losses, losses_s = self.forward_step(input)
        loss = losses['loss']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for key in losses.keys():
            losses[key] = losses[key].detach().item()
        return output, losses, losses_s

    def train_summaries(self, losses):
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)


    def test_step(self, input):
        self.model.eval()

        output, losses, losses_s = self.forward_step(input)

        for key in losses.keys():
            losses[key] = losses[key].detach().item()
        return output, losses, losses_s

    def test_summaries(self, losses):
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar('test_' + loss_name, val, self.step_count)


    def generate_step(self, input):
        self.model.eval()

        info = input['info'][0]
        scene = input['scene_input'].to(torch.float32)
        depth = input['depth_input'].to(torch.float32)
        
        #* generate grid points
        N = self.options.gen_res
        bb_min = self.options.bb_min
        bb_max = self.options.bb_max
        num_samples = N**3
        voxel_size = (bb_max - bb_min) / (N - 1)        
        max_batch = self.options.batch_res**3
        overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
        samples = torch.zeros(N**3, 6)
        samples[:, 0] = ((overall_index.long() / N) / N) % N
        samples[:, 1] = (overall_index.long() / N) % N
        samples[:, 2] = overall_index % N
        samples[:, 0] = bb_min + samples[:, 0] * voxel_size
        samples[:, 1] = bb_min + samples[:, 1] * voxel_size
        samples[:, 2] = bb_min + samples[:, 2] * voxel_size

        if not osp.exists(self.sample_path):
            np.save(self.sample_path, samples[:, :3].numpy())

        #* forward
        head = 0
        while head < num_samples:
            tail = min(head+max_batch, num_samples)
            sample_subset = samples[head:tail, :3].cuda().unsqueeze(0)
            udf = self.model(scene, depth, sample_subset)    
            udf = udf.squeeze(0)
            samples[head:tail, 3:5] = udf.detach().cpu()            
            head += max_batch
        
        #* save data
        body = input['body'][0].detach().cpu().numpy()
        scene = input['scene'][0].detach().cpu().numpy()
        depth = input['depth'][0].detach().cpu().numpy()
        trans_o2c = input['trans_o2c'][0].detach().cpu().numpy()
        trans_c2w = input['trans_c2w'][0].detach().cpu().numpy()
        center = input['center'][0].detach().cpu().numpy()
        orient = input['orient'][0].detach().cpu().numpy()
        rotmat = input['rotmat'][0].detach().cpu().numpy()
        udf_path = osp.join(self.udf_dir, info + '.npz')
        np.savez(udf_path, samples=samples[:, 3:].numpy(), body=body, scene=scene, depth=depth, trans_o2c=trans_o2c, trans_c2w=trans_c2w, center=center, orient=orient, rotmat=rotmat)
