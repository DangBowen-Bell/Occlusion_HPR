import os
import datetime
import torch
import numpy as np
from scipy.spatial.transform import Rotation


class CheckpointSaver():
    """Class that handles saving and loading checkpoints during training."""
    def __init__(self, save_dir, save_steps=1000):
        self.save_dir = os.path.abspath(save_dir)
        self.save_steps = save_steps
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.get_latest_checkpoint()
        return

    def exists_checkpoint(self, checkpoint_file=None):
        """Check if a checkpoint exists in the current directory."""
        if checkpoint_file is None:
            return False if self.latest_checkpoint is None else True
        else:
            return os.path.isfile(checkpoint_file)
    
    def save_checkpoint(self, models, optimizers, epoch, batch_idx, batch_size, dataset_perm, total_step_count):
        """Save checkpoint."""
        timestamp = datetime.datetime.now()
        # checkpoint_filename = os.path.abspath(os.path.join(self.save_dir, timestamp.strftime('%Y_%m_%d-%H_%M_%S') + '.pt'))
        checkpoint_filename = os.path.abspath(os.path.join(self.save_dir, 'best_model.pt'))
        checkpoint = {}
        for model in models:
            checkpoint[model] = models[model].state_dict()
        for optimizer in optimizers:
            checkpoint[optimizer] = optimizers[optimizer].state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['batch_idx'] = batch_idx
        checkpoint['batch_size'] = batch_size
        checkpoint['dataset_perm'] = dataset_perm
        checkpoint['total_step_count'] = total_step_count
        print(timestamp, 'Epoch:', epoch, 'Iteration:', batch_idx)
        print('Saving checkpoint file [' + checkpoint_filename + ']')
        torch.save(checkpoint, checkpoint_filename) 
        return checkpoint_filename

    def load_checkpoint(self, models, optimizers, checkpoint_file=None):
        """Load a checkpoint."""
        if checkpoint_file is None:
            print('Loading latest checkpoint [' + self.latest_checkpoint + ']')
            checkpoint_file = self.latest_checkpoint
        checkpoint = torch.load(checkpoint_file)
        for model in models:
            if model in checkpoint:
                models[model].load_state_dict(checkpoint[model])
        for optimizer in optimizers:
            if optimizer in checkpoint:
                optimizers[optimizer].load_state_dict(checkpoint[optimizer])
        return {'epoch': checkpoint['epoch'],
                'batch_idx': checkpoint['batch_idx'],
                'batch_size': checkpoint['batch_size'],
                'dataset_perm': checkpoint['dataset_perm'],
                'total_step_count': checkpoint['total_step_count']}

    def get_latest_checkpoint(self):
        """Get filename of latest checkpoint if it exists."""
        checkpoint_list = [] 
        for dirpath, dirnames, filenames in os.walk(self.save_dir):
            for filename in filenames:
                if filename.endswith('.pt'):
                    checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
        checkpoint_list = sorted(checkpoint_list)
        self.latest_checkpoint =  None if (len(checkpoint_list) == 0) else checkpoint_list[-1]
        return


def transform_points(points, trans):
    paddings = np.array([[1]] * points.shape[0])
    new_points = np.concatenate([points, paddings], axis=1)
    new_points = np.dot(new_points, trans.T)
    return new_points[:, :3]


class StepLearningRateSchedule():
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


def rotvec_to_rotmat(rotvec):
    r = Rotation.from_rotvec(rotvec)
    rotmat = np.eye(4)
    rotmat[:3, :3] = r.as_matrix()
    return rotmat
