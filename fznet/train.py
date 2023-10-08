import os

from options import *
from trainer import *


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    options = UDFOptions().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.device_id
    trainer = Trainer(options)
    trainer.train()