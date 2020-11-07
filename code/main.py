import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
import os
import logging
from pathlib import Path

from utils.timer import Timer
from datetime import datetime

from datasets.basic import BasicDataset
from modelObject import FullyConnected

DATA_PATH = ''
LOGGING_PATH = ''

if __name__ == '__main__':
    logger_path = Path(os.path.abspath(LOGGING_PATH))
    logger_path.touch(exist_ok=True)
    
    logging.basicConfig(filename=logger_path, level=logging.INFO,
                format='%(asctime)s %(message)s', datefmt='%d/%d/%Y %I:%M:%S %p')
    
    data = BasicDataset(DATA_PATH)

    params = {
        'net': {
            'size_in': ,
            'size_out': ,
            'layers': [],
            'n_data': ,
            'device': 'cuda',
        },
        'trainer': {
            'loss_name': 'L2',
            'optimizer_name': 'adam',
            'lr': ,
            'n_epochs': ,
            'lr_milestones': [],
            'batch_size': ,
            'log_every': ,
            'debug': True
        }
    }

    model = FullyConnected(**params['net'])
    model.net.summary()
     
    timestamp = str(datetime.now().strftime("%Y%m%d_%H-%M-%S"))
    model.train(data, timestamp=timestamp, **params['trainer'])

    pred = model.predict(data[:][0], batch_size=None).detach().to('cpu').numpy()