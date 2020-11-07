import torch
from tqdm import tqdm
import logging
from utils.timer import Timer
from utils.losses import L1, L2
from datasets.basic import *
from base.base_trainer import BaseTrainer

import matplotlib.pyplot as plt

class Trainer(BaseTrainer):
    def __init__(self, optimizer_name='adam', lr=1e-4, n_epochs=1000, lr_milestones=[], batch_size=128, loss_name='L2', log_every=0, debug=False, device='cuda', **kwargs):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, loss_name,
                        log_every, debug, device)
        self.valid_losses = {'L1': L1, 'L2': L2}
        assert loss_name in self.valid_losses.keys()
        self.loss_fn = self.valid_losses[loss_name] 

    def train(self, net, data, timestamp=''):
        loss_plot = []

        net = net.to(self.device)
        net = net.double()

        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        if self.log_every:
            logging.info(f'Optimization with loss: {self.loss_fn.__name__} successfully initialized.')

        loss_args = {'y': []
                     'pred': []}

        loss_epoch = 0

        with Timer():
            net.train()
            t = tqdm(range(self.n_epochs))

            if self.log_every:
                logging.info(f'Optimization started!')

            for epoch in t:
                loss_epoch = 0.
                n_batches = 0

                for data_batch in loader:
                    y = data_batch.to(self.device)
                    optimizer.zero_grad()
                    pred = net(pc)

                    if self.debug and (epoch % self.log_every == 0):
                        pass
                    
                    loss_args['y'] = y
                    loss_args['pred'] = pred
                    
                    loss_dict = self.loss_fn(**loss_args)
                    loss = loss_dict['total']
                    
                    #for name, l in loss_dict['debug'].items():
                        #loss_epoch_dict[name].append(l)

                    loss.backward()
                    optimizer.step()

                    loss_epoch += loss.item()
                    n_batches += 1
            
                #for name, l in loss_epoch_dict.items():
                    #loss_plot_dict[name].append(sum(l)/len(l))
                
                if self.log_every and (epoch % self.log_every == 0):
                    logging.info(f'\tEpoch {epoch+1}/{self.n_epochs}\tLoss: {loss_epoch/n_batches:.8f}')

                scheduler.step()
                
                if epoch in self.lr_milestones:
                    if self.log_every:
                        logging.info(f'\tLR scheduler: new learning rate is {scheduler.get_lr()[0]}')
                    print(f'\tLR scheduler: new learning rate is {scheduler.get_lr()[0]}')
                    
                t.set_description(f'Loss: {loss_epoch/n_batches:.8f}')
                t.refresh()
        
        #return loss_plot_dict
