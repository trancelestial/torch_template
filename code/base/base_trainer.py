from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, optimizer_name, lr, n_epochs, lr_milestones, batch_size, loss_name,
                 log_every, debug, device):
        self.loss_name = loss_name 

        self.optimizer_name=optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size

        self.log_every = log_every
        self.debug = debug
        self.device = device
    
    @abstractmethod
    def train(self, net, data, timestamp=''):
        pass
