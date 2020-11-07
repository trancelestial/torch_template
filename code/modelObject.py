import torch
import torch.nn.functional as F
from models.fcNet import FcNet
from trainers.basic import Trainer

class FullyConnected(object):
    def __init__(self, size_in, size_out, layers=[], device='cuda', **kwargs):
        super().__init__()
        self.net = FcNet(size_in, size_out, layers, device, **kwargs)
        self.device = device
        
        self.trainer = None

    def train(self, data, optimizer_name='adam', lr=1e-4, n_epochs=1000, lr_milestones=[],
            batch_size=128, loss_name='L2', log_every=0, debug=False, timestamp='',
            device='cuda', **kwargs):
        self.trainer = FourierTrainer(optimizer_name, lr, n_epochs, lr_milestones, batch_size, loss_name,
            log_every, debug, device, **kwargs)
        
        return self.trainer.train(self.net, data, timestamp)

    def predict(self, data, batch_size=None):
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
        self.net.eval()

        with torch.no_grad():
            predictions = torch.tensor([], device=self.device, dtype=torch.double)

            if batch_size is None:
                predictions = self.net(data.to(self.device))
            else:
                for data_batch in loader:
                    predictions = torch.cat([predictions, self.net(data_batch.to(self.device))])
            
        return predictions
