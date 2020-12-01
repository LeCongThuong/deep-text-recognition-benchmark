import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BaseModel(nn.Module):
    def __init__(self, opt):
        """
        Args:
            opt: config for setting up optimizer and freeze some layers(if necessary)
        """
        super(BaseModel, self).__init__()
        self.optim_config = opt
        self.optimizer = self.configure_optimizers()

    def set_parameter_requires_grad(self):
        mode = self.optim_config['mode']
        if mode == 0:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True

    def configure_optimizers(self):
        filtered_params = self._filter_required_params()
        if len(filtered_params) == 0:
            return None
        if self.optim_config['optimizer'] == 'adam':
            optimizer = optim.Adam(filtered_params, lr=self.optim_config['lr'], betas=(self.optim_config['beta1'], 0.999))
        else:
            optimizer = optim.Adadelta(filtered_params, lr=self.optim_config['lr'], rho=self.optim_config['rho'], eps=self.optim_config['eps'])
        return optimizer

    def _filter_required_params(self):
        filtered_parameters = []
        for p in filter(lambda p: p.requires_grad, self.parameters()):
            filtered_parameters.append(p)
        return filtered_parameters
