import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BaseModel(nn.Module):
    def __init__(self, optim_config):
        super(BaseModel, self).__init__()
        self.optim_config = optim_config

    def _set_parameter_requires_grad(self):
        mode = self.optim_config['mode']
        if mode == 0:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True

    def configure_optimizers(self):
        self._set_parameter_requires_grad()
        filtered_params = self._filter_required_params()
        if self.opt.adam:
            self.optimizer = optim.Adam(filtered_params, lr=self.optim_config['lr'], betas=(self.optim_config['beta1'], 0.999))
        else:
            self.optimizer = optim.Adadelta(filtered_params, lr=self.optim_config['lr'], rho=self.optim_config['rho'], eps=self.optim_config['eps'])
        return self.optimizer

    def _filter_required_params(self):
        filtered_parameters = []
        for p in filter(lambda p: p.requires_grad, self.parameters()):
            filtered_parameters.append(p)
        return filtered_parameters
