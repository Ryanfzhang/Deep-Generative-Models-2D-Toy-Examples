import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class VAE(object):
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = nn.Sequential(nn.Linear(2, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU()).to(self.device)
        self.encoder_mean = nn.Linear(256, self.config['hidden_size']).to(self.device)
        self.encoder_var = nn.Linear(256, self.config['hidden_size']).to(self.device)
        self.decoder = nn.Sequential(nn.Linear(self.config["hidden_size"], 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.Tanh(),
                         nn.Linear(256, 2, bias=True)).to(self.device)

        self.optimizer = torch.optim.RMSprop(self.get_parameters(), lr=self.config['lr'])

    def get_parameters(self):
        return list(self.encoder.parameters())+list(self.encoder_mean.parameters())+list(self.encoder_var.parameters())+list(self.decoder.parameters())

    def train(self, batch):
        h = self.encoder(batch)
        mean = self.encoder_mean(h)
        log_var = self.encoder_var(h)
        noise  = torch.randn_like(log_var, device=self.device)
        z = mean + noise * torch.exp(0.5 * log_var)

        recon_x = self.decoder(z)

        self.optimizer.zero_grad()
        loss = F.mse_loss(recon_x, batch) - torch.mean(0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1), dim=0)
        loss.backward()
        self.optimizer.step()
        return loss


