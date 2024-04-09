import torch
from torch import nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np

class GAN(object):
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = nn.Sequential(nn.Linear(self.config["hidden_size"], 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.Tanh(),
                         nn.Linear(256, 2, bias=True)).to(self.device)

        self.discriminator = nn.Sequential(nn.Linear(2, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256, bias=True),
                         nn.LeakyReLU(),
                         nn.Linear(256, 1, bias=True),
                         nn.Sigmoid()).to(self.device)

        self.generator_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr = self.config['lr'])
        self.discriminator_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr = self.config['lr'])
        self.epoch = 0
        self.bce_loss = nn.BCELoss()

    def train_d(self, batch):
        bs = batch.size(0)
        noise = torch.randn([bs, self.config['hidden_size']], device=self.device)
        self.discriminator_optimizer.zero_grad()
        self.generator_optimizer.zero_grad()
        with torch.no_grad():
            fake_samples = self.generator(noise)
        real_pre = self.discriminator(batch)
        fake_pre = self.discriminator(fake_samples)
        d_loss = self.get_d_loss(real_pre, fake_pre)
        d_loss.backward()
        self.discriminator_optimizer.step()
        return d_loss

    def train_g(self):
        noise = torch.randn([self.config['batch_size'], self.config['hidden_size']], device=self.device)
        fake_samples = self.generator(noise)
        fake_pre = self.discriminator(fake_samples)

        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        g_loss = self.get_g_loss(fake_pre)
        g_loss.backward()
        self.generator_optimizer.step()
        return g_loss

    def get_d_loss(self, real_pre, fake_pre):
        return self.bce_loss(real_pre, torch.ones_like(real_pre, device=self.device)) + self.bce_loss(fake_pre, torch.zeros_like(fake_pre, device=self.device))

    def get_g_loss(self, fake_pre):
        return self.bce_loss(fake_pre, torch.ones_like(fake_pre, device=self.device))
