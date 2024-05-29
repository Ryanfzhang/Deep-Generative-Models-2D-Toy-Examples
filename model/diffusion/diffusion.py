import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.diffusion.noise_schedule import coef

class DDPM(object):
    def __init__(self,config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.alpha = coef(self.config['num_steps'], self.config['noise_schedule'])
        self.beta = 1 - self.alpha
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        model = ToyNet(self.config)
        self.model = model.to(self.device)
        self.alpha_torch = torch.tensor(self.alpha_torch).float().to(self.device).unsqueeze(1)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = self.config['lr'])

    def forward(self, x, set_t=-1):
        B = x.shape[0]
        if set_t!= -1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.config['num_steps'], [B]).to(self.device)

        current_alpha = self.alpha_torch[t].detach()  # (B,)
        noise = torch.randn_like(x)
        noisy_data = (current_alpha ** 0.5) * x + (1.0 - current_alpha) ** 0.5 * noise
        noisy_data = noisy_data.to(self.device)

        predicted = self.model(noisy_data, t)
        loss = nn.functional.mse_loss(predicted, noise)
        return loss

    def train(self, x):
        self.optimizer.zero_grad()
        loss = self.forward(x)
        loss.backward()
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def sample(self, noise):
        
        current_sample_process = []
        current_sample_denoising = []
        current_sample = noise
        B = noise.shape[0]
        for t in range(self.config['num_steps'] - 1, -1, -1):
            predicted  = self.model(current_sample, (torch.ones(B) * t).long().to(self.device))
            current_sample_process.append(current_sample)
            current_sample_denoising.append(predicted)

            coeff1 = 1 / self.alpha[t] ** 0.5
            coeff2 = (1 - self.alpha[t]) / (1 - self.alpha_hat[t]) ** 0.5
            current_sample = coeff1 * (current_sample - coeff2 * predicted)
            if t > 0:
                noise = torch.randn_like(current_sample)
                sigma = (
                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                ) ** 0.5
                current_sample += sigma * noise

        current_sample_process = torch.stack(current_sample_process, dim=1)
        current_sample_denoising = torch.stack(current_sample_denoising, dim=1)

        return current_sample, {"process":current_sample_process, "denoising": current_sample_denoising}

class ToyNet(nn.Module):
    def __init__(self, config):
        super(type(self), self).__init__()
        self.config = config
        self.time_embed_dim = 128
        self.out_dim = 2
        self.hidden_dim = 256

        self.time_embedding = DiffusionEmbedding(self.config['num_steps'], self.time_embed_dim, self.hidden_dim)
        self.input_projection = nn.Linear(self.out_dim, self.hidden_dim)
        self.model = self.build_res_block()
        self.out_projection = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.out_dim))

    def forward(self, x, t):
        t = self.time_embedding(t)
        x = self.input_projection(x)
        x = x + t
        e = self.model(x)
        e = self.out_projection(e)
        return e

    def build_res_block(self):
        hid = self.hidden_dim
        layers = []
        widths =[hid]*4
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            layers.append(nn.SiLU())
        return nn.Sequential(*layers)

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table
