import torch
import yaml
import os
from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from timm.utils import AverageMeter

from utils import check_dir
from dataset.toy_2d import Toy_Dataset
from model.diffusion.diffusion import DDPM

with open("./config_diffusion.yaml", 'r') as f:
    config = yaml.safe_load(f)

base_dir = "./log/diffusion/25gaussians"
device = "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, '{}.log'.format(timestamp)), filemode='a', format='%(asctime)s - %(message)s')
print(config)
logging.info(config)
x, y = torch.meshgrid(torch.linspace(-4,4,50), torch.linspace(-4,4,50))
fixed_noise = torch.stack([x,y],dim=2)
fixed_noise = fixed_noise.reshape(-1,2).to(device)

train_dataset = Toy_Dataset(data_name="25gaussians")
train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
model = DDPM(config)
model.model = torch.load(base_dir+"final.pt", map_location=device)

_, tmp = model.sample(fixed_noise)
process = tmp['process'].detach().cpu()
denoising = tmp['denoising'].detach().cpu()
for i in range(0,process.shape[1],50):
    plt.scatter(train_dataset.data[:, 0], train_dataset.data[:, 1], color='blue', label='True', s=2, alpha=0.5)
    plt.scatter(process[:,i,0], process[:,i,1], color='red', label='Fake', s=2, alpha=0.5)
    plt.quiver(process[:,i,0], process[:,i,1], -denoising[:,i,0], -denoising[:,i,1])
    plt.xlim((-4,4))
    plt.ylim((-4,4))
    plt.grid()
    plt.savefig(os.path.join(base_dir, "final_{}.png".format(str(i+1).zfill(4))))
    plt.close()

