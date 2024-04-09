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
from model.vae import VAE

with open("./config_vae.yaml", 'r') as f:
    config = yaml.safe_load(f)

base_dir = "./log/vae/25gaussians"
device = "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, '{}.log'.format(timestamp)), filemode='a', format='%(asctime)s - %(message)s')
print(config)
logging.info(config)
fixed_noise = torch.randn([10000//10, config['hidden_size']], device=device)

train_dataset = Toy_Dataset(data_name="25gaussians")
train_data_loader = DataLoader(train_dataset, batch_size=config['epochs'], shuffle=True)
model = VAE(config)
train_process = tqdm(range(config['epochs']))

for epoch in train_process:
    mean_loss = AverageMeter()
    for train_step, batch in enumerate(train_data_loader):
        batch = batch.to(device)
        loss = model.train(batch)
        mean_loss.update(loss.detach().cpu())

    log = "Epoch: {}\t Loss of VAE: {}\t ".format(epoch, mean_loss.avg)
    logging.info(log)
    print(log)
    if (epoch+1)%config['plot_freq']==0:
        fake_samples = model.decoder(fixed_noise)
        fake_samples = fake_samples.cpu().detach().numpy()
        plt.scatter(train_dataset.data[:, 0], train_dataset.data[:, 1], color='blue', label='True', s=2, alpha=0.5)
        plt.scatter(fake_samples[:, 0], fake_samples[:, 1], color='red', label='Fake', s=2, alpha=0.5)
        plt.xlim((-4,4))
        plt.ylim((-4,4))
        plt.grid()
        plt.savefig(os.path.join(base_dir, "{}.png".format(str(epoch+1).zfill(4))))
        plt.close()
