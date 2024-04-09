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
from model.gan import GAN

with open("./config_gan.yaml", 'r') as f:
    config = yaml.safe_load(f)

base_dir = "./log/gan/8gaussians"
device = "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, '{}.log'.format(timestamp)), filemode='a', format='%(asctime)s - %(message)s')
print(config)
logging.info(config)
fixed_noise = torch.randn([10000//10, config['hidden_size']], device=device)

train_dataset = Toy_Dataset(data_name="8gaussians")
train_data_loader = DataLoader(train_dataset, batch_size=config['epochs'], shuffle=True)
model = GAN(config)
train_process = tqdm(range(config['epochs']))

for epoch in train_process:
    model.generator.train()
    model.discriminator.train()
    mean_d_loss = AverageMeter()
    mean_g_loss = AverageMeter()
    for train_step, batch in enumerate(train_data_loader):
        batch = batch.to(device)
        d_loss = model.train_d(batch)
        mean_d_loss.update(d_loss.detach().cpu())
        if train_step%config['d_updates']==0:
            g_loss = model.train_g()
            mean_g_loss.update(g_loss.detach().cpu())

    log = "Epoch: {}\t Loss of Discriminator: {}\t Loss of Generator: {}".format(epoch, mean_d_loss.avg, mean_g_loss.avg)
    logging.info(log)
    print(log)
    model.epoch+=1
    if (epoch+1)%config['plot_freq']==0:
        model.generator.eval()
        fake_samples = model.generator(fixed_noise)
        fake_samples = fake_samples.cpu().detach().numpy()
        plt.scatter(train_dataset.data[:, 0], train_dataset.data[:, 1], color='blue', label='True', s=2, alpha=0.5)
        plt.scatter(fake_samples[:, 0], fake_samples[:, 1], color='red', label='Fake', s=2, alpha=0.5)
        plt.xlim((-1.5,1.5))
        plt.ylim((-1.5,1.5))
        plt.grid()
        plt.savefig(os.path.join(base_dir, "{}.png".format(str(epoch+1).zfill(4))))
        plt.close()
