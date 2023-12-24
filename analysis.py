#%%
import os
import yaml
import torch
import argparse
import models
import numpy as np
from tqdm import tqdm
from datasets import MelDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# path
os.chdir('/mnt/work/')

# config path
data_config_path = "./Animal/configs/monkey/dataset.yaml"
dataset_config = yaml.load(open(data_config_path, "r"), Loader=yaml.FullLoader)
analy_set = MelDataset(dataset_config, used_key = ['twin_1', 'twin_2'], subset='test')
analy_loader = DataLoader(analy_set, batch_size=1, shuffle=True)

# device
torch.cuda.set_device(0)
# torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model loading 
model_config_path = "./Animal/configs/monkey/model.yaml"
model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
model_name = model_config['model_name']
model_type = getattr(models, model_name) # model choose
model = model_type(model_config, device).to(device) # model config load

#load model
exp_name = 'VAEbase-c_100.0_1.3-i_10.0_60.0'
model_path = 'Animal/output'
save_path = os.path.join(model_path, exp_name, 'checkpoint', "1000.pth.tar")
ckpt = torch.load(save_path)
model.load_state_dict(ckpt["model"])
model.eval()

# data store
total_z_indi = []
total_z_con = []
total_x_gt = [] # mel
total_x_rec = [] # reconstuct mel
total_ctID = []
total_cID = []

for mel, lenx, indi_mel, ctID, cID in tqdm(analy_loader):
        mel = mel.to(device)
        lenx = lenx.to(device)
        indi_mel = indi_mel.to(device)
        
        outputs = model(mel, lenx, indi_mel) 
        
        z_indi = outputs['z_indi'].detach().cpu().numpy() 
        z_con = outputs['z_con'].detach().cpu().numpy() 
        x_rec = outputs['x_rec'].detach().cpu().numpy() 
        
        total_z_indi.append(z_indi)
        total_z_con.append(z_con)
        total_x_gt.append(mel.detach().cpu().numpy())
        total_x_rec.append(x_rec)
        total_ctID.append(ctID.detach().cpu().numpy())
        total_cID.append(cID.detach().cpu().numpy())
        
        
        
