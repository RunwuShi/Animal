import yaml
import numpy as np
import models
import torch
import os
import matplotlib.pyplot as plt
from datasets import MelDataset
from torch.utils.data import DataLoader


# load path
os.chdir('/mnt/work/')


# test data loader
data_config_path = "./Animal/configs/monkey/dataset4.yaml"
data_config = yaml.load(open(data_config_path, "r"), Loader=yaml.FullLoader)
tst_set = MelDataset(data_config, used_key = [['calltype_1'],['twin_1_0']], subset='test')
tst_single_sampler = DataLoader(tst_set, batch_size=1, shuffle=True)


# # load model test
# model_config = './Animal/configs/monkey/model1.yaml'
# model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
# model_type = getattr(models, model_config['model_name'])
# device = 'cpu'
# model = model_type(model_config, device)


# # load model
# exp_name = 'VAEbase-highindiweight-c_100.0_1.3-i_50.0_60.0'
# model_path = 'Animal/output'
# save_path = os.path.join(model_path, exp_name, 'checkpoint', "70000.pth.tar")
# ckpt = torch.load(save_path)
# model.load_state_dict(ckpt["model"])
# print('model_name:',model_config['model_name'])


# data for input 
# val_mel, val_lenx, val_indi_mel, val_ctID, val_cID = next(iter(tst_single_sampler))
# x = torch.ones(1,80,200)
# lenx = torch.tensor(x.shape[-1]).unsqueeze(0)
# indi_ref = torch.ones(1,80,256)

# def plt_fre(x):
#     plt.imshow(x, aspect='auto', origin='lower')
#     plt.show()

# val_mel = val_mel[0].detach().numpy()
# plt_fre(val_mel)

# val_indi_mel = val_indi_mel[0].detach().numpy()
# plt_fre(val_indi_mel)

# 
# 正常的
# mel, lenx, indi_mel, ctID, cID= next(iter(tst_single_sampler))

# outputs = model(mel, lenx, indi_mel) 
# nll, indi_kl, con_kl = model.loss_fn(outputs, mel, lenx)



# lstm

# load model test
model_config = './Animal/configs/monkey/model_lstm3.yaml'
model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
model_type = getattr(models, model_config['model_name'])
device = 'cpu'
model = model_type(device = device, 
                    module_config = {'static_encoder': model_config['static_encoder']}, 
                    **model_config['DisentangledVAE1D'])

# mel, lenx, indi_mel, ctID, cID= next(iter(tst_single_sampler))
mel = torch.ones(2, 80, 400)
indi_mel = torch.ones(2, 80, 256)
outputs = model(mel, indi_mel) 

loss, kld_f, kld_z = model.loss_fn(original_seq=mel, 
                                   recon_seq=outputs['recon_x'], 
                                   f_mean=outputs['f_mean'], 
                                   f_logvar=outputs['f_logvar'], 
                                   z_post_mean=outputs['z_post_mean'], 
                                   z_post_logvar=outputs['z_post_logvar'], 
                                   z_prior_mean=outputs['z_prior_mean'], 
                                   z_prior_logvar=outputs['z_prior_logvar'])

print('loss',loss)
# out = model(val_mel, val_lenx, val_indi_mel)
# para
# total_params = sum(p.numel() for p in model.parameters())
# for key, value in out.items():
#     print(key, value.shape)


