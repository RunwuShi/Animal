import yaml
from datasets import MelDataset
import numpy as np
import models
import torch

# test data loader
# data_config_path = "./configs/monkey/dataset.yaml"
# data_config = yaml.load(open(data_config_path, "r"), Loader=yaml.FullLoader)
# trn_set = MelDataset(data_config, subset='test')


# load model test
# Open the YAML file
with open('./configs/monkey/model.yaml', 'r') as file:
    # Load the file as a dictionary
    model_config = yaml.safe_load(file)
# print(model_config)

# model = models.VAEbase(model_config)
model_type = getattr(models, model_config['model_name'])
model = model_type(model_config)
print(model_config['model_name'])

x = torch.ones(1,80,200)
lenx = torch.tensor(x.shape[-1]).unsqueeze(0)
indi_ref = torch.ones(1,80,256)
out = model(x, lenx, indi_ref)

# para
total_params = sum(p.numel() for p in model.parameters())

for key, value in out.items():
    print(key, value.shape)
    # print('value', value)

