import yaml
from datasets import MelDataset
import numpy as np

data_config_path = "./configs/monkey/dataset.yaml"

# mel start
data_config = yaml.load(open(data_config_path, "r"), Loader=yaml.FullLoader)

trn_set = MelDataset(data_config, subset='test')



# a = [1,2,3]

# for i in a:
#     print(i)


# mel_sample= np.load('G:/work_code/Acoustic/dataset/preprocessed/monkey/set_by_twin/twin_1/T1_m1_ctID2_cID0_vID641.npy')
# print(mel_sample)
