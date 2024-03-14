import json
import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import hearbaseline
from hearbaseline import torchopenl3

# torch.cuda.set_device(1) 
torch.cuda.set_device(0)
used_device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model & embeddings
model = torchopenl3.load_model().to(used_device)
model.eval()

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
# print(model)
print_model_parameters(model)


audio = torch.rand((2, model.sample_rate * 2)).to(used_device)
embeddings = torchopenl3.get_scene_embeddings(audio, model).to("cpu")
print('embeddings',embeddings.shape)

# get the data
refer_info_path = '/mnt/work/dataset/preprocessed/great_bird/data_info'+'/'+'ori_test_info.json'
# refer_info_path = '/mnt/work/dataset/preprocessed/great_bird/data_info'+'/'+'ori_data_info_caller.json'
wav_data_path = '/mnt/work/dataset/preprocessed/great_bird/original_data'

def process_subset_file(data_path):     
    # total data list
    data_path_list = [] # npy file path
    c_id_index_list = []
    cID_list = []
    cID_type_list = []
    with open(data_path, 'r') as f:
        data_info = json.load(f)

    for caller_ID in data_info:
        for caller_type in data_info[caller_ID]:
            for item in data_info[caller_ID][caller_type]:
                
                # reduce too len
                # before 240214 
                # if item['len_mel'] >= 199 or item['len_mel'] <= 99:
                #     continue
                
                # after 240214
                if item['len_mel'] >= 399 or item['len_mel'] <= 99:
                    continue
                
                class_id_index = item['class_id_index']

                # wav file path
                wav_path = item['original_wav']
                
                data_path_list.append(wav_path)
                # class_id_index eg 20221W67_0_0
                # c_id_index_list.append(class_id_index)
                # caller id
                cID_list.append(caller_ID)
                # caller type 
                cID_type_list.append(caller_type)
                
    return data_path_list, cID_list, cID_type_list


data_path_list, cID_list, cID_type_list = process_subset_file(refer_info_path)

# obtain embeddings
def getitem(index):
    import soundfile as sf

    path = data_path_list[index]
    wav, curr_sample_rate = sf.read(path, dtype="float32")
    wav /= np.max(np.abs(wav))

    feats = torch.from_numpy(wav).float()
    return feats

wavs = [getitem(index) for index in range(len(data_path_list))]
ids = [cID_type_list[index] for index in range(len(cID_type_list))]

embedding_total = []
for wav_id in tqdm(range(0, len(wavs))):
    audio = wavs[wav_id].unsqueeze(0).to(used_device)
    embeddings = torchopenl3.get_scene_embeddings(audio, model).squeeze(0).to("cpu").numpy()
    embedding_total.append(embeddings)

# save path
save_path = '/mnt/work/Animal/output/embeddings/torchopenl3/first_ex'

em_save = True
# em_save = False
if em_save:
    print(len(embedding_total))
    save_df = pd.DataFrame(embedding_total)
    save_df.to_csv(os.path.join(save_path, 'embedding.csv'), index=False, header=False)
    print(save_df.head())

    # caller_type_df = pd.DataFrame(ids) 
    # caller_type_df.to_csv(os.path.join(save_path, 'caller_type_label.csv'), index=False, header=False)
    # print(caller_type_df.head())

print('over')


