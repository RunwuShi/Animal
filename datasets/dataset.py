import os
import copy
import torch
import random
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MelDataset(Dataset):
    def __init__(self, dataset_config, used_key, subset='train'):
        self.dataset_name = dataset_config["dataset"]
        self.preprocessed_path = dataset_config["path"]["preprocessed_path"]
        self.segment_size = dataset_config["preprocessing"]["segment_size"]
        self.chunk_size = dataset_config["preprocessing"]["chunk_size"]
        assert self.segment_size % self.chunk_size == 0
        
        # data path info & label info
        used_key = used_key
        data_path = self.preprocessed_path+'/'+'{}_info.json'.format(subset)
        self.data_path_list, self.ctID_list, self.cID_list \
            = self.process_subset_file(data_path, used_key)
        
        # print(self.data_path_list)
        # load total data
        self.mel_data = []
        self.mel_len = []
        self.indi_mel_data = []
        for mel_path in tqdm(self.data_path_list):
            mel_sample= np.load(self.preprocessed_path+'/'+mel_path)
            self.mel_data.append(mel_sample)
            # plt.imshow(mel_sample, aspect='auto', origin='lower')
            # plt.show()
            indi_mel_sample = self.mel_process(mel_sample)
            self.indi_mel_data.append(indi_mel_sample)
            # plt.imshow(indi_mel_sample, aspect='auto', origin='lower')
            # plt.show()
            len = mel_sample.shape[-1]
            self.mel_len.append(len)
    
    def __len__(self):
        return len(self.mel_len)
            
    def __getitem__(self, idx):
        mel= self.mel_data[idx]
        lenx = self.mel_len[idx]
        indi_mel = self.indi_mel_data[idx]
        ctID = self.ctID_list[idx]
        cID = self.cID_list[idx]
        
        return mel, lenx, indi_mel, ctID, cID
    
    
    def chunk_shuffle(self, x, chunk_size):
        """
        :param x: [dim, time]
        :return: shuffled version
        """

        time = x.shape[1]
        dim = x.shape[0]
        x_T = x.T
        
        x_reshaped = np.reshape(x_T, [time // chunk_size, -1, dim])
        # print(x_reshaped.shape)
        np.random.shuffle(x_reshaped)
        x_shuffled = np.reshape(x_reshaped, [-1, dim]).T
        return x_shuffled
    
    
    def mel_process(self, mel_data):
        # lenth > segment_size
        mel_data_ext = copy.deepcopy(mel_data)
        while mel_data_ext.shape[1] < self.segment_size:
            mel_data_ext = np.concatenate([mel_data_ext, mel_data], axis=1)
        
        # shuffle in time dimension
        pos = random.randint(0, mel_data_ext.shape[1] - self.segment_size)
        indi_mel = mel_data_ext[:, pos:pos + self.segment_size]
        indi_mel = self.chunk_shuffle(indi_mel, self.chunk_size)
        return indi_mel
    
    def plt_fre(self, x):
        plt.imshow(x, aspect='auto', origin='lower')
        plt.show()
    
    def process_subset_file(self, data_path, used_key):     
        # total data list
        data_path_list = []
        ctID_list = []
        cID_list = []
        with open(data_path, 'r') as f:
            data_info = json.load(f)
            
        # used key in data_info
        used_key = used_key
        
        # used_key = ['twin_1', 'twin_2', 'twin_3', 'twin_4']
        for key in used_key:
            if key in data_info:
                for i in data_info[key]:
                    data_path_list.append(i['path'])
                    ctID_list.append(i['ctID'])
                    cID_list.append(i['cID'])
                    
        return data_path_list, ctID_list, cID_list
    
    @staticmethod
    def collate_fn(batch):
        """
           batch: a list of data (fid, mel, mel_ext, mel.shape[1])
           ,      mel, lenx, indi_mel, ctID, cID
        """
        mels = [data[0] for data in batch]
        lens = np.array([data[1] for data in batch])
        indi_mel = np.array([data[2] for data in batch])
        ctID = np.array([data[3] for data in batch])
        cID = np.array([data[4] for data in batch])
        
        # mel batch
        mel_batch = [torch.from_numpy(mel.T) for mel in mels]
        mel_batch = torch.nn.utils.rnn.pad_sequence(mel_batch, batch_first=True)
        mel_batch = mel_batch.permute((0, 2, 1))
        # length of mel batch
        lens_batch = torch.from_numpy(lens).to(torch.int32)
        # individual mel batch
        indi_mel_batch = torch.from_numpy(indi_mel).to(torch.float32)
        
        return mel_batch, lens_batch, indi_mel_batch, ctID, cID
    


