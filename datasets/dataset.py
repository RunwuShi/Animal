import os
import copy
import torch
import random
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from traitlets import Instance

class MelDataset(Dataset):
    def __init__(self, dataset_config, used_key, subset='train'):
        self.dataset_name = dataset_config["dataset"]
        self.preprocessed_path = dataset_config["path"]["preprocessed_path"]
        self.segment_size = dataset_config["preprocessing"]["segment_size"]
        self.chunk_size = dataset_config["preprocessing"]["chunk_size"]
        assert self.segment_size % self.chunk_size == 0
        
        print('dataset marmoset loading...')
        
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
            # mel normal sample 
            mel_sample= np.load(self.preprocessed_path+'/'+mel_path)
            mel_sample_to = torch.from_numpy(mel_sample)
            self.mel_data.append(mel_sample_to)
            
            # individual sample 
            indi_mel_sample = self.mel_process(mel_sample)
            self.indi_mel_data.append(indi_mel_sample)
            
            # length of mel sample
            len = mel_sample.shape[-1]
            len_to = torch.tensor(len, dtype=torch.int32)
            self.mel_len.append(len_to)
            
            # plt.imshow(mel_sample, aspect='auto', origin='lower')
            # plt.show()
            # plt.imshow(indi_mel_sample, aspect='auto', origin='lower')
            # plt.show()
    
    def __len__(self):
        return len(self.mel_len)
            
            
    def __getitem__(self, idx):
        mel= self.mel_data[idx]
        lenx = self.mel_len[idx]
        indi_mel = self.indi_mel_data[idx]
        cID = self.cID_list[idx]
        ctID = self.ctID_list[idx]
        
        return mel, lenx, indi_mel, cID, ctID
    
    
    def chunk_shuffle(self, x, chunk_size):
        """
        :param x: [dim, time]
        :return: shuffled version
        """

        time = x.shape[1]
        dim = x.shape[0]
        x_T = x.T
        
        x_reshaped = np.reshape(x_T, [time // chunk_size, -1, dim]) # [T, C]->[T//chunk_size, _, C] eg [256, 80]->[8, 32, 80]
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
        pos = random.randint(0, mel_data_ext.shape[1] - self.segment_size) # [C, T] eg [80, 1100]
        indi_mel = mel_data_ext[:, pos:pos + self.segment_size]
        indi_mel = self.chunk_shuffle(indi_mel, self.chunk_size)
        indi_mel_t = torch.tensor(indi_mel, dtype=torch.float32)
        return indi_mel_t
    
    
    def plt_fre(self, x):
        plt.imshow(x, aspect='auto', origin='lower')
        plt.show()
   
    def list_depth(self, lst):
        if isinstance(lst, list):
            return 1 + max(self.list_depth(item) for item in lst)
        else:
            return 0 
    
    def process_subset_file(self, data_path, used_key):     
        # total data list
        data_path_list = []
        cID_list = []
        ctID_list = []
        with open(data_path, 'r') as f:
            data_info = json.load(f)
            
        # used key in data_info
        used_key = used_key
        
        # used_key = ['twin_1', 'twin_2', 'twin_3', 'twin_4']
        if self.list_depth(used_key) == 1:
            # print('aaaaaaaaaaaa')
            for key in used_key:
                if key in data_info:
                    for i in data_info[key]:
                        if i['len'] >= 1000 or i['len_mel'] <= 30:
                            continue
                        data_path_list.append(i['path'])
                        cID_list.append(i['cID'])
                        ctID_list.append(i['ctID'])
        else: #[calltype_1]['twin_1_0']
            Dict_1 = used_key[0]
            Dict_2 = used_key[1]
            print('Dict_1',Dict_1)
            print('Dict_2',Dict_2)
            for main_key in Dict_1:
                for sub_key in Dict_2:
                    for i in data_info[main_key][sub_key]:
                        if i['len'] >= 1000 or i['len'] <= 30:
                            continue
                        data_path_list.append(i['path'])
                        cID_list.append(i['cID'])
                        ctID_list.append(i['ctID'])
                    
        return data_path_list, ctID_list, cID_list
    
    
    @staticmethod
    def collate_fn(batch):
        """
           batch: a list of data (fid, mel, mel_ext, mel.shape[1])
           ,      mel, lenx, indi_mel, ctID, cID
        """
        # mel batch
        mels = [data[0] for data in batch]
        mel_batch = [mel.T for mel in mels]
        mel_batch = torch.nn.utils.rnn.pad_sequence(mel_batch, batch_first=True)
        mel_batch = mel_batch.permute((0, 2, 1))        
        # len
        lens_batch = torch.stack([data[1] for data in batch])
        # indi 
        indi_mel_batch = torch.stack([data[2] for data in batch])
        # cid
        cID = [data[3] for data in batch]
        ctID = [data[4] for data in batch]
        
        return mel_batch, lens_batch, indi_mel_batch, cID, ctID
    
class MelDataset_lstm(Dataset):
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
        
        for mel_path in tqdm(self.data_path_list):
            # mel normal sample 
            mel_sample= np.load(self.preprocessed_path+'/'+mel_path)
            self.mel_data.append(mel_sample)
            # length of mel sample
            len = mel_sample.shape[-1]
            self.mel_len.append(len)
            
    
    def __len__(self):
        return len(self.mel_len)
            
            
    def __getitem__(self, idx):
        mel= self.mel_data[idx]
        lenx = self.mel_len[idx]
        # indi_mel = self.indi_mel_data[idx]
        ctID = self.ctID_list[idx]
        cID = self.cID_list[idx]
        
        return mel, lenx, ctID, cID
        
    
    def plt_fre(self, x):
        plt.imshow(x, aspect='auto', origin='lower')
        plt.show()
   
    def list_depth(self, lst):
        if isinstance(lst, list):
            return 1 + max(self.list_depth(item) for item in lst)
        else:
            return 0 
    
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
        if self.list_depth(used_key) == 1:
            # print('aaaaaaaaaaaa')
            for key in used_key:
                if key in data_info:
                    for i in data_info[key]:
                        data_path_list.append(i['path'])
                        ctID_list.append(i['ctID'])
                        cID_list.append(i['cID'])
        else: #[calltype_1]['twin_1_0']
            Dict_1 = used_key[0]
            Dict_2 = used_key[1]
            print('Dict_1',Dict_1)
            print('Dict_2',Dict_2)
            for main_key in Dict_1:
                for sub_key in Dict_2:
                    for i in data_info[main_key][sub_key]:
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
        ctID = np.array([data[2] for data in batch])
        cID = np.array([data[3] for data in batch])
        
        # mel batch
        mel_batch = [torch.from_numpy(mel.T) for mel in mels]
        mel_batch = torch.nn.utils.rnn.pad_sequence(mel_batch, batch_first=True)
        mel_batch = mel_batch.permute((0, 2, 1))
        # length of mel batch
        lens_batch = torch.from_numpy(lens).to(torch.int32)
        
        
        return mel_batch, lens_batch, ctID, cID

class vctkDataset(Dataset):
    def __init__(self, dataset_config, used_key=None, subset='train'):
        self.dataset_name = dataset_config["dataset"]
        self.preprocessed_path = dataset_config["path"]["preprocessed_path"]
        self.segment_size = dataset_config["preprocessing"]["segment_size"]
        self.chunk_size = dataset_config["preprocessing"]["chunk_size"]
        assert self.segment_size % self.chunk_size == 0
        
        # data path info & label info
        used_key = used_key
        data_path = self.preprocessed_path+'/'+'{}_info.json'.format(subset)
        self.data_path_list, self.cID_file_list, self.cID_list \
            = self.process_subset_file(data_path, used_key)
        
        # print(self.data_path_list)
        # load total data
        self.mel_data = []
        self.mel_len = []
        self.indi_mel_data = []
        for mel_path in tqdm(self.data_path_list):
            # mel normal sample 
            mel_sample= np.load(mel_path)
            self.mel_data.append(mel_sample)
            # individual sample 
            indi_mel_sample = self.mel_process(mel_sample)
            self.indi_mel_data.append(indi_mel_sample)
            # length of mel sample
            len = mel_sample.shape[-1]
            self.mel_len.append(len)
            
            # plt.imshow(mel_sample, aspect='auto', origin='lower')
            # plt.show()
            # plt.imshow(indi_mel_sample, aspect='auto', origin='lower')
            # plt.show()
    
    def __len__(self):
        return len(self.mel_len)
            
            
    def __getitem__(self, idx):
        mel= self.mel_data[idx]
        lenx = self.mel_len[idx]
        indi_mel = self.indi_mel_data[idx]
        cID_file = self.cID_file_list[idx]
        cID = self.cID_list[idx]
        
        return mel, lenx, indi_mel, cID_file, cID
    
    
    def chunk_shuffle(self, x, chunk_size):
        """
        :param x: [dim, time]
        :return: shuffled version
        """

        time = x.shape[1]
        dim = x.shape[0]
        x_T = x.T
        
        x_reshaped = np.reshape(x_T, [time // chunk_size, -1, dim]) # [T, C]->[T//chunk_size, _, C] eg [256, 80]->[8, 32, 80]
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
        pos = random.randint(0, mel_data_ext.shape[1] - self.segment_size) # [C, T] eg [80, 1100]
        indi_mel = mel_data_ext[:, pos:pos + self.segment_size]
        indi_mel = self.chunk_shuffle(indi_mel, self.chunk_size)
        return indi_mel
    
    
    def plt_fre(self, x):
        plt.imshow(x, aspect='auto', origin='lower')
        plt.show()
   
    def list_depth(self, lst):
        if isinstance(lst, list):
            return 1 + max(self.list_depth(item) for item in lst)
        else:
            return 0 
    
    def process_subset_file(self, data_path, used_key):     
        # total data list
        data_path_list = [] # npy file path
        cID_file_list = []
        cID_list = []
        with open(data_path, 'r') as f:
            data_info = json.load(f)
        
        # used key in data_info
        used_key = used_key
        
        for key in data_info:
            value = data_info[key]
            for item in value:
                
                cID_file, cID, mel_path, lf0_path, energy_path, *_ = item.strip("\n").split("|")
                # print('mel_path',mel_path)
                
                data_path_list.append(mel_path)
                cID_file_list.append(cID_file)
                cID_list.append(cID) 
            
                    
        return data_path_list, cID_file_list, cID_list
    
    
    @staticmethod
    def collate_fn(batch):
        """
           batch: a list of data (fid, mel, mel_ext, mel.shape[1])
           ,      mel, lenx, indi_mel, ctID, cID
        """
        mels = [data[0] for data in batch]
        lens = np.array([data[1] for data in batch])
        indi_mel = np.array([data[2] for data in batch])
        cID = np.array([data[3] for data in batch])
        
        # mel batch
        mel_batch = [torch.from_numpy(mel.T) for mel in mels]
        mel_batch = torch.nn.utils.rnn.pad_sequence(mel_batch, batch_first=True)
        mel_batch = mel_batch.permute((0, 2, 1))
        # length of mel batch
        lens_batch = torch.from_numpy(lens).to(torch.int32)
        # individual mel batch
        indi_mel_batch = torch.from_numpy(indi_mel).to(torch.float32)
        
        return mel_batch, lens_batch, indi_mel_batch, cID

class greatbirdDataset(Dataset):
    def __init__(self, dataset_config, used_key=None, subset='train'):
        self.dataset_name = dataset_config["dataset"]
        self.preprocessed_path = dataset_config["path"]["preprocessed_path"]
        self.segment_size = dataset_config["preprocessing"]["segment_size"]
        self.chunk_size = dataset_config["preprocessing"]["chunk_size"]
        assert self.segment_size % self.chunk_size == 0
        
        print('dataset greatbird loading...')
        
        # data path info & label info
        used_key = used_key
        data_path = self.preprocessed_path+'/'+'{}_info.json'.format(subset)
        self.data_path_list, self.c_id_index_list, self.cID_list, self.cID_type_list \
            = self.process_subset_file(data_path, used_key)
            
        self.mel_data = []
        self.mel_len = []
        self.indi_mel_data = []
        for mel_path in tqdm(self.data_path_list):
            # mel normal sample 
            mel_sample= np.load(mel_path)
            mel_sample_to = torch.from_numpy(mel_sample)
            self.mel_data.append(mel_sample_to)
            
            # individual sample 
            indi_mel_sample = self.mel_process(mel_sample)
            self.indi_mel_data.append(indi_mel_sample)
            
            # length of mel sample
            len = mel_sample.shape[-1]
            len_to = torch.tensor(len, dtype=torch.int32)
            self.mel_len.append(len_to)
            
        
    def __len__(self):
        return len(self.mel_len)
        
        
    def __getitem__(self, idx):
        mel= self.mel_data[idx]
        lenx = self.mel_len[idx]
        indi_mel = self.indi_mel_data[idx]
        
        cID = self.cID_list[idx] # W67
        cID_type = self.cID_type_list[idx] # W67_0
        
        return mel, lenx, indi_mel, cID, cID_type
    
    
    def chunk_shuffle(self, x, chunk_size):
        """
        :param x: [dim, time]
        :return: shuffled version
        """

        time = x.shape[1]
        dim = x.shape[0]
        x_T = x.T
        
        x_reshaped = np.reshape(x_T, [time // chunk_size, -1, dim]) # [T, C]->[T//chunk_size, _, C] eg [256, 80]->[8, 32, 80]
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
        pos = random.randint(0, mel_data_ext.shape[1] - self.segment_size) # [C, T] eg [80, 1100]
        indi_mel = mel_data_ext[:, pos:pos + self.segment_size]
        indi_mel = self.chunk_shuffle(indi_mel, self.chunk_size)
        indi_mel_t = torch.tensor(indi_mel, dtype=torch.float32)
        return indi_mel_t
    
    
    def plt_fre(self, x):
        plt.imshow(x, aspect='auto', origin='lower')
        plt.show()
       
        
    def list_depth(self, lst):
        if isinstance(lst, list):
            return 1 + max(self.list_depth(item) for item in lst)
        else:
            return 0 
        
        
    def process_subset_file(self, data_path, used_key):     
        # total data list
        data_path_list = [] # npy file path
        c_id_index_list = []
        cID_list = []
        cID_type_list = []
        with open(data_path, 'r') as f:
            data_info = json.load(f)
        
        # used key in data_info
        used_key = used_key
        
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
                    mel_path = item['wav']

                    # data list
                    data_path_list.append(mel_path)
                    # class_id_index eg 20221W67_0_0
                    c_id_index_list.append(class_id_index)
                    # caller id
                    cID_list.append(caller_ID)
                    # caller type 
                    cID_type_list.append(caller_type)
                   
        return data_path_list, c_id_index_list, cID_list, cID_type_list
    
    @staticmethod
    def collate_fn(batch):
        """
            mel, lenx, indi_mel, cID, cID_type, c_id_index
        """
        # mel batch
        mels = [data[0] for data in batch]
        mel_batch = [mel.T for mel in mels]
        mel_batch = torch.nn.utils.rnn.pad_sequence(mel_batch, batch_first=True)
        mel_batch = mel_batch.permute((0, 2, 1))
        # len
        lens_batch = torch.stack([data[1] for data in batch])
        # indi 
        indi_mel_batch = torch.stack([data[2] for data in batch])
        # cid
        cID = [data[3] for data in batch]
        cID_type = [data[4] for data in batch]
        
        return mel_batch, lens_batch, indi_mel_batch, cID, cID_type
    

class finchbird2(Dataset):
    def __init__(self, dataset_config, used_key=None, subset='train'):
        self.dataset_name = dataset_config["dataset"]
        self.preprocessed_path = dataset_config["path"]["preprocessed_path"]
        self.segment_size = dataset_config["preprocessing"]["segment_size"]
        self.chunk_size = dataset_config["preprocessing"]["chunk_size"]
        assert self.segment_size % self.chunk_size == 0
        
        print('dataset finchbird loading...')
        
        # data path info & label info
        used_key = used_key
        data_path = self.preprocessed_path+'/'+'{}_info.json'.format(subset)
        self.data_path_list, self.c_id_index_list, self.cID_list, self.cID_type_list \
            = self.process_subset_file(data_path, used_key)
            
        self.mel_data = []
        self.mel_len = []
        self.indi_mel_data = []
        for mel_path in tqdm(self.data_path_list):
            # mel normal sample 
            mel_sample= np.load(mel_path)
            mel_sample_to = torch.from_numpy(mel_sample)
            self.mel_data.append(mel_sample_to)
            
            # individual sample 
            indi_mel_sample = self.mel_process(mel_sample)
            self.indi_mel_data.append(indi_mel_sample)
            
            # length of mel sample
            len = mel_sample.shape[-1]
            len_to = torch.tensor(len, dtype=torch.int32)
            self.mel_len.append(len_to)
            
        
    def __len__(self):
        return len(self.mel_len)
        
        
    def __getitem__(self, idx):
        mel= self.mel_data[idx]
        lenx = self.mel_len[idx]
        indi_mel = self.indi_mel_data[idx]
        
        cID = self.cID_list[idx] # W67
        cID_type = self.cID_type_list[idx] # W67_0
        
        return mel, lenx, indi_mel, cID, cID_type
    
    
    def chunk_shuffle(self, x, chunk_size):
        """
        :param x: [dim, time]
        :return: shuffled version
        """

        time = x.shape[1]
        dim = x.shape[0]
        x_T = x.T
        
        x_reshaped = np.reshape(x_T, [time // chunk_size, -1, dim]) # [T, C]->[T//chunk_size, _, C] eg [256, 80]->[8, 32, 80]
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
        pos = random.randint(0, mel_data_ext.shape[1] - self.segment_size) # [C, T] eg [80, 1100]
        indi_mel = mel_data_ext[:, pos:pos + self.segment_size]
        indi_mel = self.chunk_shuffle(indi_mel, self.chunk_size)
        indi_mel_t = torch.tensor(indi_mel, dtype=torch.float32)
        return indi_mel_t
    
    
    def plt_fre(self, x):
        plt.imshow(x, aspect='auto', origin='lower')
        plt.show()
       
        
    def list_depth(self, lst):
        if isinstance(lst, list):
            return 1 + max(self.list_depth(item) for item in lst)
        else:
            return 0 
        
        
    def process_subset_file(self, data_path, used_key):     
        # total data list
        data_path_list = [] # npy file path
        c_id_index_list = []
        cID_list = []
        cID_type_list = []
        with open(data_path, 'r') as f:
            data_info = json.load(f)
        
        # used key in data_info
        used_key = used_key
        
        for caller_ID in data_info:
            for caller_type in data_info[caller_ID]:
                for item in data_info[caller_ID][caller_type]:
                    
                    # reduce too len
                    # if item['len_mel'] >= 199 or item['len_mel'] <= 99:
                    #     continue
                    
                    class_id_index = item['class_id_index']
                    mel_path = item['wav']

                    # data list
                    data_path_list.append(mel_path)
                    # class_id_index eg 20221W67_0_0
                    c_id_index_list.append(class_id_index)
                    # caller id
                    cID_list.append(caller_ID)
                    # caller type 
                    cID_type_list.append(caller_type)
                   
        return data_path_list, c_id_index_list, cID_list, cID_type_list
    
    @staticmethod
    def collate_fn(batch):
        """
            mel, lenx, indi_mel, cID, cID_type, c_id_index
        """
        # mel batch
        mels = [data[0] for data in batch]
        mel_batch = [mel.T for mel in mels]
        mel_batch = torch.nn.utils.rnn.pad_sequence(mel_batch, batch_first=True)
        mel_batch = mel_batch.permute((0, 2, 1))
        # len
        lens_batch = torch.stack([data[1] for data in batch])
        # indi 
        indi_mel_batch = torch.stack([data[2] for data in batch])
        # cid
        cID = [data[3] for data in batch]
        cID_type = [data[4] for data in batch]
        
        return mel_batch, lens_batch, indi_mel_batch, cID, cID_type
  
  
class western_bird(Dataset):
    def __init__(self, dataset_config, used_key=None, subset='train'):
        self.dataset_name = dataset_config["dataset"]
        self.preprocessed_path = dataset_config["path"]["preprocessed_path"]
        self.segment_size = dataset_config["preprocessing"]["segment_size"]
        self.chunk_size = dataset_config["preprocessing"]["chunk_size"]
        assert self.segment_size % self.chunk_size == 0
        
        print('dataset winterbird loading...')
        
        # data path info & label info
        used_key = used_key
        data_path = self.preprocessed_path+'/'+'{}_info.json'.format(subset)
        
        # extracte info for dataset
        self.data_path_list, self.c_id_index_list, _, self.cID_type_list \
            = self.process_subset_file(data_path, used_key)
            
        self.mel_data = []
        self.mel_len = []
        self.indi_mel_data = []
        for mel_path in tqdm(self.data_path_list):
            # mel normal sample 
            mel_sample= np.load(mel_path)
            mel_sample_to = torch.from_numpy(mel_sample)
            self.mel_data.append(mel_sample_to)
            
            # individual sample 
            indi_mel_sample = self.mel_process(mel_sample)
            self.indi_mel_data.append(indi_mel_sample)
            
            # length of mel sample
            len = mel_sample.shape[-1]
            len_to = torch.tensor(len, dtype=torch.int32)
            self.mel_len.append(len_to)
            
        
    def __len__(self):
        return len(self.mel_len)
        
        
    def __getitem__(self, idx):
        mel= self.mel_data[idx]
        lenx = self.mel_len[idx]
        indi_mel = self.indi_mel_data[idx]
        
        # cID = self.cID_list[idx] # W67
        cID_type = self.cID_type_list[idx] # W67_0
        
        return mel, lenx, indi_mel, 0, cID_type
    
    
    def chunk_shuffle(self, x, chunk_size):
        """
        :param x: [dim, time]
        :return: shuffled version
        """

        time = x.shape[1]
        dim = x.shape[0]
        x_T = x.T
        
        x_reshaped = np.reshape(x_T, [time // chunk_size, -1, dim]) # [T, C]->[T//chunk_size, _, C] eg [256, 80]->[8, 32, 80]
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
        pos = random.randint(0, mel_data_ext.shape[1] - self.segment_size) # [C, T] eg [80, 1100]
        indi_mel = mel_data_ext[:, pos:pos + self.segment_size]
        indi_mel = self.chunk_shuffle(indi_mel, self.chunk_size)
        indi_mel_t = torch.tensor(indi_mel, dtype=torch.float32)
        return indi_mel_t
    
    
    def plt_fre(self, x):
        plt.imshow(x, aspect='auto', origin='lower')
        plt.show()
       
        
    def list_depth(self, lst):
        if isinstance(lst, list):
            return 1 + max(self.list_depth(item) for item in lst)
        else:
            return 0 
        
        
    def process_subset_file(self, data_path, used_key):     
        # total data list
        data_path_list = [] # npy file path
        c_id_index_list = []
        cID_list = []
        cID_type_list = []
        with open(data_path, 'r') as f:
            data_info = json.load(f)
        
        # used key in data_info
        used_key = used_key
        

        for caller_type in data_info:
            for item in data_info[caller_type]:
                
                # reduce too len
                # if item['len_mel'] >= 199 or item['len_mel'] <= 99:
                #     continue
                
                class_id_index = item['class']
                mel_path = item['wav']

                # data list
                data_path_list.append(mel_path)
                # class_id_index eg 20221W67_0_0
                c_id_index_list.append(class_id_index)
                # caller id
                # cID_list.append(caller_ID)
                # caller type 
                cID_type_list.append(caller_type)
                   
        return data_path_list, c_id_index_list, None, cID_type_list
    
    @staticmethod
    def collate_fn(batch):
        """
            mel, lenx, indi_mel, cID, cID_type, c_id_index
        """
        # mel batch
        mels = [data[0] for data in batch]
        mel_batch = [mel.T for mel in mels]
        mel_batch = torch.nn.utils.rnn.pad_sequence(mel_batch, batch_first=True)
        mel_batch = mel_batch.permute((0, 2, 1))
        # len
        lens_batch = torch.stack([data[1] for data in batch])
        # indi 
        indi_mel_batch = torch.stack([data[2] for data in batch])
        # cid
        # cID = [data[3] for data in batch]
        cID_type = [data[4] for data in batch]
        
        return mel_batch, lens_batch, indi_mel_batch, 0, cID_type
