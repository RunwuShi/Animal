import os
import json

class monkey:
    def __init__(self, config):
        self.data_dir = config["path"]["origin_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.summary_file = config["path"]["dataset_summary"]

        self.dataset_summary = {}
        self.validate_dir()
        
    def validate_dir(self):
        if not os.path.isdir(self.data_dir):
            raise NotADirectoryError('{} not found!'.format(self.data_dir))
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        return

    
    
    
        