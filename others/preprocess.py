import yaml
import argparse
from datasets import Preprocessor, monkey


if __name__ == "__main__":
    
    with open('configs/monkey/preprocess.yaml','r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
    dataset = monkey(config)
    dataset.write_summary()
        