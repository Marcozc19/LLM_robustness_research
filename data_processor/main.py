import json
import numpy as np
import pandas as pd
import os
from . import truthful_qa
from . import distortion 
import util

data_source = {
    "truthful_qa": truthful_qa.main()
}

class Data:
    def __init__(self, config):
        self.config = config
        self.distortion_percentage = 0.3
        self.distortion_type = self.config["distortion"]["type"]
        self.file_path = util.get_dataset_path(config)
        self.data = self.load_data()

    def load_data(self):
        print("================ Loading Data ================")
        if os.path.exists(self.dataset_path):
            df = pd.read_json(self.dataset_path)
        else:
            if self.config["dataset"]["name"] in data_source:
                df = data_source[self.config["dataset"]["name"]]
            else:
                raise ValueError(f"Dataset {self.config['dataset']['name']} not found. Please choose one of {list(data_source.keys())}.")
            # save the data to a json file
            processor = distortion.DistortionProcessor(df, self.distortion_type, self.distortion_percentage)
            df = processor.apply_distortions()
            df.to_json(self.file_path, orient='records')
        return df.iloc[:,:]
    