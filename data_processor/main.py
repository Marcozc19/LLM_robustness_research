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
    def __init__(self, config, distortion_type):
        self.config = config
        self.distortion_percentage = config["dataset"]["distortion_percentage"]
        self.distortion_type= distortion_type
        self.dataset_path = util.get_dataset_path(config, distortion_type)
        self.folder_path = util.get_folder_path(config)
        self.data = self.load_data()
        self.output_path = util.get_output_path(config, distortion_type)
        self.eval_path = util.get_eval_path(config, distortion_type)
        self.log_path = util.get_log_path(config, distortion_type)
        
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
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path)
            df.to_json(self.dataset_path, orient='records')
        return df.iloc[:,:]
    