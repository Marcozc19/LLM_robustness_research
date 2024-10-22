import json
import numpy as np
import pandas as pd
import os
from . import truthful_qa
from . import distortion 

data_source = {
    "truthful_qa": truthful_qa.main()
}

class Data:
    def __init__(self, config):
        self.config = config
        self.distortion_type = self.config["distortion"]["type"]
        self.file_path = self.get_filepath()
        self.data = self.load_data()
        processor = distortion.DistortionProcessor(self.data, self.distortion_type, distortion_percentage=0.3)
        self.data = processor.apply_distortions()

    def get_filepath(self):
        dataset_path = "data/" +  str(self.config['dataset']['name']) + "/" +str(self.config['dataset']['name']) + "_" + str(self.config['distortion']['type']) + ".json"
        return dataset_path



    def load_data(self):
        if os.path.exists(self.file_path):
            df = pd.read_json(self.file_path)
        else:
            if self.config["dataset"]["name"] in data_source:
                df = data_source[self.config["dataset"]["name"]]
            else:
                raise ValueError(f"Dataset {self.config['dataset']['name']} not found. Please choose one of {list(data_source.keys())}.")
            # save the data to a json file
            df.to_json(self.file_path, orient='records')
        return df
    