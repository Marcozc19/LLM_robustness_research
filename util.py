import pandas as pd
import argparse
import json
import yaml

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Path to config file")
    args=parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    return config

def read_json(path):
    return pd.read_json(path)

def get_dataset_path(config):
    return "data/" +  str(config['dataset']['name']) + "/" +str(config['dataset']['name']) +"_"+str(config['model']['name'])+ "_" + str(config['distortion']['type']) + "_dataset.json"
def get_output_path(config):
    return "data/" +  str(config['dataset']['name']) + "/" +str(config['dataset']['name'])  + "_"+str(config['model']['name']) + "_" + str(config['distortion']['type']) +"_output.json"
def get_baseline_path(config):
    return "data/" +  str(config['dataset']['name']) + "/" +str(config['dataset']['name'])  + "_"+str(config['model']['name'])+ "_[]_output.json"

if __name__ == '__main__':
    config = load_config()
    print("baseline:", pd.read_json(get_baseline_path(config)))
    print("output:", pd.read_json(get_output_path(config)))