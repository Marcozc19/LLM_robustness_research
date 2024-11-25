import argparse
import json
import yaml
from data_processor.main import Data
import eval
import inference
from huggingface_hub import login
import os
from dotenv import load_dotenv


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Path to config file")
    args=parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config):
    # create the dataset if not already created
    data = Data(config)
    # run inference using the dataset created
    inference.main(config, data)
    # print(eval.eval(config))


if __name__ == '__main__':
    # Retrieve token from environment variable
    load_dotenv(dotenv_path='/home/mz572/LLM_robustness_research/.env')
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:   
        raise ValueError("Hugging Face token not found. Please set HUGGING_FACE_HUB_TOKEN.")
    config = load_config()
    print("Running with config:\n", config)
    main(config)
