import argparse
import json
import yaml
from data_processor.main import Data
import eval
import inference

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
    print(eval.eval(config))


if __name__ == '__main__':
    config = load_config()
    print("Running with config:\n", config)
    main(config)
