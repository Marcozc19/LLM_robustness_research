from models.chatgpt import Model
import importlib.util
import os
from evaluations.truthful_qa import TruthfulQA_Eval
import util
import pandas as pd

def eval(config):
    eval_dict = {
        "truthful_qa": TruthfulQA_Eval
    }

    evalutor = eval_dict[config['dataset']['name']]()
    pred_ans_path = util.get_baseline_path(config)
    distorted_ans_path = util.get_output_path(config)
    # Run the evaluation (assuming the eval function exists in the evaluation module)

    return evalutor.evaluate(pd.read_json(pred_ans_path), pd.read_json(distorted_ans_path))
if __name__ == '__main__':
    chatgpt = Model()
    chatgpt.init()