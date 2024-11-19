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
    evaluation_path = util.get_eval_path(config)
    log_file_path = util.get_log_path(config)
    # Run the evaluation (assuming the eval function exists in the evaluation module)
    results_df, avg_result = evalutor.evaluate(pd.read_csv(pred_ans_path)['response'], pd.read_csv(distorted_ans_path))
    results_df.to_csv(evaluation_path, index=False)
    print(f"Results saved to {evaluation_path}")
    with open(log_file_path, 'a') as log_file:
        log_file.write(str(avg_result) + '\n') 
    return avg_result

if __name__ == '__main__':
    chatgpt = Model()
    chatgpt.init()