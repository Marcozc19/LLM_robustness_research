from models.chatgpt import Model
import importlib.util
import os
from evaluations.truthful_qa import TruthfulQA_Eval
import util
import pandas as pd 

def eval(data,config):
    eval_dict = {
        "truthful_qa": TruthfulQA_Eval
    }

    evalutor = eval_dict[config['dataset']['name']]()
    pred_ans_path = util.get_baseline_path(config)
    distorted_ans_path = data.output_path
    evaluation_path = data.eval_path
    log_file_path = data.log_path
    # Run the evaluation (assuming the eval function exists in the evaluation module)
    results_df, avg_result = evalutor.evaluate(pd.read_csv(pred_ans_path)['response'], pd.read_csv(distorted_ans_path))
    results_df.to_csv(evaluation_path, index=False)
    print(f"Results saved to {evaluation_path}")
    avg_result = pd.DataFrame([avg_result])
    if os.path.exists(log_file_path):
        # Read the existing CSV file
        existing_data = pd.read_csv(log_file_path)
        # Append the new data
        updated_data = pd.concat([existing_data, avg_result], ignore_index=True)
    else:
        # If the file doesn't exist, use the new data as the initial data
        updated_data = avg_result
    updated_data.to_csv(log_file_path, index=False)

if __name__ == '__main__':
    chatgpt = Model()
    chatgpt.init()