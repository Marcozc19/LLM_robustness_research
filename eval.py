from models.chatgpt import Model
import importlib.util
import os

def load_evaluation_module(evaluation_file_path):
    # Dynamically import the evaluation module
    module_name = os.path.basename(evaluation_file_path).replace('.py', '')
    
    # Load the module from the file path
    spec = importlib.util.spec_from_file_location(module_name, evaluation_file_path)
    eval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_module)
    
    return eval_module
def eval(config):
    evaluation_file = config["evaluation"]["file"]
    eval_module = load_evaluation_module(evaluation_file)
    # Run the evaluation (assuming the eval function exists in the evaluation module)
    eval_module.main(config)

if __name__ == '__main__':
    chatgpt = Model()
    chatgpt.init()