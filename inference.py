from models import chatgpt, llama
import json
import util

def main(config, data):
    model_choice = config["model"]["name"]

    available_models = {
        'chatgpt': chatgpt,
        'llama': llama
    }

    if model_choice in available_models:
        model = available_models[model_choice].Model(config, data)
    else:
        raise ValueError(f"Model {model_choice} is not available. Please choose one of {list(available_models.keys())}.")
    
    # run the query


    queries, results = model.query()

    output_file = util.get_output_path(config)

    with open(output_file, "w") as json_file:
            json.dump(results, json_file, indent=4)


