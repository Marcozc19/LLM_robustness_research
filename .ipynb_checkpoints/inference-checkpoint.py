from models import chatgpt, llama, pythia, smolLM2
import os
import json

def main(config, data):
    model_choice = config["model"]["name"]

    available_models = {
        'chatgpt': chatgpt,
        'llama': llama,
        'pythia': pythia,
        'smollm2': smolLM2
    }

    if model_choice in available_models:
        model = available_models[model_choice].Model(config, data)
    else:
        raise ValueError(f"Model {model_choice} is not available. Please choose one of {list(available_models.keys())}.")
    
    # run the query


    result_df = model.query()

    output_file = data.output_path
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8",newline='') as f:
        result_df.to_csv(f, index=False)


