from models import chatgpt, llama

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
    results = model.query()


