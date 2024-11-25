import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
import pandas as pd


class Model():
    def __init__(self, config, data):
        self.config = config
        self.data = data.load_data()
        self.model = LlamaForCausalLM.from_pretrained("C:/Users/zhuan/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf")
        self.tokenizer = LlamaTokenizer.from_pretrained("C:/Users/zhuan/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf")
    
    def query(self):
        if isinstance(self.data, pd.DataFrame):
            queries = self.data['question'].tolist()  # Extract all questions into a list
        else:
            raise ValueError("Data is not in expected format (DataFrame).")
        inputs = self.tokenizer.encode(queries, return_tensors="pt")
        print("input:", inputs)
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask = inputs['attention_mask'],
            do_sample=True,
            max_length=50,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        print("output:", outputs)

        response = [self.tokenizer.batch_decode(output, skip_special_tokens=True)[0] for output in outputs]
        return response
    

if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/share/garg/Marco/Models")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto", cache_dir="/share/garg/Marco/Models")

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])
