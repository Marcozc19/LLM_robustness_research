import transformers
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import pandas as pd


class Model():
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    
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
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    print("tokenizer loaded")
    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    print(inputs)
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    print(generate_ids)
    res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(res)


    

