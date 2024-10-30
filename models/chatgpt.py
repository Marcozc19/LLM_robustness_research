from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd


class Model():
    def __init__(self, config, data):
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.tokenizer.padding_side = 'left' 
        self.data = data.load_data()

    def batching(self):
        
        pass

    def query(self):
        if isinstance(self.data, pd.DataFrame):
            queries = self.data['question'].tolist()  # Extract all questions into a list
        else:
            raise ValueError("Data is not in expected format (DataFrame).")
        inputs = self.tokenizer(queries, return_tensors="pt", padding= True, truncation=True)
        attention_mask = inputs['attention_mask'] 


        outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask = attention_mask,
            do_sample=True,
            max_new_tokens=50,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        for query, response in zip(queries, responses):
            print(f"Query: {query}")
            print(f"Response: {response}")
            print("="*50)



        return queries, responses
