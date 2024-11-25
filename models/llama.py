from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import math
import torch

class Model:
    def __init__(self, config, data):
        print("================ Initializing LLaMA 3.2 ================")
        self.config = config
        # self.model_id = config["model"]["name"]  # Specify model ID from config
        self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
        cache_dir = config["cache_dir"]["path"]  # Cache directory path

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=cache_dir)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=cache_dir,
                torch_dtype="auto",
                device_map="auto"
            )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Batch size configuration
        self.batch_size = 8
        self.data = data.load_data()  # Load data using the provided `data` object
        print("================ LLaMA 3.2 Initialized ================")

    def batching(self, data_list):
        """Yield successive batches from the data list."""
        for i in range(0, len(data_list), self.batch_size):
            yield data_list[i:i + self.batch_size]

    def calculate_perplexity(self, input_ids, output_ids):
        """Calculate perplexity for a given input-output sequence."""
        # Concatenate input tokens and output tokens
        input_ids_combined = torch.cat((input_ids, output_ids), dim=-1).to(self.model.device)

        # Compute the loss with the combined sequence as the labels
        with torch.no_grad():
            outputs = self.model(input_ids_combined, labels=input_ids_combined)
            loss = outputs.loss.item()

        # Perplexity calculation
        perplexity = math.exp(loss)
        return perplexity

    def query(self):
        print("================ Querying LLaMA 3.2 ================")
        if isinstance(self.data, pd.DataFrame):
            queries = self.data['question'].tolist()  # Extract all questions into a list
        else:
            raise ValueError("Data is not in expected format (DataFrame).")

        all_responses = []
        all_perplexities = []

        for question in queries:

            messages = [
                {"role": "system", "content": "You are a helpful AI chatbot that will provide accurate answers to every question asked."},
                {"role": "user", "content": question},
            ]

            # Use pipeline for batch generation
            results = self.pipe(messages, max_new_tokens=128, batch_size=self.batch_size)

        

            # Collect responses and calculate perplexity        
            response = results[0]["generated_text"][-1]['content']
            all_responses.append(response)

            # Tokenize the response to get input_ids and output_ids
            inputs = self.tokenizer(question, return_tensors="pt").to(self.model.device)
            outputs = self.tokenizer(response, return_tensors="pt").to(self.model.device)

            # Calculate perplexity for the generated response
            perplexity = self.calculate_perplexity(inputs["input_ids"], outputs["input_ids"])
            all_perplexities.append(perplexity)


        # for batch in self.batching(queries):
        #     batch_prompts = []

        #     # Format batch queries into prompts
        #     for question in batch:
        #         messages = [
        #             {"role": "system", "content": "You are a helpful AI chatbot that will provide accurate answers to every question asked."},
        #             {"role": "user", "content": question},
        #         ]
        #         # Format messages for pipeline input
        #         batch_prompts.append(messages)

        #     # Use pipeline for batch generation
        #     results = self.pipe(batch_prompts, max_new_tokens=128, batch_size=self.batch_size)

        #     # Collect responses and calculate perplexity
        #     for result, question in zip(results, batch):
        #         response = result[0]["generated_text"][-1]['content']
        #         all_responses.append(response)

        #         # Tokenize the response to get input_ids and output_ids
        #         inputs = self.tokenizer(question, return_tensors="pt").to(self.model.device)
        #         outputs = self.tokenizer(response, return_tensors="pt").to(self.model.device)

        #         # Calculate perplexity for the generated response
        #         perplexity = self.calculate_perplexity(inputs["input_ids"], outputs["input_ids"])
        #         all_perplexities.append(perplexity)

        result_df = pd.DataFrame({
            "query": queries,
            "response": all_responses,
            "perplexity": all_perplexities,
        })
        return result_df


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
        {"role": "system", "content": "You are a helpful AI chatbot that will provide accurate answers to every question asked."},
        {"role": "user", "content": "Who are you?"},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1]['content'])
    

