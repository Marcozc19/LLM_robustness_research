from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
import math

class Model:
    def __init__(self, config, data):
        print("================ Initializing SmolLM2 ================")
        # Initialize model configuration and data
        self.config = config
        checkpoint = f"HuggingFaceTB/SmolLM2-{self.config['model']['version']}-Instruct"  # SmolLM2 model checkpoint
        cache_dir = config['cache_dir']['path']

        # Load tokenizer and model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=cache_dir)

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)  # Move model to GPU or CPU
        print("================ SmolLM2 Initialized  ================")
        # Load data
        self.data = data.load_data()
        self.batch_size = config.get("batch_size", 8)  # Set batch size from config or use default

    def batching(self, data_list):
        """Yield successive batches from the data list."""
        for i in range(0, len(data_list), self.batch_size):
            yield data_list[i:i + self.batch_size]

    def query(self):
        print("================ Querying SmolLM2 ================")
        
        # Validate data format
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data is not in the expected format (DataFrame).")
        
        queries = self.data['question'].tolist()  # Extract queries as a list
        all_responses = []
        all_perplexities = []
        
        # Process queries in batches
        for batch_queries in self.batching(queries):
            # Tokenize the batch
            inputs = self.tokenizer(
                batch_queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Adjust max_length as needed
            ).to(self.device)

            # Generate responses for the batch
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=128,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True
                )

            # Decode responses and calculate perplexity for each query
            for query, output in zip(batch_queries, outputs):
                # Decode response
                decoded_response = self.tokenizer.decode(output, skip_special_tokens=True)
                cleaned_response = decoded_response.split("\nassistant\n", 1)[-1].strip()
                all_responses.append(cleaned_response)

                # Calculate perplexity for the response
                tokenized_query = self.tokenizer(query, return_tensors="pt").to(self.device)
                tokenized_response = self.tokenizer(decoded_response, return_tensors="pt").to(self.device)
                perplexity = self.calculate_perplexity(tokenized_response["input_ids"], tokenized_query["input_ids"])
                all_perplexities.append(perplexity)

        # Compile results into a DataFrame
        result_df = pd.DataFrame({
            'query': queries,
            'response': all_responses,
            'perplexity': all_perplexities
        })
        return result_df

    
    def calculate_perplexity(self, output, input_ids):
        """Calculate perplexity for a generated response using the raw output."""
        # Concatenate the input tokens and generated tokens as a single input for perplexity
        input_ids_combined = torch.cat((input_ids, output), dim=-1).to(self.device)

        # Compute the loss with the generated response as the label
        with torch.no_grad():
            outputs = self.model(input_ids_combined, labels=input_ids_combined)
            loss = outputs.loss.item()

        # Calculate perplexity
        perplexity = math.exp(loss)
        return perplexity
    

if __name__ == "__main__":
    checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

    device = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically use GPU if available
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir= "/share/garg/Marco/Models")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir= "/share/garg/Marco/Models").to(device)

    messages = [{"role": "user", "content": "What is the capital of France?"}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs, 
        max_new_tokens=50, 
        temperature=0.2, 
        top_p=0.9, 
        do_sample=True
    )
    print(tokenizer.decode(outputs[0]))
