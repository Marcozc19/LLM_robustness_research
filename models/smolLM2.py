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
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)#, cache_dir=cache_dir)
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)#, cache_dir=cache_dir)

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
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
        if isinstance(self.data, pd.DataFrame):
            queries = self.data['question'].tolist()
        else:
            raise ValueError("Data is not in expected format (DataFrame).")

        all_responses = []
        all_perplexities = []

        # Process queries in batches
        for question in queries:
            messages = [{"role": "user", "content": question}]
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

            # Tokenize the batch of templated inputs
            input = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)

            # Generate responses
            output = self.model.generate(
                input["input_ids"], 
                attention_mask=input['attention_mask'],
                max_new_tokens=64, 
                temperature=0.2, 
                top_p=0.9, 
                do_sample=True
            )
            output_token = output.tolist()[0]
            # Decode and calculate perplexity directly from the generated responses
            input_id =input["input_ids"]
            decoded_response = self.tokenizer.decode(output_token, skip_special_tokens=True)
            cleaned_response = decoded_response.split("\nassistant\n", 1)[-1].strip()
            all_responses.append(cleaned_response)

                # Calculate perplexity for the generated response
                perplexity = self.calculate_perplexity(cleaned_response)
                all_perplexities.append(perplexity)

                # Print query-response pair and perplexity
                print(f"Query: {input_text}")
                print(f"Response: {cleaned_response}")
                print(f"Perplexity: {perplexity:.4f}")
                print("=" * 50)

        result_df = pd.DataFrame({
            'query': queries,
            'response': all_responses,
            'perplexity': all_perplexities
        })
        return result_df
    
    def calculate_perplexity(self, output, input_ids):
        """Calculate perplexity for a generated response using the raw output."""
        # Tokenize the response
        # Concatenate the input tokens and generated tokens as a single input for perplexity
        input_ids_combined = torch.cat((input_ids, output), dim=-1)

        # Compute the loss with the generated response as the label
        with torch.no_grad():
            outputs = self.model(input_ids_combined, labels=input_ids_combined)
            loss = outputs.loss.item()

        # Calculate perplexity
        perplexity = math.exp(loss)
        return perplexity
    

if __name__ == "__main__":
    checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

    device = "cpu" # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir= "/share/garg/Marco/Models")
    # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
    model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir= "/share/garg/Marco/Models").to(device)

    messages = [{"role": "user", "content": "What is the capital of France."}]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
                inputs, 
                max_new_tokens=50, 
                temperature=0.2, 
                top_p=0.9, 
                do_sample=True)
    print(tokenizer.decode(outputs[0]))

    # model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # SmolLM2 model checkpoint
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # # Load tokenizer and model from Hugging Face
    
    
    # # Sample configuration dictionary

    # # Run the query method and get responses
    # prompt = "Hey, are you conscious? Can you talk to me?"

    # inputs = tokenizer(prompt, return_tensors="pt")
    # generate_ids = model.generate(inputs.input_ids, max_length=30)
    # res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # # Print results
    # print(prompt, res)