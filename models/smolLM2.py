from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
import math
from torch.utils.data import Dataset, DataLoader



class CustomDataset(Dataset):
    """Custom Dataset for handling queries with templating."""
    def __init__(self, queries, tokenizer, max_length=512):
        self.queries = queries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        # Get the raw query
        question = self.queries[idx]

        # Apply the template: prepend the user role message
        messages = [{"role": "user", "content": question}]
        templated_query = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Tokenize the templated query
        tokens = self.tokenizer(
            templated_query,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "question": question,  # For debugging or logging
            "templated_query": templated_query,  # Optional, useful for debugging
        }

class Model:
    def __init__(self, config, data):
        print("================ Initializing SmolLM2 ================")
        self.config = config
        checkpoint = f"HuggingFaceTB/SmolLM2-{self.config['model']['version']}-Instruct"
        cache_dir = config['cache_dir']['path']

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=cache_dir)

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print("================ SmolLM2 Initialized ================")

        # Load data and prepare DataLoader
        self.batch_size = config.get("batch_size", 8)
        self.data = data.load_data()

        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data is not in the expected format (DataFrame).")

        # Create the dataset and dataloader
        self.dataset = CustomDataset(self.data['question'].tolist(), self.tokenizer)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def query(self):
        print("================ Querying SmolLM2 ================")
        all_responses = []
        all_perplexities = []
        all_questions = []

        for batch in self.dataloader:
            # Move batch data to GPU
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            questions = batch["question"]

            with torch.no_grad():
                # Generate responses
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True
                )

            for question, output in zip(questions, outputs):
                # Decode response
                decoded_response = self.tokenizer.decode(output, skip_special_tokens=True)
                cleaned_response = decoded_response.split("\nassistant\n", 1)[-1].strip()
                all_questions.append(question)
                all_responses.append(cleaned_response)

                # Calculate perplexity for the response
                tokenized_query = self.tokenizer(question, return_tensors="pt").to(self.device)
                tokenized_response = self.tokenizer(decoded_response, return_tensors="pt").to(self.device)
                perplexity = self.calculate_perplexity(tokenized_response["input_ids"], tokenized_query["input_ids"])
                all_perplexities.append(perplexity)
                # print(f"Question: {question}\nResponse: {cleaned_response}\nPerplexity: {perplexity}\n")
                # print(50*"=")

        # Compile results into a DataFrame
        result_df = pd.DataFrame({
            'query': all_questions,
            'response': all_responses,
            'perplexity': all_perplexities
        })
        return result_df

    def calculate_perplexity(self, output, input_ids):
        """Calculate perplexity for a generated response using the raw output."""
        input_ids_combined = torch.cat((input_ids, output), dim=-1).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids_combined, labels=input_ids_combined)
            loss = outputs.loss.item()

        return math.exp(loss)
