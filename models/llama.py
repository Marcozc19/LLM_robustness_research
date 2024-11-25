from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import math


class CustomDataset(Dataset):
    """Custom Dataset for handling queries."""
    def __init__(self, queries, tokenizer, max_length=128):
        self.queries = queries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        question = self.queries[idx]
        tokens = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "question": question,
        }


class Model:
    def __init__(self, config, data):
        print("================ Initializing LLaMA 3.2 ================")
        self.config = config
        version = self.config["model"]["version"]
        self.model_id = f"meta-llama/Llama-{version}-Instruct"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto"
        )

        # Load data and prepare DataLoader
        self.batch_size = 8
        self.data = data.load_data()  # Load data using the provided `data` object
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data is not in expected format (DataFrame).")
        
        self.dataset = CustomDataset(self.data['question'].tolist(), self.tokenizer)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        print("================ LLaMA 3.2 Initialized ================")

    def calculate_perplexity(self, input_ids, output_ids):
        """Calculate perplexity for a given input-output sequence."""
        input_ids_combined = torch.cat((input_ids, output_ids), dim=-1).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids_combined, labels=input_ids_combined)
            loss = outputs.loss.item()

        perplexity = math.exp(loss)
        return perplexity

    def query(self):
        print("================ Querying LLaMA 3.2 ================")
        all_responses = []
        all_perplexities = []
        all_question = []

        for batch in self.dataloader:
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)
            questions = batch["question"]

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            for question, output in zip(questions, outputs):
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                cleaned_response = response.replace(question, "").strip()
                all_question.append(question)
                all_responses.append(response)

                # Calculate perplexity
                tokenized_question = self.tokenizer(question, return_tensors="pt").to(self.model.device)
                tokenized_response = self.tokenizer(response, return_tensors="pt").to(self.model.device)

                # Calculate perplexity for the specific input-output pair
                perplexity = self.calculate_perplexity(tokenized_question["input_ids"], tokenized_response["input_ids"])
                all_perplexities.append(perplexity)
                # print(f"Query: {input_text}")
                print(f"Response: {cleaned_response}")
                print(f"Perplexity: {perplexity:.4f}")
                print("=" * 50)

        result_df = pd.DataFrame({
            "query": all_question,
            "response": all_responses,
            "perplexity": all_perplexities,
        })
        return result_df


if __name__ == "__main__":
    config = {
        "cache_dir": {"path": "/share/garg/Marco/Models"}
    }

    class MockDataLoader:
        def load_data(self):
            return pd.DataFrame({"question": ["Who are you?", "What is AI?"]})

    data = MockDataLoader()

    model = Model(config, data)
    result_df = model.query()
    print(result_df)
