import transformers
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
import pandas as pd
transformers.logging.set_verbosity_debug()

class Model():
    def __init__(self, config, data):
        print("================ Initializing Pythia ================")
        self.config = config
        # model_name = "EleutherAI/pythia-6.9b" 
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # print("Tokenizer loaded")
        # self.model = GPTNeoXForCausalLM.from_pretrained(model_name)
        # print("Model loaded")
        model_name = "EleutherAI/pythia-6.9b-deduped"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision="step143000",
            cache_dir=f"./{model_name}/stepstep143000",
            )
        print("Tokenizer loaded")
        self.model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-6.9b-deduped",
            revision="step143000",
            cache_dir=f"./{model_name}/stepstep143000",
            )
        print("Model loaded")
        print("================ Pythia Initialized ================")
        # Set padding and tokenization settings
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.tokenizer.padding_side = 'left' 
        
        # Load data
        print("================ Loading data  Pythia================")
        self.data = data.load_data()

        # Batch size for efficient processing (set in config or use default)
        self.batch_size = config.get("batch_size", 8)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def batching(self, data_list):
        """Yield successive batches from the data list."""
        for i in range(0, len(data_list), self.batch_size):
            yield data_list[i:i + self.batch_size]

    def query(self):
        print("================ running Pythia query ================")
        if isinstance(self.data, pd.DataFrame):
            queries = self.data['question'].tolist()  # Extract all questions into a list
        else:
            raise ValueError("Data is not in expected format (DataFrame).")

        all_responses = []
        
        # Process queries in batches
        for batch in self.batching(queries):
            print(batch)
            # Tokenize the batch
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            attention_mask = inputs['attention_mask']

            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate responses
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=attention_mask,
                do_sample=True,
                max_new_tokens=50,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Decode responses
            responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            all_responses.extend(responses)

            # Print batch results
            for query, response in zip(batch, responses):
                print(f"Query: {query}")
                print(f"Response: {response}")
                print("=" * 50)

        return queries, all_responses
