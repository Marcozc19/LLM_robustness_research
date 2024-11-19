from transformers import pipeline

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
chatbot = pipeline("text-generation", model="mistralai/Mistral-Nemo-Instruct-2407",max_new_tokens=128)
chatbot(messages)

# from transformers import LlamaTokenizerFast, MistralForCausalLM
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"
# tokenizer = LlamaTokenizerFast.from_pretrained('mistralai/Mistral-Small-Instruct-2409')
# tokenizer.pad_token = tokenizer.eos_token

# model = MistralForCausalLM.from_pretrained('mistralai/Mistral-Small-Instruct-2409', torch_dtype=torch.bfloat16)
# model = model.to(device)

# prompt = "How often does the letter r occur in Mistral?"

# messages = [
#     {"role": "user", "content": prompt},
#  ]

# model_input = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
# gen = model.generate(model_input, max_new_tokens=150)
# dec = tokenizer.batch_decode(gen)
# print(dec)

# from transformers import LlamaTokenizerFast, MistralForCausalLM
# import torch
# import pandas as pd

# class Model:
#     def __init__(self, config, data):
#         print("================ Initializing Mistral ================")
#         # Initialize model configuration and data
#         self.config = config
#         checkpoint = "mistralai/Mistral-Small-Instruct-2409"  # Mistral model checkpoint

#         # Load tokenizer and model
#         self.tokenizer = LlamaTokenizerFast.from_pretrained(checkpoint)
#         self.tokenizer.pad_token = self.tokenizer.eos_token  # Set padding token
#         self.tokenizer.padding_side = 'left'
#         self.model = MistralForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)

#         # Use GPU if available
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = self.model.to(self.device)
#         print("================ Mistral Initialized  ================")
#         # Load data
#         self.data = data.load_data()
#         self.batch_size = config.get("batch_size", 8)  # Set batch size from config or use default

#     def batching(self, data_list):
#         """Yield successive batches from the data list."""
#         for i in range(0, len(data_list), self.batch_size):
#             yield data_list[i:i + self.batch_size]

#     def query(self):
#         print("================ Querying Mistral ================")
#         if isinstance(self.data, pd.DataFrame):
#             queries = self.data['question'].tolist()  # Extract all questions into a list
#         else:
#             raise ValueError("Data is not in expected format (DataFrame).")

#         all_responses = []

#         # Process queries in batches
#         for batch in self.batching(queries):
#             batch_inputs = []

#             # Apply chat template to each question in the batch
#             for question in batch:
#                 # Format question as a message
#                 messages = [{"role": "user", "content": question}]
#                 # Apply chat template
#                 input_text = self.tokenizer.apply_chat_template(
#                     messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
#                 )
#                 # Add to batch inputs
#                 batch_inputs.append(input_text)

#             # Combine batch inputs
#             inputs = torch.cat(batch_inputs, dim=0).to(self.device)

#             # Generate responses
#             outputs = self.model.generate(
#                 inputs["input_ids"],
#                 attention_mask=inputs['attention_mask'],
#                 max_new_tokens=50,
#                 temperature=0.2,
#                 top_p=0.9,
#                 do_sample=True
#             )

#             # Decode responses
#             for output in outputs:
#                 decoded_response = self.tokenizer.decode(output, skip_special_tokens=True)
#                 all_responses.append(decoded_response.strip())

#         result_df = pd.DataFrame({
#             'query': queries,
#             'response': all_responses
#         })
#         return result_df


# if __name__ == "__main__":
#     # Example usage
#     class DummyDataLoader:
#         def load_data(self):
#             # Replace this with your actual data loading logic
#             return pd.DataFrame({"question": ["What is the capital of France?", "How often does the letter r occur in Mistral?"]})

#     config = {"batch_size": 4}  # Example configuration dictionary
#     data_loader = DummyDataLoader()

#     mistral_model = Model(config, data_loader)
#     responses = mistral_model.query()

#     # Print results
#     print(responses)
