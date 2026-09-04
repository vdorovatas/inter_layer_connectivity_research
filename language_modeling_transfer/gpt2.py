from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Choose your model
model_name = "gpt2"

# Load and save locally
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Save to a local folder
tokenizer.save_pretrained("local_gpt2")
model.save_pretrained("local_gpt2")

