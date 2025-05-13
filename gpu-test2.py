import os
from outlines import models, generate
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure CUDA is enabled and appropriate memory settings are applied
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Define the data schema using Pydantic
class User(BaseModel):
    name: str
    last_name: str
    id: int

# Load the model and tokenizer using standard Transformers loading
try:
    model = models.transformers(
        "google/gemma-3-4b-it",
        device="cuda"
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize the generator
try:
    generator = generate.json(model, User)
except Exception as e:
    print(f"Error initializing generator: {e}")
    exit()

# Define the input prompt
prompt = "Create a user profile with the fields name, last_name and id"

# Generate output
try:
    result = generator(prompt)
    print("Generated Result:", result)
except ValidationError as e:
    print(f"Validation Error: {e}")
except Exception as e:
    print(f"Error during generation: {e}")
