import os
from outlines import models, generate
from pydantic import BaseModel, ValidationError
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from PIL import Image
import json
import torch

# Ensure CUDA is enabled and appropriate memory settings are applied
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the data schema using Pydantic
class ImageDescription(BaseModel):
    description: str

# Load BLIP model
try:
    model = models.transformers(
        "Salesforce/blip2-opt-2.7b",
        model_class=Blip2ForConditionalGeneration,
        device=device,
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load BLIP Processor
try:
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    print("Processor loaded successfully.")
except Exception as e:
    print(f"Error loading processor: {e}")
    exit()

# Initialize the generator
try:
    generator = generate.json(model, ImageDescription)
    print("Generator initialized successfully.")
except Exception as e:
    print(f"Error initializing generator: {e}")
    exit()

# Define the image path (Replace with your image path)
image_path = "page_7.jpg"

# Load the image
try:
    image = Image.open(image_path).convert("RGB")
    print("Image loaded successfully.")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Generate output
try:
    # Define the prompt for conditional generation
    prompt = """
    Describe the content of the image in a single sentence. 
    Return the response as a JSON object with the key "description".
    """

    # Generate output using the image
    output = generator(
        prompt,
        media=[image]
    )

    # Inspect raw output
    print("Raw Output:", output)

    # Attempt to parse the output as JSON
    try:
        # Validate against the schema
        result = ImageDescription.parse_raw(output)
        print("Parsed Result:", result)
    except ValidationError as ve:
        print(f"Validation Error: {ve}")
    except json.JSONDecodeError as je:
        print(f"JSON Decode Error: {je}")
    except Exception as e:
        print(f"Unexpected Error during JSON parsing: {e}")

except Exception as e:
    print(f"Error during generation: {e}")
