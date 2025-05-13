import os
from outlines import models, generate
from pydantic import BaseModel, ValidationError
from transformers import BlipForConditionalGeneration, AutoProcessor
from PIL import Image
import json

# Ensure CUDA is enabled and appropriate memory settings are applied
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Define the data schema using Pydantic
class DocumentData(BaseModel):
    name: str
    last_name: str
    id: int

# Load Donut model
try:
    model = models.transformers_vision(
        "Salesforce/blip2-opt-2.7b",
        model_class=BlipForConditionalGeneration,
        device="cuda"
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load Processor
try:
    processor = AutoProcessor.from_pretrained("naver-clova-ix/donut-base")
    print("Processor loaded successfully.")
except Exception as e:
    print(f"Error loading processor: {e}")
    exit()

# Initialize the generator
try:
    generator = generate.json(model, DocumentData)
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
    # Define the structured prompt for JSON output
    prompt = """
    Extract the following information as a JSON object:
    {
        "name": "User's full name",
        "last_name": "User's last name",
        "id": "A numeric identifier"
    }
    Ensure the output is a valid JSON object.
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
        result = DocumentData.parse_raw(output)
        print("Parsed Result:", result)
    except ValidationError as ve:
        print(f"Validation Error: {ve}")
    except json.JSONDecodeError as je:
        print(f"JSON Decode Error: {je}")
    except Exception as e:
        print(f"Unexpected Error during JSON parsing: {e}")

except Exception as e:
    print(f"Error during generation: {e}")
