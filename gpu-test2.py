import os
from outlines import models, generate
from pydantic import BaseModel, ValidationError
from transformers import VisionEncoderDecoderModel, AutoProcessor
from PIL import Image

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
    model = models.transformers(
        "naver-clova-ix/donut-base",
        model_class=VisionEncoderDecoderModel,
        device="cuda",
        dtype="float16"
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
    # Generate output using the image
    result = generator(
        "Extract name, last_name, and id from the document",
        media=[image]
    )
    print("Generated Result:", result)
except ValidationError as e:
    print(f"Validation Error: {e}")
except Exception as e:
    print(f"Error during generation: {e}")
