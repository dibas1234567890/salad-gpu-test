import os
import torch
from transformers import VisionEncoderDecoderModel
from outlines import models, generate
from PIL import Image, UnidentifiedImageError
from schema import DigitalServicesForm

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name())

try:
    model = models.transformers_vision(
        "naver-clova-ix/donut-base",
        model_class=VisionEncoderDecoderModel, device = "cuda"
    )
except Exception as e:
    print(f"Model loading error: {e}")
    exit()

image_path = "/content/page_7.jpg" 

try:
    image = Image.open(image_path)
    image.verify()  
    image = Image.open(image_path) 
    max_size = (1024, 1024)
    image = image.resize(max_size)
    print(f"Image {image_path} loaded successfully.")
except UnidentifiedImageError:
    print("Error: Unable to open the image. The file may be corrupted or not an image.")
    exit()
except FileNotFoundError:
    print(f"Error: File not found - {image_path}")
    exit()

# Initialize the generator
try:
    generator = generate.json(model, DigitalServicesForm)
except Exception as e:
    print(f"Generator initialization error: {e}")
    exit()

# Run the generator with input text and media
try:
    result = generator(
        "Extracted all data provided in the photo as per the schema",
        media=[image]
    )
    print("Extraction result:", result)
except RuntimeError as e:
    print(f"RuntimeError during generation: {e}")
except Exception as e:
    print(f"Unexpected error during generation: {e}")
