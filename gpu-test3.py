import os
import torch
import json
from outlines import models, generate
from pydantic import BaseModel, Field
from transformers import VisionEncoderDecoderModel, AutoProcessor, DonutProcessor
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure CUDA is enabled with appropriate memory settings
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Set torch memory settings to avoid CUDA errors
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# Define the data schema using Pydantic
class DocumentData(BaseModel):
    name: str = Field(description="User's full name")
    last_name: str = Field(description="User's last name") 
    id: int = Field(description="A numeric identifier")

def process_document(image_path):
    """Process a document image and extract structured data"""
    try:
        # Check CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        if device == "cuda":
            # Print GPU info
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
        # Load Donut processor and model
        logger.info("Loading Donut processor...")
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        
        logger.info("Loading Donut model...")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base").to(device)
        
        # Load the image
        logger.info(f"Loading image from {image_path}...")
        image = Image.open(image_path).convert("RGB")
        
        # Process the image with the Donut processor
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        
        # Define task prompt for document processing
        task_prompt = "<s_docvqa><s_question>Extract name, last_name, and id</s_question><s_answer>"
        
        # Set up the decoder input ids
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
        # Generate outputs
        logger.info("Generating document analysis...")
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        
        # Decode the outputs
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(task_prompt, "").replace("</s_answer>", "")
        
        logger.info(f"Raw output: {sequence}")
        
        # Parse output to create structured JSON
        parsed_data = {}
        
        # Look for patterns in the output text
        lines = sequence.strip().split("\n")
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == "name" or key == "full name":
                    parsed_data["name"] = value
                elif key == "last name" or key == "last_name":
                    parsed_data["last_name"] = value
                elif key == "id" or key == "identifier":
                    # Try to convert ID to int
                    try:
                        parsed_data["id"] = int(value)
                    except ValueError:
                        parsed_data["id"] = 0  # Default value if conversion fails
        
        # Ensure all required fields exist
        if "name" not in parsed_data:
            parsed_data["name"] = ""
        if "last_name" not in parsed_data:
            # If we have a full name but no last name, try to extract it
            if "name" in parsed_data and parsed_data["name"]:
                name_parts = parsed_data["name"].split()
                if len(name_parts) > 1:
                    parsed_data["last_name"] = name_parts[-1]
                else:
                    parsed_data["last_name"] = ""
            else:
                parsed_data["last_name"] = ""
        if "id" not in parsed_data:
            parsed_data["id"] = 0
        
        # Validate against our schema
        try:
            result = DocumentData(**parsed_data)
            return result.dict()
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return parsed_data
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def main():
    # Define the image path (Replace with your image path)
    image_path = "page_7.jpg"
    
    # Process the document
    result = process_document(image_path)
    
    # Print the result
    print("\nExtracted Document Data:")
    print(json.dumps(result, indent=2))
    
    return result

if __name__ == "__main__":
    main()