import outlines
import outlines.models 
from outlines import generate
from pydantic import BaseModel
from transformers import
class User(BaseModel): 
    applicant_name:str

model = outlines.models.transformers_vision("google/gemma-3-4b-pt", device="cuda")

generator = generate.json(model, User)

generator.