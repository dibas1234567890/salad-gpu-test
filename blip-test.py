import outlines
from transformers import Blip2ForConditionalGeneration

model = outlines.models.transformers_vision(
    "Salesforce/blip2-opt-2.7b",
    model_class=Blip2ForConditionalGeneration,
    device="cuda",
)