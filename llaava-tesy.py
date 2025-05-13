import outlines
from transformers import LlavaNextForConditionalGeneration

model = outlines.models.transformers_vision(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    model_class=LlavaNextForConditionalGeneration,
    device="cuda",
)