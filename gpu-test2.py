from outlines import models, generate
from pydantic import BaseModel

model = models.vllm("google/gemma-3-4b-it",dtype="float16",trust_remote_code=True)
class User(BaseModel):
    name: str
    last_name: str
    id: int

generator = generate.json(model, User)
result = generator(
    "Create a user profile with the fields name, last_name and id"
)
print(result)