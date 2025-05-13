from outlines import models
model = models.vllm("google/gemma-3-4b-it",dtype="float16",trust_remote_code=True)