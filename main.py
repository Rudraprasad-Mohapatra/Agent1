import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load variables from .env
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    model="moonshotai/Kimi-K2.5",
    token=HF_TOKEN
)

output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of France is"},
    ],
    stream=False,
    max_tokens=1024,
    extra_body={
        'thinking': {'type': 'disabled'},
    }
)

print(output.choices[0].message.content)