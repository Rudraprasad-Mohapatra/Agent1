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

# 3. System prompt (Agent brain)
SYSTEM_PROMPT = """Answer the following questions as best you can.

You have access to the following tools:

get_weather: Get the current weather in a given location

To use a tool, return a JSON like:
{
  "action": "get_weather",
  "action_input": {"location": "London"}
}

Use this format:

Question: ...
Thought: ...
Action: ...
Observation: ...
Thought: ...
Final Answer: ...
"""

# 4. Step 1 → Ask question
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"}
]

# 5. Step 2 → Let AI think (STOP before fake observation)
response = client.chat.completions.create(
    messages=messages,
    max_tokens=150,
    stop=["Observation:"]
)

assistant_reply = response.choices[0].message.content
print("---- AI Thought & Action ----")
print(assistant_reply)

# 6. Step 3 → Our real function (tool)
def get_weather(location):
    return f"The weather in {location} is sunny with low temperatures."

# 7. Step 4 → Run tool manually
tool_result = get_weather("London")

print("\n---- Tool Output ----")
print(tool_result)

# 8. Step 5 → Send result back to AI
messages.append({
    "role": "assistant",
    "content": assistant_reply + "\nObservation:\n" + tool_result
})

# 9. Step 6 → Final answer from AI
final_response = client.chat.completions.create(
    messages=messages,
    max_tokens=150
)

print("\n---- Final Answer ----")
print(final_response.choices[0].message.content)