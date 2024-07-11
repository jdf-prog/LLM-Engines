import anthropic
import os
from typing import List
from anthropic import NOT_GIVEN

# no image, multi-turn, do not use openai_generate, but can refer to it
def call_worker_claude(messages:List[str], model_name, conv_system_msg=None, **generate_kwargs) -> str:
    # change messages to mistral format
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    new_messages = []
    for i, message in enumerate(messages):
        new_messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": message})
    
    response = client.messages.create(
        model=model_name,
        messages=new_messages,
        system=conv_system_msg if conv_system_msg else NOT_GIVEN,
        **generate_kwargs,
    )
    return response.content[0].text
    
if __name__ == "__main__":
    from icecream import ic
    ic(call_worker_claude(["Hello", "Hi, I am claude", "What did I ask in the last response?"], "claude-3-opus-20240229"))