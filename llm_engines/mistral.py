import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from mistralai.exceptions import MistralException
from typing import List

# no image, multi-turn, do not use openai_generate, but can refer to it
def call_worker_mistral(messages:List[str], model_name, conv_system_msg=None, **generate_kwargs) -> str:
    # change messages to mistral format
    client = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY"))
    new_messages = []
    if conv_system_msg:
        new_messages.append(ChatMessage(role="system", content=conv_system_msg))
    for i, message in enumerate(messages):
        new_messages.append(ChatMessage(role="user" if i % 2 == 0 else "assistant", content=message))
        
    assert new_messages[-1].role == "user", "The last message must be from the user"

    response = client.chat(
        model=model_name,
        messages=new_messages,
        temperature=float(generate_kwargs.get("temperature", 0.0)),
        max_tokens=min(int(generate_kwargs.get("max_new_tokens", 1024)), 1024),
        top_p=float(generate_kwargs.get("top_p", 1.0)),
    )
    return response.choices[0].message.content
    
if __name__ == "__main__":
    from icecream import ic
    ic(call_worker_mistral(["Hello", "Hi, I am mistral", "What did I ask in the last response?"], "mistral-large-latest"))