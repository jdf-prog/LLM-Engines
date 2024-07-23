import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from mistralai.exceptions import MistralException
from typing import List
from .utils import with_timeout

# no image, multi-turn, do not use openai_generate, but can refer to it
def call_worker_mistral(messages:List[str], model_name, timeout:int=120, conv_system_msg=None, **generate_kwargs) -> str:
    # change messages to mistral format
    client = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY"), timeout=timeout)
    new_messages = []
    if conv_system_msg:
        new_messages.append(ChatMessage(role="system", content=conv_system_msg))
    for i, message in enumerate(messages):
        new_messages.append(ChatMessage(role="user" if i % 2 == 0 else "assistant", content=message))

    @with_timeout(timeout)
    def generate_content():
        return client.chat(
            model=model_name,
            messages=new_messages,
            **generate_kwargs,
        )
    response = generate_content()
    return response.choices[0].message.content
    
if __name__ == "__main__":
    from icecream import ic
    ic(call_worker_mistral(["Hello", "Hi, I am mistral", "What did I ask in the last response?"], "mistral-large-latest"))