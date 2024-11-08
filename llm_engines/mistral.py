import os
from mistralai import Mistral
from typing import List
from .utils import with_timeout

# no image, multi-turn, do not use openai_generate, but can refer to it
def call_worker_mistral(messages:List[str], model_name, timeout:int=120, conv_system_msg=None, **generate_kwargs) -> str:
    # change messages to mistral format
    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    new_messages = []
    if conv_system_msg:
        new_messages.append({"role": "system", "content": conv_system_msg})
    for i, message in enumerate(messages):
        new_messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": message})

    if "n" in generate_kwargs:
        generate_kwargs.pop("n") # mistral does not have n
    if "logprobs" in generate_kwargs:
        raise ValueError("logprobs is not supported in mistral")
    stream = generate_kwargs.pop("stream", False)
    @with_timeout(timeout)
    def generate_content():
        completion = client.chat.complete(
            model=model_name,
            messages=new_messages,
            **generate_kwargs,
        )
        if len(completion.choices) > 1:
            return [c.message.content for c in completion.choices]
        else:
            return completion.choices[0].message.content
    
    @with_timeout(timeout)
    def stream_content():
        completion = client.chat.stream(
            model=model_name,
            messages=new_messages,
            **generate_kwargs,
        )
        def generate_stream():
            for chunk in completion:
                if chunk.data.choices[0].delta.content is not None:
                    yield chunk.data.choices[0].delta.content
        return generate_stream()
    if not stream:
        return generate_content()
    else:
        return stream_content()
    
if __name__ == "__main__":
    from icecream import ic
    ic(call_worker_mistral(["Hello", "Hi, I am mistral", "What did I ask in the last response?"], "mistral-large-latest"))