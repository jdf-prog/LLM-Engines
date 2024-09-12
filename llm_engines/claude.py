import anthropic
import os
from typing import List
from anthropic import NOT_GIVEN
from .utils import with_timeout

# no image, multi-turn, do not use openai_generate, but can refer to it
def call_worker_claude(messages:List[str], model_name, timeout:int=60, conv_system_msg=None, **generate_kwargs) -> str:
    # change messages to mistral format
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    new_messages = []
    for i, message in enumerate(messages):
        new_messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": message})
             
    generate_kwargs.pop("n", None) # claude does not have n
    if not generate_kwargs.get("max_tokens", None):
        generate_kwargs["max_tokens"] = 1024
    stream = generate_kwargs.pop("stream", False)
    @with_timeout(timeout)
    def get_response():
        completion = client.messages.create(
            model=model_name,
            messages=new_messages,
            system=conv_system_msg if conv_system_msg else NOT_GIVEN,
            timeout=timeout,
            **generate_kwargs,
        )
        if len(completion.content) > 1:
            return [c.text for c in completion.content]
        else:
            return completion.content[0].text
        
    @with_timeout(timeout)
    def stream_response():
        with client.messages.stream(
            model=model_name,
            messages=new_messages,
            system=conv_system_msg if conv_system_msg else NOT_GIVEN,
            timeout=timeout,
            **generate_kwargs,
        ) as stream:
            for text in stream.text_stream:
                yield text
                
    if not stream:
        return get_response()
    else:
        return stream_response()
    
if __name__ == "__main__":
    from icecream import ic
    ic(call_worker_claude(["Hello", "Hi, I am claude", "What did I ask in the last response?"], "claude-3-opus-20240229"))