import os
from typing import List, Union
from fireworks.client import Fireworks

# no image, multi-turn
def call_worker_fireworks(messages:List[str], model_name, timeout:int=60, conv_system_msg=None, **generate_kwargs) -> str:
    # change messages to openai format
    if conv_system_msg:
        new_messages = [{"role": "system", "content": conv_system_msg}] + messages
    else:
        new_messages = messages
    # initialize openai client
    client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])
    # call fireworks
    completion = client.chat.completions.create(
        model=model_name,
        messages=new_messages,
        **generate_kwargs,
    )
    stream = generate_kwargs.get("stream", False)
    
    if "logprobs" in generate_kwargs:
        return_logprobs = True
    
    if not stream:
        if "logprobs" not in generate_kwargs or not generate_kwargs["logprobs"]:
            if len(completion.choices) > 1:
                return [c.message.content for c in completion.choices]
            else:
                return completion.choices[0].message.content
        else:
            if len(completion.choices) > 1:
                return [c.message.content for c in completion.choices], [c.logprobs.dict() for c in completion.choices]
            else:
                return completion.choices[0].message.content, completion.choices[0].logprobs.dict()
    else:
        def generate_stream():
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        return generate_stream()

def call_worker_fireworks_completion(prompt:str, model_name, timeout:int=60, **generate_kwargs) -> str:
    # initialize openai client
    client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])
    # call fireworks
    print(generate_kwargs)
    if "max_tokens" not in generate_kwargs:
        generate_kwargs["max_tokens"] = 256 # have to set max_tokens to be explicit
    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        **generate_kwargs,
    )
    stream = generate_kwargs.get("stream", False)
    if not stream:
        if len(completion.choices) > 1:
            return [c.text for c in completion.choices]
        else:
            return completion.choices[0].text
    else:
        def generate_stream():
            for chunk in completion:
                if chunk.choices[0].text is not None:
                    yield chunk.choices[0].text
        return generate_stream()

if __name__ == "__main__":
    from icecream import ic
    generate_kwargs = {
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "max_tokens": 20480
    }
    ic(call_worker_fireworks(["Hello"], "accounts/fireworks/models/deepseek-r1", **generate_kwargs))
    ic(call_worker_fireworks_completion("Hello", "accounts/fireworks/models/deepseek-r1", **generate_kwargs))