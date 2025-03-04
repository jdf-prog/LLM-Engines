import os
import json
import hashlib
import time
import filelock
import random
import openai
from datetime import datetime
from openai import OpenAI
from typing import List, Union
from pathlib import Path
from tqdm import tqdm
from .cache import get_printable_messages

# no image, multi-turn, do not use grok_generate, but can refer to it
def call_worker_grok(messages:List[str], model_name, timeout:int=60, conv_system_msg=None, **generate_kwargs) -> str:
    # change messages to openai format
    if conv_system_msg:
        new_messages = [{"role": "system", "content": conv_system_msg}] + messages
    else:
        new_messages = messages
    # initialize openai client
    client = OpenAI(api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1")
    # call grok
    completion = client.chat.completions.create(
        model=model_name,
        messages=new_messages,
        timeout=timeout,
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

def call_worker_grok_completion(prompt:str, model_name, timeout:int=60, **generate_kwargs) -> str:
    # initialize openai client
    client = OpenAI(api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1")
    # call grok
    if "max_tokens" not in generate_kwargs:
        generate_kwargs["max_tokens"] = 256 # have to set max_tokens to be explicit
    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        timeout=timeout,
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
    ic(call_worker_grok(["Hello"], "grok-2-latest"))
    ic(call_worker_grok_completion("Hello", "grok-2-latest"))