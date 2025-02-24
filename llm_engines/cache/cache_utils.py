import os
import hashlib
import base64
from copy import deepcopy
from typing import List, Union
from cachetools import LRUCache
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=None)
def get_cache_file(model_name_or_path, cache_dir):
    model_name = model_name_or_path.split("/")[-2:]
    model_name = "/".join(model_name)
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{model_name}.jsonl"
    else:
        cache_file = Path(os.path.expanduser(f"~/llm_engines/generation_cache/{model_name}.jsonl"))
    if not cache_file.parent.exists():
        cache_file.parent.mkdir(parents=True)
    return cache_file

@lru_cache(maxsize=None)
def get_batch_cache_dir(model_name_or_path, cache_dir):
    model_name = model_name_or_path.split("/")[-2:]
    model_name = "/".join(model_name)
    if cache_dir is not None:
        batch_cache_dir = Path(cache_dir) / f"{model_name}_batch_cache"
    else:
        batch_cache_dir = Path(os.path.expanduser(f"~/llm_engines/generation_cache/{model_name}_batch_cache"))
    if not batch_cache_dir.exists():
        batch_cache_dir.mkdir(parents=True)
    return batch_cache_dir

MAX_PRINTABLE_IMAGE_URL_LENGTH = 100
def get_printable_messages(messages):
    # mainly for image_url, only keep the first 100 characters
    messages = deepcopy(messages)
    for message in messages:
        if isinstance(message["content"], list):
            for sub_message in message["content"]:
                if sub_message["type"] == "image_url":
                    sub_message["image_url"]["url"] = sub_message["image_url"]["url"][:MAX_PRINTABLE_IMAGE_URL_LENGTH] + \
                        (f"... ({len(sub_message['image_url']['url']) - MAX_PRINTABLE_IMAGE_URL_LENGTH} more characters)" if len(sub_message['image_url']['url']) > MAX_PRINTABLE_IMAGE_URL_LENGTH else "")
                elif sub_message["type"] == "image":
                    sub_message["image"] = sub_message["image"][:100]
    return messages

def get_inputs_hash(inputs:Union[str, List[dict]], conv_system_msg, generate_kwargs=None):
    
    inputs = inputs.copy()
    if isinstance(inputs, str):
        try:
            return hashlib.md5(inputs.encode()).hexdigest()
        except UnicodeEncodeError as e:
            return hashlib.md5(inputs.encode('utf-16', 'surrogatepass').decode('utf-16').encode('utf-8')).hexdigest()
    
    # inputs is a list of dicts in openai format
    
    if conv_system_msg:
        to_hash_messages = [{
            "role": "system",
            "content": conv_system_msg
        }] + inputs
    else:
        to_hash_messages = inputs
    
    to_hash_inputs = []
    for message in to_hash_messages:
        role = message["role"]
        content = message["content"]
        if isinstance(content, str):
            to_hash_inputs.append(f"{role}:{content}")
        elif isinstance(content, list):
            strs = []
            for sub_content in content:
                if sub_content["type"] == "text":
                    strs.append(sub_content["text"])
                elif sub_content["type"] == "image_url":
                    if "url" not in sub_content["image_url"]:
                        raise ValueError("image_url must have a url key")
                    image_url_hash = hashlib.md5(sub_content["image_url"]["url"].encode()).hexdigest()
                    strs.append(f"image_url:{image_url_hash}")
                else:
                    raise ValueError(f"Unknown content type {sub_content['type']}")
            to_hash_inputs.append(f"{role}:{''.join(strs)}")
        else:
            raise ValueError(f"Unknown content type {type(content)}")
    
    if generate_kwargs:
        to_hash_inputs.append(str(generate_kwargs))
    
    try:
        return hashlib.md5("".join(to_hash_inputs).encode()).hexdigest()
    except UnicodeEncodeError as e:
        return hashlib.md5("".join(to_hash_inputs).encode('utf-16', 'surrogatepass').decode('utf-16').encode('utf-8')).hexdigest()
    
    
    
