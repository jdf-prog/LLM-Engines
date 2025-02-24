import json
import os
import time
from functools import partial
from typing import Union, List
from pathlib import Path
from .cache_sqlite3 import load_cache as load_sqlite3_cache
from .cache_lru import load_cache as load_lru_cache
from .cache_dict import load_cache as load_dict_cache
from .cache_utils import get_cache_file, get_inputs_hash, get_printable_messages, get_batch_cache_dir

load_cache = load_sqlite3_cache

def _generation_cache_wrapper(inputs: Union[str, List[dict]], call_model_worker, model_name, cache_dir=None, overwrite_cache=False, **generate_kwargs):
    cache_dict = load_cache(model_name, cache_dir)
    
    conv_system_msg = generate_kwargs.get("conv_system_msg", "")
    if "n" in generate_kwargs:
        non_hash_keys = ["timeout", "stream"]
        inputs_hash = get_inputs_hash(inputs, conv_system_msg, {k: v for k, v in generate_kwargs.items() if k not in non_hash_keys})
    else:
        inputs_hash = get_inputs_hash(inputs, conv_system_msg)
    
    if not overwrite_cache:
        cached_value = cache_dict[inputs_hash]
        if cached_value:
            if "logprobs" not in generate_kwargs or not generate_kwargs["logprobs"]:
                return cached_value["output"]
            elif "logprobs" in cached_value:
                return cached_value["output"], cached_value["logprobs"]
    
    response = call_model_worker(inputs, **generate_kwargs)
    if isinstance(response, tuple):
        generated_text, logprobs = response
    else:
        generated_text = response
        logprobs = None
    cache_item = {
        "input": get_printable_messages(inputs),
        "output": generated_text,
        "logprobs": logprobs,
        "model_name": model_name,
        'tstamp': time.time(),
        "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        "generate_kwargs": generate_kwargs
    }
    
    # cache_dict[inputs_hash] = cache_item
    
    cache_file = get_cache_file(model_name, cache_dir)
    with open(cache_file, "a+") as f:
        f.write(json.dumps({inputs_hash: cache_item}) + "\n")
    
    return response

def generation_cache_wrapper(call_model_worker, model_name, cache_dir=None, overwrite_cache=False):
    print(f"Using efficient multi-level cache for model {model_name}")
    if cache_dir is None:
        env_cache_dir = os.getenv("LLM_ENGINES_CACHE_DIR")
        if env_cache_dir:
            cache_dir = Path(env_cache_dir)
        else:
            cache_dir = Path(os.path.expanduser(f"~/llm_engines/generation_cache"))
    print(f"Cache directory: {cache_dir}")
    load_cache(model_name, cache_dir) # preload cache
    
    return partial(_generation_cache_wrapper, call_model_worker=call_model_worker, model_name=model_name, cache_dir=cache_dir, overwrite_cache=overwrite_cache)