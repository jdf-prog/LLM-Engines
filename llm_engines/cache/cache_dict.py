import json
import os
from pathlib import Path
from .cache_utils import get_cache_file
from collections import defaultdict
cache_dict = {}
loaded_cache_files = defaultdict(list)
def load_cache(model_name, cache_dir=None):
    global cache_dict
    global loaded_cache_files
    if model_name not in cache_dict:
        if cache_dir is None:
            cache_dir = Path(os.path.expanduser(f"~/llm_engines/generation_cache"))
        else:
            cache_dir = Path(cache_dir)
        cache_file = get_cache_file(model_name, cache_dir)
        if cache_file.exists():
            print("Cache file exists at:", cache_file.absolute())
            if model_name not in loaded_cache_files or cache_file.absolute() not in loaded_cache_files[model_name]:
                print(f"Loading cache from {cache_file}")
                with open(cache_file, "r") as f:
                    model_cache_dict = [json.loads(line) for line in f.readlines()]
                model_cache_dict = {list(item.keys())[0]: list(item.values())[0] for item in model_cache_dict}
                # only keep the output in the value
                cache_dict[model_name] = defaultdict(lambda: None)
                for key, value in model_cache_dict.items():
                    cache_dict[model_name][key] = value["output"]
                loaded_cache_files[model_name].append(cache_file.absolute())
        else:
            cache_dict[model_name] = {}
    return cache_dict[model_name]