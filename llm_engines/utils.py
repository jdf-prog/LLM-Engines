import subprocess
import threading
import time
import os
import signal
import yaml
import os
import json
import hashlib
from pathlib import Path
from typing import Union, List
from typing import List
from transformers import AutoTokenizer
class SubprocessMonitor:
    def _monitor(self):
        while True:
            if self.proc.poll() is not None:
                print("Subprocess has exited with code", self.proc.returncode)
                os.kill(os.getpid(), signal.SIGTERM)  # Exit the main process
                break
            time.sleep(5)
            
    def __init__(self, command, **kwargs):
        self.proc = subprocess.Popen(command, **kwargs)
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()
    
    def __getattr__(self, name):
        return getattr(self.proc, name)
    
class ChatTokenizer:
    def __init__(self, model_name):
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.system_message = None
        if self.tokenizer.chat_template:
            self.apply_chat_template = self.apply_chat_template_default
            print("Using hugging face chat template for model", model_name)
            self.chat_template_source = "huggingface"
        else:
            raise NotImplementedError("Chat template not implemented for model", model_name)
        print("Example prompt: \n", self.example_prompt())
        
    def apply_chat_template_default(
        self, 
        messages:List[str],
        add_generation_prompt:bool=True,
        chat_template:str=None
    ):
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            chat_template=chat_template,
        )
        return prompt
    
    def example_prompt(self):
        example_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good, how about you?"},
        ]
        return self.apply_chat_template(example_messages)
    
    def __call__(self, messages:List[str], **kwargs):
        return self.apply_chat_template(messages, **kwargs)


cache_dict = None
def generation_cache_wrapper(call_model_worker, model_name, cache_dir=None):
    print(f"Using cache for model {model_name}")
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{model_name}.jsonl"
    else:
        cache_file = Path(os.path.abspath(__file__)).parent / "generation_cache" / f"{model_name}.jsonl"
    if cache_file.exists():
        print(f"Cache file exists at {cache_file}")
    print(f"Each single input will be cached in hash-input:output format in {cache_file}")
    def wrapper(inputs:Union[str, List[str]], **generate_kwargs):
        global cache_dict
        if cache_dir is not None:
            cache_file = Path(cache_dir) / f"{model_name}.jsonl"
        else:
            cache_file = Path(os.path.abspath(__file__)).parent.parent / "generation_cache" / f"{model_name}.jsonl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_dict is None:
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    cache_dict = [json.loads(line) for line in f.readlines()]
                cache_dict = {list(item.keys())[0]: list(item.values())[0] for item in cache_dict}
            else:
                cache_dict = {}
        if isinstance(inputs, str):
            inputs_hash = hashlib.md5(inputs.encode()).hexdigest()
        else:
            inputs_hash = hashlib.md5("".join(inputs).encode()).hexdigest()
        if inputs_hash in cache_dict:
            return cache_dict[inputs_hash]["output"]
        else:
            generated_text = call_model_worker(inputs, **generate_kwargs)
            cache_dict[inputs_hash] = {"input": inputs, "output": generated_text}
            with open(cache_file, "a+") as f:
                f.write(json.dumps({inputs_hash: cache_dict[inputs_hash]}, ensure_ascii=False) + "\n")
            return generated_text
    return wrapper

def retry_on_failure(call_model_worker, num_retries=5):
    def wrapper(*args, **kwargs):
        for i in range(num_retries):
            try:
                return call_model_worker(*args, **kwargs)
            except Exception as e:
                print("Error in call_model_worker, retrying", e)
                time.sleep(1)
        raise Exception("Failed after multiple retries")
    return wrapper