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
from fastchat.model.model_adapter import get_conversation_template
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
        self.conv = get_conversation_template(model_name)
        self.system_message = None
        if self.tokenizer.chat_template:
            self.apply_chat_template = self.apply_chat_template_default
            print("Using hugging face chat template for model", model_name)
            self.chat_template_source = "huggingface"
        else:
            if self.conv.name == "one_shot":
                raise ValueError(f"No chat template found for model {model_name}")
            print("Using fastchat chat template for model", model_name)
            self.apply_chat_template = self.apply_chat_template_fschat
            self.chat_template_source = "fastchat"
            self.system_message = self.conv.system_message
        print("Example prompt: \n", self.example_prompt())
        
    def apply_chat_template_default(self, messages:List[str]):
        assert len(messages) % 2 == 1, "The number of messages must be odd, meaning the last message is from the user"
        new_messages = []
        if self.system_message:
            new_messages.append({"role": "system", "content": self.system_message})
        for i, message in enumerate(messages):
            new_messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": message})
        assert new_messages[-1]["role"] == "user", "The last message must be from the user"
        prompt = self.tokenizer.apply_chat_template(new_messages, add_generation_prompt=True, tokenize=False)
        return prompt
    
    def apply_chat_template_fschat(self, messages:List[str]):
        assert len(messages) % 2 == 1, "The number of messages must be odd, meaning the last message is from the user"
        conv = self.conv
        for i, message in enumerate(messages):
            conv.append_message(conv.roles[i % 2], message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt
    
    def example_prompt(self):
        example_messages = ["Hello", "Hi", "How are you?", "I'm fine, thank you", "Goodbye"]
        return self.apply_chat_template(example_messages)
    
    def __call__(self, messages:List[str]):
        return self.apply_chat_template(messages)


cache_dict = None
def generation_cache_wrapper(call_model_worker, model_name, cache_dir=None):
    print(f"Using cache for model {model_name}")
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{model_name}.jsonl"
    else:
        cache_file = Path(os.path.abspath(__file__)).parent.parent / "generation_cache" / f"{model_name}.jsonl"
    if cache_file.exists():
        print(f"Cache file exists at {cache_file}")
    print(f"Each single input will be cached in hash-input:output format in {cache_file}")
    def wrapper(inputs:Union[str, List[str]]):
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
            generated_text = call_model_worker(inputs)
            cache_dict[inputs_hash] = {"input": inputs, "output": generated_text}
            with open(cache_file, "a+") as f:
                f.write(json.dumps({inputs_hash: cache_dict[inputs_hash]}, ensure_ascii=False) + "\n")
            return generated_text
    return wrapper