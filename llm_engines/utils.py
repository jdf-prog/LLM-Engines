import subprocess
import time
import os
import signal
import os
import signal
import json
import hashlib
import traceback
import threading
import openai
import inspect
from pathlib import Path
from typing import Union, List
from typing import List
from transformers import AutoTokenizer
from functools import partial
from .cache import load_cache, get_inputs_hash, get_cache_file

default_gen_params = {
    "temperature": 0.0,
    "max_tokens": None,
    "top_p": 1.0,
    "timeout": 600,
}
class SubprocessMonitor:
    def _monitor(self):
        while True:
            if self.proc.poll() is not None:
                print("Subprocess has exited with code", self.proc.returncode)
                os.kill(os.getpid(), signal.SIGTERM)  # Exit the main process
                break
            time.sleep(5)
            
    def __init__(self, command, **kwargs):
        print("Launching subprocess with command:\n", " ".join(command))
        self.proc = subprocess.Popen(command, **kwargs)
        # self.monitor_thread = threading.Thread(target=self._monitor)
        # self.monitor_thread.start()
    
class ChatTokenizer:
    def __init__(self, model_name):
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.system_message = None
        try:
            self.max_length = self.tokenizer.model_max_length
        except AttributeError:
            self.max_length = 4096
        if not isinstance(self.max_length, int):
            self.max_length = 4096
        if self.max_length > 1e6:
            self.max_length = 1e6
            
        if self.tokenizer.chat_template:
            self.apply_chat_template = self.apply_chat_template_default
            print("Using hugging face chat template for model", model_name)
            self.chat_template_source = "huggingface"
        else:
            self.apply_chat_template = None
            self.chat_template_source = None
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
        if not self.apply_chat_template:
            return "Chat template not available for this model"
        else:
            example_messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm good, how about you?"},
            ]
            return self.apply_chat_template(example_messages)
    
    def __call__(self, messages:List[str], **kwargs):
        if not self.apply_chat_template:
            raise NotImplementedError("Chat template not available for this model")
        return self.apply_chat_template(messages, **kwargs)


def convert_messages(messages:Union[List[str], List[dict], str]):
    """
    Convert messages to the format expected by the model
    """
    if all(isinstance(item, dict) for item in messages):
        assert all("content" in item for item in messages), "content key not found in messages"
        assert all("role" in item for item in messages), "role key not found in messages"
        if messages[0]["role"] == "system":
            conv_system_msg = messages[0]["content"]
            messages = messages[1:]
        else:
            conv_system_msg = None
        new_messages = []
        for i, message in enumerate(messages):
            if i % 2 == 0:
                assert message["role"] == "user", "User message must be at even index"
            else:
                assert message["role"] == "assistant", "Assistant message must be at odd index"
            new_messages.append(message["content"])
        return new_messages, conv_system_msg
    else:
        if isinstance(messages, str):
            messages = [messages]
        assert all(isinstance(item, str) for item in messages)
        return messages, None

def _convert_messages_wrapper(messages:Union[List[str], List[dict], str], call_model_worker, is_completion=False, **generate_kwargs):
    if not is_completion:
        messages, conv_system_msg = convert_messages(messages)
        generate_kwargs["conv_system_msg"] = conv_system_msg
    else:
        assert isinstance(messages, str), "Completion model only accepts a single string input"
    # add default generation parameters
    for key, value in default_gen_params.items():
        if key not in generate_kwargs:
            generate_kwargs[key] = value
    return call_model_worker(messages, **generate_kwargs)

def convert_messages_wrapper(call_model_worker, is_completion=False):
    return partial(_convert_messages_wrapper, call_model_worker=call_model_worker, is_completion=is_completion)

def _generation_cache_wrapper(inputs: Union[str, List[str]], call_model_worker, model_name, cache_dir=None, overwrite_cache=False, **generate_kwargs):
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
            if "logprobs" not in generate_kwargs:
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
        "input": inputs,
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
class MaxRetriesExceededError(Exception):
    pass

short_error_instances = [
    openai.BadRequestError,
]

def _retry_on_failure(*args, call_model_worker=None, num_retries=5, **kwargs):
    if not call_model_worker:
        raise ValueError("call_model_worker is required")
    try:
        return call_model_worker(*args, **kwargs)
    except Exception as e:
        if not num_retries:
            if any(isinstance(e, error_instance) for error_instance in short_error_instances):
                print(str(e))
            else:
                print(traceback.format_exc())
            raise MaxRetriesExceededError(f"Max retries exceeded for call_model_worker (num_retries={num_retries})")
        for i in range(num_retries):
            try:
                return call_model_worker(*args, **kwargs)
            except Exception as e:
                print("Error in call_model_worker, retrying... (Error: {})".format(e))
                time.sleep(1)
                if i >= num_retries - 1 and not isinstance(e, TimeoutError):
                    if any(isinstance(e, error_instance) for error_instance in short_error_instances):
                        print(str(e))
                    else:
                        # format dump of the last error and
                        print(traceback.format_exc())
        raise MaxRetriesExceededError(f"Max retries exceeded for call_model_worker (num_retries={num_retries})")
    
def retry_on_failure(call_model_worker, num_retries=5):
    return partial(_retry_on_failure, call_model_worker=call_model_worker, num_retries=num_retries)
    return wrapper

def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

def with_timeout(timeout):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [TimeoutError(f"Function call timed out (timeout={timeout})")]
            stop_event = threading.Event()

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout)
            if thread.is_alive():
                stop_event.set()
                raise TimeoutError(f"Function call timed out (timeout={timeout})")
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator

def get_function_arg_names(func):
    signature = inspect.signature(func)
    parameters = signature.parameters
    
    arg_names = []
    kwarg_names = []
    
    for name, param in parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            arg_names.append(f"*{name}")
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            kwarg_names.append(f"**{name}")
        else:
            arg_names.append(name)
    
    return arg_names, kwarg_names