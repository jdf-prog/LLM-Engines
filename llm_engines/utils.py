import subprocess
import time
import os
import signal
import os
import signal
import json
import traceback
import threading
import openai
import inspect
import base64
import datetime
import regex as re
import requests
from io import BytesIO
from PIL import Image
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
        new_messages = messages
    else:
        if isinstance(messages, str):
            messages = [messages]
        assert all(isinstance(item, str) for item in messages)
        new_messages = []
        for i, message in enumerate(messages):
            if i % 2 == 0:
                new_messages.append({"role": "user", "content": message})
            else:
                new_messages.append({"role": "assistant", "content": message})
        conv_system_msg = None
    
    # assert the correct format of images
    for message in new_messages:
        assert "role" in message, "role key not found in message"
        assert "content" in message, "content key not found in message"
        if isinstance(message["content"], str):
            pass
        elif isinstance(message["content"], list):
            for sub_content in message["content"]:
                assert sub_content["type"] in sub_content, f"'{sub_content['type']}' key not found in sub_content of type {sub_content['type']}"
                if sub_content["type"] == "text":
                    assert isinstance(sub_content["text"], str), "text key not found in sub_content"
                elif sub_content["type"] == "image_url":
                    assert "url" in sub_content["image_url"] and isinstance(sub_content["image_url"]["url"], str), "url key not found in sub_content['image_url']"
                elif sub_content["type"] == "image":
                    assert isinstance(sub_content["image"], Image.Image), "The image key in of 'image' type must be a PIL Image object"
                    # change image to image_url
                    sub_content["type"] = "image_url"
                    sub_content["image_url"] = {"url": encode_base64_image_url(sub_content["image"])}
                    del sub_content["image"]
                else:
                    raise ValueError(f"Unsupported sub_content type: {sub_content['type']}")
        else:
            raise ValueError(f"Unsupported content type: {type(message['content'])}")
    
    return new_messages, conv_system_msg

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

def max_retry_wrapper(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except MaxRetriesExceededError as e:
        print(str(e))
        return None

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

def encode_base64_image(image:Image.Image, image_format="PNG") -> str:
    im_file = BytesIO()
    image.save(im_file, format=image_format)
    im_bytes = im_file.getvalue()
    im_64 = base64.b64encode(im_bytes).decode("utf-8")
    return im_64

def encode_base64_image_url(image:Image.Image, image_format="PNG") -> str:
    return f"data:image/{image_format};base64,{encode_base64_image(image, image_format)}"

def decode_base64_image_url(base64_uri:str) -> Image.Image:
    # Split the URI to get the base64 data
    try:
        # Remove the "data:image/format;base64," prefix
        header, base64_data = base64_uri.split(',', 1)
        # Get image format from header
        image_format = header.split('/')[1].split(';')[0]
        # Decode base64 string
        image_data = base64.b64decode(base64_data)
        # Create image from binary data
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")
    
def is_base64_image_url(base64_uri:str) -> bool:
    return base64_uri.startswith("data:image/")

def load_image(image_path:str) -> Image.Image:
    # either http or local file path
    if image_path.startswith("http"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image