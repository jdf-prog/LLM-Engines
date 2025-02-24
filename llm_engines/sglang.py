import os
import time
import torch
import random
import openai
import importlib.util
from pathlib import Path
from typing import List
from sglang import function, system, user, assistant, gen
from .utils import SubprocessMonitor, ChatTokenizer, with_timeout, get_function_arg_names
worker_initiated = False
sglang_workers = {}
def launch_sglang_worker(
    model_name: str,
    num_gpus: int=None,
    gpu_ids: List[int]=None,
    dtype: str="auto",
    quantization: str=None,
    port: int=34200,
    host: str="127.0.0.1",
    additional_args: List[str]=[]
) -> str:
    """
    Launch a model worker and return the address
    Args:
        model_name: the model name to launch
    Returns:
        the address of the launched model
    """
    # python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000
    ### For debug
    # port, additonal_ports = allocate_init_ports(port)
    # print(f"Launching SGLang model {model_name} on port {port}")
    # print(f"Additional ports: {additonal_ports}")
    ### For debug
    worker_addr = f"http://{host}:{port}"
    log_file = Path(os.path.abspath(__file__)).parent / "logs" / f"{model_name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if gpu_ids:
        num_gpus = len(gpu_ids)
    else:
        if not num_gpus:
            num_gpus = torch.cuda.device_count()
            print(f"Warning: num_gpus or gpu_ids not provided, using {num_gpus} GPUs")
        gpu_ids = list(range(num_gpus))
    env = os.environ.copy()
    # Set the CUDA_VISIBLE_DEVICES environment variable
    env["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu_id) for gpu_id in gpu_ids])
    
    # check flashinfer
    flashinfer = importlib.util.find_spec("flashinfer")
    if flashinfer is None:
        print("flashinfer not found, please first install flashinfer for sglang")
        print("Simple Command: pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/")
        print("Please refer to https://docs.flashinfer.ai/installation.html for detailed installation instructions")
        exit(1) 
    else:
        print("flashinfer found, enable flashinfer for sglang")
        flashinfer_args = []
    if quantization:
        available_quantizations = "awq,fp8,gptq,marlin,gptq_marlin,awq_marlin,squeezellm,bitsandbytes"
        available_quantizations = available_quantizations.split(",")
        if quantization not in available_quantizations:
            raise ValueError(f"Quantization {quantization} not supported, available quantizations: {available_quantizations}")
        flashinfer_args = ["--quantization", quantization]
    # additonal_ports = [port+i for i in range(1, 9)]
    proc = SubprocessMonitor([
        "python3", "-m", "sglang.launch_server",
        "--model-path", model_name,
        "--host", host,
        "--port", str(port),
        "--dtype", dtype,
        # "--api-key", "sglang",
        "--log-level", "warning",
        "--tp-size",  str(num_gpus) if num_gpus is not None else "1",
        # "--additional-ports"] + [str(port) for port in additonal_ports
    ] + flashinfer_args + additional_args ,env=env)
    print(f"Launching SGLang model {model_name} with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    sglang_workers[worker_addr] = proc
    return worker_addr, proc

@function
def multi_turn_question(s, messages, system_message=None):
    if system_message:
        s += system(system_message)
    for i, message in enumerate(messages):
        if i % 2 == 0:
            s += user(message)
        else:
            s += assistant(message)
    s += assistant(gen("answer"))
    
@function
def question(s, prompt):
    s += prompt
    s += gen("answer")

chat_tokenizers = {}
def call_sglang_worker(messages, model_name, worker_addrs, timeout:int=300, conv_system_msg=None, **generate_kwargs) -> str:
    global worker_initiated
    global chat_tokenizers
    
    if model_name not in chat_tokenizers:
        chat_tokenizers[model_name] = ChatTokenizer(model_name)
    chat_tokenizer = chat_tokenizers[model_name]
    
    # change messages to openai format
    if conv_system_msg:
        chat_messages = [{"role": "system", "content": conv_system_msg}] + messages
    else:
        chat_messages = messages
        
    # prompt = chat_tokenizer(chat_messages)

    worker_addr = random.choice(worker_addrs)
    
    client = openai.OpenAI(
        base_url=f"{worker_addr}/v1",
        api_key="sglang-engine-token",
    )
    
    generate_kwargs['max_tokens'] = generate_kwargs.get('max_tokens', chat_tokenizer.max_length) # for sglang, max_tokens is required and must > 0
    args_names, kwargs_names = get_function_arg_names(client.chat.completions.create)
    extra_body_params = {}
    for key in list(generate_kwargs.keys()):
        if key not in args_names + kwargs_names:
            extra_body_params[key] = generate_kwargs[key]
            del generate_kwargs[key]
    generate_kwargs["extra_body"] = extra_body_params
    
    stream = generate_kwargs.get("stream", False)
    if stream:
        generate_kwargs.pop("n", None)
    @with_timeout(timeout)
    def get_response():
        while True:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=chat_messages,
                    **generate_kwargs,
                )
                break
            except openai.APIConnectionError as e:
                if not worker_initiated:
                    time.sleep(5)
                    continue
                print(f"API connection error: {e}")
                time.sleep(5)
                continue
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
    return get_response()

def call_sglang_worker_completion(prompt:str, model_name, worker_addrs, timeout:int=300, **generate_kwargs) -> str:
    global worker_initiated
    global chat_tokenizers
    if model_name not in chat_tokenizers:
        chat_tokenizers[model_name] = ChatTokenizer(model_name)
    chat_tokenizer = chat_tokenizers[model_name]
    
    if "max_new_tokens" in generate_kwargs:
        if "max_tokens" not in generate_kwargs:
            generate_kwargs["max_tokens"] = generate_kwargs["max_new_tokens"]
        del generate_kwargs["max_new_tokens"]
        
    worker_addr = random.choice(worker_addrs)
    
    client = openai.OpenAI(
        base_url=f"{worker_addr}/v1",
        api_key="sglang-engine-token",
    )
    
    generate_kwargs['max_tokens'] = generate_kwargs.get('max_tokens', chat_tokenizer.max_length) # for sglang, max_tokens is required and must > 0
    args_names, kwargs_names = get_function_arg_names(client.completions.create)
    extra_body_params = {}
    for key in list(generate_kwargs.keys()):
        if key not in args_names + kwargs_names:
            extra_body_params[key] = generate_kwargs[key]
            del generate_kwargs[key]
    generate_kwargs["extra_body"] = extra_body_params
    
    stream = generate_kwargs.get("stream", False)
    if stream:
        generate_kwargs.pop("n", None)
    @with_timeout(timeout)
    def get_response():
        while True:
            try:
                completion = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    **generate_kwargs,
                )
                break
            except openai.APIConnectionError as e:
                if not worker_initiated:
                    time.sleep(5)
                    continue
                print(f"API connection error: {e}")
                time.sleep(5)
                continue
        
        # return completion.choices[0].text
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
    return get_response()