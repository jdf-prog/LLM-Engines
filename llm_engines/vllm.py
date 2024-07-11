import os
import time
import torch
import requests
import json
import openai
import vllm
from pathlib import Path
from typing import List
from .utils import SubprocessMonitor, ChatTokenizer
worker_initiated = False

def launch_vllm_worker(
    model_name: str,
    use_vllm: bool=True,
    num_gpus: int=None,
    gpu_ids: List[int]=None,
    dtype: str="auto",
    port: int=34200,
    host: str="127.0.0.1",
    root_path: str=None,
    subprocess: bool=True,
) -> str:
    """
    Launch a model worker and return the address
    Args:
        model_name: the model name to launch
    Returns:
        the address of the launched model
    """
    print(f"Launching model {model_name}")
    # python3 -m arena.serve.model_worker --model-path liuhaotian/llava-v1.6-vicuna-7b --port 31011 --worker http://127.0.0.1:31011 --host=127.0.0.1 --no-register
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
    print(num_gpus, gpu_ids)
    if use_vllm:
        # python -m vllm.entrypoints.openai.api_server --model NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
        proc = SubprocessMonitor([
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--dtype", dtype,
            "--api-key", "vllm-engine-token",
            "--port", str(port),
            "--host", host,
            "--tensor-parallel-size", str(num_gpus),
            "--disable-log-requests",
        ] + (["--root-path", root_path] if root_path else [])
        ,env=env)
        print(f"Launched VLLM model {model_name} at address {worker_addr}")
    print(f"Launching VLLM model {model_name} with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    return f"http://127.0.0.1:{port}", proc

chat_tokenizers = {}

def call_vllm_worker(messages, model_name, worker_addrs, conv_system_msg=None, **generate_kwargs) -> str:
    global worker_initiated
    global chat_tokenizers
    
    if model_name not in chat_tokenizers:
        chat_tokenizers[model_name] = ChatTokenizer(model_name)
    chat_tokenizer = chat_tokenizers[model_name]
    
    chat_messages = []
    if conv_system_msg:
        chat_messages.append({"role": "system", "content": conv_system_msg})
    for i, message in enumerate(messages):
        chat_messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": message})

    prompt = chat_tokenizer(chat_messages)

    if not hasattr(call_vllm_worker, "worker_id_to_call"):
        call_vllm_worker.worker_id_to_call = 0
    call_vllm_worker.worker_id_to_call = (call_vllm_worker.worker_id_to_call + 1) % len(worker_addrs)
    worker_addr = worker_addrs[call_vllm_worker.worker_id_to_call]
    
    client = openai.OpenAI(
        base_url=f"{worker_addr}/v1",
        api_key="vllm-engine-token",
    )
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
    
    return completion.choices[0].message.content
    
def call_vllm_worker_completion(prompt:str, model_name, worker_addrs, **generate_kwargs) -> str:
    global worker_initiated
    
    if not hasattr(call_vllm_worker_completion, "worker_id_to_call"):
        call_vllm_worker_completion.worker_id_to_call = 0
    call_vllm_worker_completion.worker_id_to_call = (call_vllm_worker_completion.worker_id_to_call + 1) % len(worker_addrs)
    worker_addr = worker_addrs[call_vllm_worker_completion.worker_id_to_call]
    
    client = openai.OpenAI(
        base_url=f"{worker_addr}/v1",
        api_key="vllm-engine-token",
    )
    
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
    
    return completion.choices[0].text