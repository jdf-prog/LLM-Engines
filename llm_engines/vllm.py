import os
import time
import torch
import requests
import json
from pathlib import Path
from typing import List
from .utils import SubprocessMonitor, ChatTokenizer
worker_initiated = False

def launch_vllm_worker(
    model_name: str,
    use_vllm: bool=True,
    num_gpus: int=None,
    gpu_ids: List[int]=None,
    port: int=34200,
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
    worker_addr = f"http://127.0.0.1:{port}"
    log_file = Path(os.path.abspath(__file__)).parent / "logs" / f"{model_name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if not num_gpus:
        num_gpus = torch.cuda.device_count()
        print(f"Warning: num_gpus not provided, using {num_gpus} GPUs")
    if not gpu_ids:
        gpu_ids = list(range(num_gpus))
    env = os.environ.copy()
    # Set the CUDA_VISIBLE_DEVICES environment variable
    env["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu_id) for gpu_id in gpu_ids])
    if use_vllm:
        proc = SubprocessMonitor([
            "python3", "-m", "fastchat.serve.vllm_worker",
            "--model-path", model_name,
            "--port", str(port),
            "--worker-address", worker_addr,
            "--host", "127.0.0.1",
            "--num-gpus", str(num_gpus) if num_gpus is not None else "1",
            "--no-register",
            "--disable-log-stats",
            "--disable-log-requests"
        ], env=env)
        print(f"Launched VLLM model {model_name} at address {worker_addr}")
    else:
        proc = SubprocessMonitor([
            "python3", "-m", "fastchat.serve.model_worker",
            "--model-path", model_name,
            "--port", str(port),
            "--worker", worker_addr,
            "--host", "127.0.0.1",
            "--num-gpus", str(num_gpus) if num_gpus is not None else "1",
            "--no-register",
            "--disable-log-stats",
            "--disable-log-requests"
        ], env=env)
        print(f"Launched Normal model {model_name} at address {worker_addr}")
    print(f"Launching VLLM model {model_name} with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    return f"http://127.0.0.1:{port}", proc

chat_tokenizers = {}

def call_vllm_worker(messages, model_name, worker_addrs, conv_system_msg=None, **generate_kwargs) -> str:
    global worker_initiated
    global chat_tokenizers
    
    if not model_name in chat_tokenizers:
        chat_tokenizers[model_name] = ChatTokenizer(model_name)
    chat_tokenizer = chat_tokenizers[model_name]
    
    prompt = chat_tokenizer(messages)
    
    params = {
        "prompt": prompt,
        **generate_kwargs
    }
    timeout = 100
    
    if not hasattr(call_vllm_worker, "worker_id_to_call"):
        call_vllm_worker.worker_id_to_call = 0
    call_vllm_worker.worker_id_to_call = (call_vllm_worker.worker_id_to_call + 1) % len(worker_addrs)
    worker_addr = worker_addrs[call_vllm_worker.worker_id_to_call]
    
    max_retry = 5
    retry = 0
    while True:
        try:
            # starlette StreamingResponse
            response = requests.post(
                worker_addr + "/worker_generate",
                json=params,
                stream=True,
                timeout=timeout,
            )
            if response.status_code == 200:
                worker_initiated = True
            break
        except requests.exceptions.ConnectionError as e:
            if not worker_initiated:
                time.sleep(5) 
                continue
            if retry > max_retry:
                return None
            retry += 1  
            print("Connection error, retrying...")
            time.sleep(5)
        except requests.exceptions.ReadTimeout as e:
            print("Read timeout, adding 10 seconds to timeout and retrying...")
            timeout += 10
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            print("Unknown request exception: ", e, "retrying...")
            time.sleep(5)
        except Exception as e:
            print("Unknown exception: ", e, "retrying...")
            raise e
            
    generated_text = json.loads(response.content.decode("utf-8"))['text']
    generated_text = generated_text.strip("\n ")
    return generated_text