import os
import sys
import random
import atexit
import signal
import psutil
from functools import partial
from .utils import generation_cache_wrapper, retry_on_failure, MaxRetriesExceededError

ENGINES = ["vllm", "sglang", "openai", "gemini", "mistral", "together", "claude"]
workers = []
def get_call_worker_func(
    model_name, 
    worker_addrs=None, 
    cache_dir=None, 
    conv_system_msg=None,
    use_cache=True,
    completion=False, 
    num_workers=1,
    num_gpu_per_worker=1,
    dtype="auto",
    engine="vllm",
    max_retry=None,
) -> str:
    """
    Return a function that calls the model worker, takes a list of messages (user, gpt, user, ...) and returns the generated text
    Args:
        model_name: model name
        worker_addrs: worker addresses, if None, launch local workers
        cache_dir: cache directory
        conv_system_msg: conversation system message
        use_cache: use cache or not
        completion: use completion or not (use chat by default)
        num_workers: number of workers
        num_gpu_per_worker: number of gpus per worker
        dtype: data type
        engine: engine name
    """
    if engine == "openai":
        from .openai_text import call_worker_openai, call_worker_openai_completion
        call_model_worker = call_worker_openai if not completion else call_worker_openai_completion
    elif engine == "gemini":
        if completion:
            raise ValueError(f"Engine {engine} does not support completion")
        from .gemini import call_worker_gemini
        call_model_worker = call_worker_gemini
    elif engine == "claude":
        if completion:
            raise ValueError(f"Engine {engine} does not support completion")
        from .claude import call_worker_claude
        call_model_worker = call_worker_claude
    elif engine == "mistral":
        if completion:
            raise ValueError(f"Engine {engine} does not support completion")
        from .mistral import call_worker_mistral
        call_model_worker = call_worker_mistral
    elif engine == "together":
        from .together import call_worker_together, call_worker_together_completion
        call_model_worker = call_worker_together if not completion else call_worker_together_completion
    elif engine in ["vllm", "sglang"]:
        if engine == "vllm":
            from .vllm import launch_vllm_worker, call_vllm_worker, call_vllm_worker_completion
            call_worker_func = call_vllm_worker if not completion else call_vllm_worker_completion
            launch_worker_func = launch_vllm_worker
        elif engine == "sglang":
            from .sglang import launch_sglang_worker, call_sglang_worker, call_sglang_worker_completion
            call_worker_func = call_sglang_worker if not completion else call_sglang_worker_completion
            launch_worker_func = launch_sglang_worker
        else:
            raise ValueError(f"Internal error: engine {engine} not supported")
        if worker_addrs is None:
            import torch
            
            print(f"Launching model worker {model_name} locally")
            worker_addrs = []
            total_gpus = torch.cuda.device_count()
            if total_gpus < num_workers * num_gpu_per_worker:
                if total_gpus >= num_gpu_per_worker:
                    print(f"Warning: total gpus ({total_gpus}) is less than num_workers * num_gpu_per_worker ({num_workers * num_gpu_per_worker}), using {total_gpus // num_gpu_per_worker} workers instead")
                    num_workers = total_gpus // num_gpu_per_worker
                else:
                    print(f"Error: total gpus ({total_gpus}) is less than num_gpu_per_worker ({num_gpu_per_worker}), exiting...")
                    sys.exit(1)
                    
            if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
                gpus_ids = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
                gpu_ids = [int(gpu_id) for gpu_id in gpus_ids]
            else:
                gpu_ids = list(range(total_gpus))
            start_port = random.randint(31000, 32000)
            for i in range(num_workers):
                worker_addr, worker = launch_worker_func(model_name, 
                    num_gpus=num_gpu_per_worker, 
                    gpu_ids=gpu_ids[i*num_gpu_per_worker:(i+1)*num_gpu_per_worker], 
                    port=start_port+i*10,
                    dtype=dtype)
                worker_addrs.append(worker_addr)
                workers.append(worker)
        else:
            print(f"Using existing worker at {worker_addrs}")
            if not isinstance(worker_addrs, list):
                worker_addrs = [worker_addr]
        call_model_worker = partial(call_worker_func, worker_addrs=worker_addrs)        
    else:
        raise ValueError(f"Engine {engine} not supported, available engines: {ENGINES}")
    
    # wrap the call_model_worker with the model_name and other arguments
    if not completion:
        call_model_worker = partial(call_model_worker, model_name=model_name, 
            conv_system_msg=conv_system_msg)
        # test local worker connection
        test_response = call_model_worker(["Hello"], temperature=0, max_tokens=256, timeout=None)
    else:
        call_model_worker = partial(call_model_worker, model_name=model_name)
        # test local worker connection
        test_response = call_model_worker("Hello", temperature=0, max_tokens=256, timeout=None)
    if not test_response:
        print("Error: failed to connect to the worker, exiting...")
        for worker in workers:
            cleanup_process(worker)
        sys.exit(1)
    else:
        print(f"Successfully connected to the workers")
        print("Test prompt: \n", "Hello")
        print("Test response: \n", test_response)
    # add cache wrapper
    if use_cache:
        call_model_worker = generation_cache_wrapper(call_model_worker, model_name, cache_dir)
    else:
        print("Cache is disabled")
    call_model_worker = retry_on_failure(call_model_worker, num_retries=max_retry)
    return call_model_worker


def kill_process_and_children(pid):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"No process with PID {pid} found.")
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            child.kill()
            # print(f"Killed child process {child.pid}")
        except psutil.NoSuchProcess:
            print(f"Child process {child.pid} already terminated.")
    
    try:
        parent.kill()
        # print(f"Killed parent process {pid}")
    except psutil.NoSuchProcess:
        print(f"Parent process {pid} already terminated.")
        
        
def cleanup_process(proc):
    # os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    # os.kill(proc.pid, signal.SIGTERM)
    kill_process_and_children(proc.pid)
    print("Model Worker terminated.")

@atexit.register
def cleanup_all_workers():
    for worker in workers:
        cleanup_process(worker)
    if workers:
        print("All workers terminated.")
    workers.clear()
