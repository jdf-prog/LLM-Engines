import os
import json
import hashlib
import time
from datetime import datetime
from openai import OpenAI
from typing import List, Union
from pathlib import Path
from tqdm import tqdm

batch_submission_status_file = Path(os.path.expanduser(f"~/llm_engines/generation_cache")) / "openai_batch_cache" / "batch_submission_status.json"
# no image, multi-turn, do not use openai_generate, but can refer to it
def call_worker_openai(messages:List[str], model_name, timeout:int=60, conv_system_msg=None, **generate_kwargs) -> str:
    # change messages to openai format
    new_messages = []
    if conv_system_msg:
        new_messages.append({"role": "system", "content": conv_system_msg})
    for i, message in enumerate(messages):
        new_messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": message})
    # initialize openai client
    client = OpenAI()
    # call openai
    completion = client.chat.completions.create(
        model=model_name,
        messages=new_messages,
        timeout=timeout,
        **generate_kwargs,
    )
    stream = generate_kwargs.get("stream", False)
    if not stream:
        if len(completion.choices) > 1:
            return [c.message.content for c in completion.choices]
        else:
            return completion.choices[0].message.content
    else:
        def generate_stream():
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        return generate_stream()

def call_worker_openai_completion(prompt:str, model_name, timeout:int=60, **generate_kwargs) -> str:
    # initialize openai client
    client = OpenAI()
    # call openai
    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        timeout=timeout,
        **generate_kwargs,
    )
    stream = generate_kwargs.get("stream", False)
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
    
def save_batch_file(batch_messages:List[Union[str, dict]], model:str, batch_name:str=None, cache_dir=None, custom_ids=None, **generate_kwargs):
    if isinstance(batch_messages[0], str):
        batch_messages = [{"role": "user", "content": message} for message in batch_messages]

    if cache_dir is None:
        cache_dir = Path(os.path.expanduser(f"~/llm_engines/generation_cache")) / "openai_batch_cache"
    else:
        cache_dir = Path(cache_dir) / "openai_batch_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if custom_ids is None:
        custom_ids = [f"request-{i+1}" for i in range(len(batch_messages))]
    assert len(custom_ids) == len(batch_messages)
    if batch_name is None:
        # batch_name = f"{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        hash_str = "".join([str(message) for message in batch_messages])
        batch_name = f"{model}_{hashlib.md5(hash_str.encode()).hexdigest()}"
    file_path = cache_dir / f"{batch_name}.jsonl"
    with open(file_path, "w") as f:
        for custom_id, message in zip(custom_ids, batch_messages):
            if isinstance(message, str):
                message = {"role": "user", "content": message}
            f.write(json.dumps({"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": {"model": model, "messages": message, **generate_kwargs}}, ensure_ascii=False) + "\n")
    return file_path

def submit_batch_file(batch_file:str, output_path:str=None, project_name:str=None, description:str=None):
    """
    Submit a batch of queries to OpenAI API
    Args:
        batch_file: str, path to the batch jsonl file
        output_path: str, path to the output file; if not provided, will be saved in the same directory as the batch_file, ending with ".batch_results.jsonl"
        project_name: str, project name, default to "vapo", keyword to filter batches
        description: str, description of the batch, default to the batch_file name, can also be used to filter batches
    Returns:
        batch_result_id: str, the id of the batch submission
    """
    if isinstance(batch_file, str):
        batch_file = Path(batch_file)
        
    # internally maintain a batch submission status json
    print("Loading batch submission status from", batch_submission_status_file)
    if batch_submission_status_file.exists():
        with open(batch_submission_status_file, "r") as f:
            batch_submission_status = json.load(f)
    else:
        batch_submission_status = {}
        
    
    client = OpenAI()
    batch_input_file = client.files.create(
        file=open(batch_file, "rb"),
        purpose="batch"
    )
    if not project_name:
        project_name = "llm_engines"
    description = description if description is not None else batch_file.stem
    output_path = output_path if output_path is not None else batch_file.parent / f"{description}.batch_results.jsonl"
    batch_input_file_id = batch_input_file.id
    
    batch_file_hash = hashlib.md5(batch_file.read_bytes()).hexdigest()
    batch_file_size = batch_file.stat().st_size
    
    for key, value in batch_submission_status.items():
        value_input_file_metadata = value["input_path_metadata"]
        if not os.path.samefile(value["input_path"], batch_file):
            # print(f"Batch {key} has a different input file. need to resubmit.")
            continue
        if batch_file_size != value_input_file_metadata["size"]:
            # print(f"Batch {key} has a newer version of the input file. need to resubmit.")
            continue
        if batch_file_hash != value_input_file_metadata["hash"]:
            # print(f"Batch {key} has a different input hash. need to resubmit.")
            continue
        print(f"Batch {key} already submitted. Skipping submission.")
        return key
        
    completion_window = "24h"
    endpoint = "/v1/chat/completions"
    batch_result = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata={
            "project": project_name,
            "description": description,
            "input_path": str(batch_file),
            "output_path": str(output_path)
        }
    )
    print(f"Batch {batch_result.id} submitted")
    
    
    # time should be in the current timezone, in the format like 2022-01-01T00:00:00
    batch_submission_status[batch_result.id] = {    
        "project": project_name,
        "description": description,
        "endpoint": endpoint,
        "completion_window": completion_window,
        "input_path": str(batch_file),
        "input_path_metadata": {
            "hash": hashlib.md5(batch_file.read_bytes()).hexdigest(),
            "size": batch_file.stat().st_size
        },
        "output_path": str(output_path),
        "batch_input_file_id": batch_input_file_id,
        "batch_result_id": batch_result.id,
        "status": "submitted",
        "timeline": {
            "submitted": str(datetime.now()),
            "completed": None,
            "failed": None,
            "downloaded": None
        },
        "last_updated": str(datetime.now()),
        "openai_batch_metadata": batch_result.to_dict()
    }
    with open(batch_submission_status_file, "w") as f:
        json.dump(batch_submission_status, f, indent=4)
    return batch_result.id

def check_batch_status(batch_result_id, overwrite:bool=False):
     # internally maintain a batch submission status json
    if batch_submission_status_file.exists():
        with open(batch_submission_status_file, "r") as f:
            batch_submission_status = json.load(f)
    else:
        batch_submission_status = {}
    
    client = OpenAI()
    batch = client.batches.retrieve(batch_result_id)
    batch_id = batch.id
    batch_status = batch.status
    batch_desc = batch.metadata["description"] if batch.metadata is not None and "description" in batch.metadata else ""
    batch_project_name = batch.metadata["project"] if batch.metadata is not None and "project" in batch.metadata else ""
    if batch.metadata is not None and "output_path" in batch.metadata:
        batch_output_path = batch.metadata["output_path"]
    else:
        batch_output_path = f"./batch_results/{batch_id}.batch_results.jsonl"
        batch_submission_status[batch_id]["output_path"] = batch_output_path
    # print(f"{batch_id: <20} {batch_status: <20} {batch_project_name: <20} {batch_desc: <20}")
    if batch_status == "completed":
        print(f"Downloading output file for batch {batch_id}")
        content = client.files.content(batch.output_file_id)
        output_path = batch_output_path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            print(f"File {output_path} already exists. Skipping writing to file.")
        else:
            if output_path.exists() and overwrite:
                print(f"Overwriting file {output_path}")
            content.write_to_file(output_path)
            print(f"Output file written to {batch_desc}.jsonl")
        batch_submission_status[batch_id]["status"] = "completed"
        batch_submission_status[batch_id]["timeline"]["completed"] = str(datetime.now())
        batch_submission_status[batch_id]["timeline"]["downloaded"] = str(datetime.now())
        batch_submission_status[batch_id]["last_updated"] = str(datetime.now())
        batch_submission_status[batch_id]["openai_batch_metadata"].update(batch.to_dict())
    elif batch_status == "failed":
        print(f"Batch {batch_id} failed.")
        batch_submission_status[batch_id]["status"] = "failed"
        batch_submission_status[batch_id]["timeline"]["failed"] = str(datetime.now())
        batch_submission_status[batch_id]["last_updated"] = str(datetime.now())
        batch_submission_status[batch_id]["openai_batch_metadata"].update(batch.to_dict())
    else:
        batch_submission_status[batch_id]["status"] = batch_status
        batch_submission_status[batch_id]["last_updated"] = str(datetime.now())
        batch_submission_status[batch_id]["openai_batch_metadata"].update(batch.to_dict())
    with open(batch_submission_status_file, "w") as f:
        json.dump(batch_submission_status, f, indent=4)
    return batch_submission_status[batch_id]

def openai_batch_request(
    model_name:str,
    batch_messages:List[Union[str, List[str], List[dict]]],
    conv_system_msg:str=None,
    desc:str=None,
    **generate_kwargs
):
    if isinstance(batch_messages[0], str):
        batch_messages = [[{"role": "user", "content": message}] for message in batch_messages]
    elif isinstance(batch_messages[0], list):
        if isinstance(batch_messages[0][0], str):
            batch_messages = [[{
                "role": "user" if i % 2 == 0 else "assistant",
                "content": message
            } for i, message in enumerate(messages)] for messages in batch_messages]
        elif isinstance(batch_messages[0][0], dict):
            assert all("role" in message for message in batch_messages[0]), "Error: role key not found in the message"
            assert all("content" in message for message in batch_messages[0]), "Error: content key not found in the message"
        else:
            raise ValueError("Error: unknown message format")
    else:
        raise ValueError("Error: unknown message format")
    if conv_system_msg:
        batch_messages = [[{"role": "system", "content": conv_system_msg}] + messages for messages in batch_messages]
    if "stream" in generate_kwargs:
        generate_kwargs.pop("stream")
    batch_file = save_batch_file(batch_messages, model_name, **generate_kwargs)
    batch_result_id = submit_batch_file(batch_file)
    num_total = len(batch_messages)
    tqdm_bar = tqdm(total=num_total, desc=desc or "LLMEngine Batch Inference")
    while True:
        batch_status = check_batch_status(batch_result_id)
        num_completed = batch_status["openai_batch_metadata"]['request_counts']['completed']
        num_total = batch_status["openai_batch_metadata"]['request_counts']['total']
        num_failed = batch_status["openai_batch_metadata"]['request_counts']['failed']
        tqdm_bar.n = num_completed
        tqdm_bar.total = num_total
        tqdm_bar.set_postfix(completed=num_completed, total=num_total, failed=num_failed)
        if batch_status["status"] == "completed":
            tqdm_bar.close()
            break
        elif batch_status["status"] == "finalizing":
            tqdm_bar.desc = "Finalizing"
            tqdm_bar.refresh()
        elif batch_status["status"] == "validating":
            tqdm_bar.desc = "Validating"
            tqdm_bar.refresh()
        elif batch_status["status"] == "failed":
            tqdm_bar.close()
            print("Batch failed")
            break
        elif batch_status["status"] == "in_progress":
            tqdm_bar.desc = "In Progress"
            tqdm_bar.refresh()
        elif batch_status["status"] == "cancelled":
            tqdm_bar.close()
            print("Batch cancelled")
            break
        elif batch_status["status"] == "expired":
            tqdm_bar.close()
            print("Batch expired")
            break
        elif batch_status["status"] == "cancelling":
            tqdm_bar.desc = "Cancelling"
            tqdm_bar.refresh()
        else:
            tqdm_bar.desc = batch_status["status"]
            tqdm_bar.refresh()
        time.sleep(10)
        
    if batch_status["status"] == "completed":
        output_path = batch_status["output_path"]
        results = []
        with open(output_path, "r") as f:
            results = [json.loads(line) for line in f.readlines()]
        all_completions = [[choice['message']['content'] for choice in x['response']['body']['choices']] for x in results]
        if all(len(x) == 1 for x in all_completions):
            all_completions = [x[0] for x in all_completions]
        results = all_completions
    else:
        results = None
    return results

if __name__ == "__main__":
    from icecream import ic
    ic(call_worker_openai(["Hello"], "gpt-3.5-turbo"))
    ic(call_worker_openai_completion("Hello", "gpt-3.5-turbo-instruct"))