import anthropic
import os
import requests
import json
import hashlib
import time
import filelock
import random
import regex as re
from copy import deepcopy
from typing import List, Union
from anthropic import NOT_GIVEN
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from .utils import with_timeout, is_base64_image_url, encode_base64_image, load_image, decode_base64_image_url
batch_submission_status_file = Path(os.path.expanduser(f"~/llm_engines/generation_cache")) / "claude_batch_cache" / "batch_submission_status.json"
batch_submission_status_file.parent.mkdir(parents=True, exist_ok=True)


def read_batch_submission_status():
    print("Loading batch submission status from", batch_submission_status_file)
    if batch_submission_status_file.exists():
        
        lock = filelock.FileLock(str(batch_submission_status_file) + ".lock", timeout=30)
        try:
            with lock:
                with open(batch_submission_status_file, "r") as f:
                    batch_submission_status = json.load(f)
        except filelock.Timeout as e:
            print("Timeout acquiring lock")
            raise e
    else:
        batch_submission_status = {}
    return batch_submission_status

def write_batch_submission_status(batch_submission_status):
    batch_submission_status_file.parent.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(str(batch_submission_status_file) + ".lock", timeout=30)
    try:
        with lock:
            with open(batch_submission_status_file, "w") as f:
                json.dump(batch_submission_status, f, indent=4)
    except filelock.Timeout as e:
        print("Timeout acquiring lock")
        raise e

# 5MB
MAX_IMAGE_SIZE = 5 * 1024 * 1024
def preprocess_claude_messages(messages:List[dict]) -> List[dict]:
    messages = deepcopy(messages)
    for message in messages:
        if isinstance(message['content'], list):
            for sub_message in message['content']:
                if sub_message['type'] == "image_url":
                    if is_base64_image_url(sub_message['image_url']['url']):
                        # if image size is greater than 5MB, decode and resize and re-encode
                        im64 = sub_message['image_url']['url'].split(",", 1)[1]
                        current_size = len(im64)
                        # print("current_size", current_size)
                        if current_size > MAX_IMAGE_SIZE:
                            print("Warning: Image size is greater than 5MB. Resizing image due to Claude API limit.")
                            image = decode_base64_image_url(sub_message['image_url']['url'])
                            image_format = image.format if image.format else "png"
                            scale_factor = (MAX_IMAGE_SIZE / current_size) ** 0.6
                            image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)))
                            im64 = encode_base64_image(image, image_format)
                            media_type = "image/png"
                        else:
                            start_idx = sub_message['image_url']['url'].find("image/")
                            end_idx = sub_message['image_url']['url'].find(";base64")
                            media_type = sub_message['image_url']['url'][start_idx:end_idx].lower()
                    else:
                        image = load_image(sub_message['image_url']['url'])
                        current_size = image.size[0] * image.size[1] * 3
                        if current_size > MAX_IMAGE_SIZE:
                            print("Warning: Image size is greater than 5MB. Resizing image due to Claude API limit.")
                            scale_factor = (MAX_IMAGE_SIZE / current_size) ** 0.6
                            image = image.resize((int(image.size[0] * scale_factor), int(image.size[1] * scale_factor)))
                        image_format = image.format if image.format else "png"
                        im64= encode_base64_image(image, image_format)
                        media_type = f"image/{image_format}".lower()
                    sub_message['source'] = {
                        "type": "base64",
                        "media_type": media_type,
                        "data": im64
                    }
                    sub_message['type'] = "image"
                    sub_message.pop('image_url')
    return messages
            
# no image, multi-turn, do not use openai_generate, but can refer to it
def call_worker_claude(messages:List[str], model_name, timeout:int=60, conv_system_msg=None, **generate_kwargs) -> str:
    # change messages to mistral format
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # change messages to openai format
    if conv_system_msg:
        new_messages = [{"role": "system", "content": conv_system_msg}] + messages
    else:
        new_messages = messages
    new_messages = preprocess_claude_messages(new_messages)
             
    generate_kwargs.pop("n", None) # claude does not have n
    if not generate_kwargs.get("max_tokens", None):
        generate_kwargs["max_tokens"] = 1024
    stream = generate_kwargs.pop("stream", False)
    if "logprobs" in generate_kwargs:
        raise ValueError("Error: logprobs is not supported in claude")
    @with_timeout(timeout)
    def get_response():
        completion = client.messages.create(
            model=model_name,
            messages=new_messages,
            system=conv_system_msg if conv_system_msg else NOT_GIVEN,
            timeout=timeout,
            **generate_kwargs,
        )
        if len(completion.content) > 1:
            return [c.text for c in completion.content]
        else:
            return completion.content[0].text
        
    @with_timeout(timeout)
    def stream_response():
        with client.messages.stream(
            model=model_name,
            messages=new_messages,
            system=conv_system_msg if conv_system_msg else NOT_GIVEN,
            timeout=timeout,
            **generate_kwargs,
        ) as stream:
            for text in stream.text_stream:
                yield text
    
    if not stream:
        return get_response()
    else:
        return stream_response()

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
            f.write(json.dumps({"custom_id": custom_id, "params": {"model": model, "messages": message, **generate_kwargs}}, ensure_ascii=False) + "\n")
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
    batch_submission_status = read_batch_submission_status()
        
    
    client = anthropic.Anthropic()
    with open(batch_file, "r") as f:
        batch_inputs = [json.loads(line) for line in f.readlines()]
    if not project_name:
        project_name = "llm_engines"
    description = description if description is not None else batch_file.stem
    output_path = output_path if output_path is not None else batch_file.parent / f"{description}.batch_results.jsonl"
    
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
        if value['status'] in ["errored", "expired", "canceled"]:
            continue
        print(f"Batch {key} is still in progress. Skipping submission.")
        return key
    
    
    batch_result = client.beta.messages.batches.create(
        requests=batch_inputs,
    )
    print(f"Batch {batch_result.id} submitted")
    
    claude_batch_metadata = batch_result.to_dict()
    for key in claude_batch_metadata:
        if isinstance(claude_batch_metadata[key], datetime):
            claude_batch_metadata[key] = str(claude_batch_metadata[key])
    # time should be in the current timezone, in the format like 2022-01-01T00:00:00
    batch_submission_status[batch_result.id] = {    
        "project": project_name,
        "description": description,
        "endpoint": None,
        "completion_window": None,
        "input_path": str(batch_file),
        "input_path_metadata": {
            "hash": hashlib.md5(batch_file.read_bytes()).hexdigest(),
            "size": batch_file.stat().st_size,
            "total": len(batch_inputs)
        },
        "output_path": str(output_path),
        "batch_input_file_id": None,
        "batch_result_id": batch_result.id,
        "status": "submitted",
        "timeline": {
            "submitted": str(datetime.now()),
            "completed": None,
            "failed": None,
            "downloaded": None,
        },
        "last_updated": str(datetime.now()),
        "claude_batch_metadata": claude_batch_metadata
    }
    
    write_batch_submission_status(batch_submission_status)
    return batch_result.id

def check_batch_status(batch_result_id, overwrite:bool=False):
    batch_submission_status = read_batch_submission_status()
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    try:
        batch = client.beta.messages.batches.retrieve(batch_result_id)
    except anthropic.NotFoundError as e:
        return None
    batch_id = batch.id
    batch_status = batch.processing_status
    batch_desc = batch_submission_status[batch_id]["description"] if batch_id in batch_submission_status else ""
    batch_project_name = batch_submission_status[batch_id]["project"] if batch_id in batch_submission_status else ""
    if batch_submission_status[batch_id]["output_path"]:
        batch_output_path = batch_submission_status[batch_id]["output_path"]
    else:
        batch_output_path = f"./batch_results/{batch_id}.batch_results.jsonl"
        batch_submission_status[batch_id]["output_path"] = batch_output_path
    # print(f"{batch_id: <20} {batch_status: <20} {batch_project_name: <20} {batch_desc: <20}")
    claude_batch_metadata = batch.to_dict()
    for key in claude_batch_metadata:
        if isinstance(claude_batch_metadata[key], datetime):
            claude_batch_metadata[key] = str(claude_batch_metadata[key])
    if batch_status == "ended":
        output_path = batch_output_path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            print(f"File {output_path} already exists. Skipping writing to file.")
        else:
            if output_path.exists() and overwrite:
                print(f"Overwriting file {output_path}")
                
            # retrieve the results
            print(f"Downloading output file for batch {batch_id}")
            results = client.beta.messages.batches.results(batch_id)
            # save url to file via web requests, download the results_url file, and save to output_path
            with open(output_path, "wb") as f:
                for result in results:
                    f.write(json.dumps(result.to_dict(), ensure_ascii=False).encode() + b"\n")
            print(f"Output file written to {output_path}")
        with open(output_path, "r") as f:
            results = [json.loads(line) for line in f.readlines()]
        all_result_types = [x['result']['type'] for x in results]
        if all(all_result_types) == "errored" or all(all_result_types) == "error":
            batch_status = "errored"
            batch_submission_status[batch_id]["status"] = "errored"
            batch_submission_status[batch_id]["timeline"]["failed"] = str(datetime.now())
        elif all(all_result_types) == "canceled":
            batch_status = "canceled"
            batch_submission_status[batch_id]["status"] = "canceled"
            batch_submission_status[batch_id]["timeline"]["canceled"] = str(datetime.now())
        elif all(all_result_types) == "expired":
            batch_status = "expired"
            batch_submission_status[batch_id]["status"] = "expired"
            batch_submission_status[batch_id]["timeline"]["expired"] = str(datetime.now())
        else:
            batch_submission_status[batch_id]["status"] = "completed"
        batch_submission_status[batch_id]["timeline"]["completed"] = str(datetime.now())
        batch_submission_status[batch_id]["timeline"]["downloaded"] = str(datetime.now())
        batch_submission_status[batch_id]["last_updated"] = str(datetime.now())
        batch_submission_status[batch_id]["claude_batch_metadata"].update(claude_batch_metadata)
    else:
        batch_submission_status[batch_id]["status"] = batch_status
        batch_submission_status[batch_id]["last_updated"] = str(datetime.now())
        batch_submission_status[batch_id]["claude_batch_metadata"].update(claude_batch_metadata)
    write_batch_submission_status(batch_submission_status)
    return batch_submission_status[batch_id]

def get_batch_progress(batch_result_id):
    batch_status = check_batch_status(batch_result_id)
    num_succeeded = batch_status["claude_batch_metadata"]['request_counts']['succeeded']
    num_processing = batch_status["claude_batch_metadata"]['request_counts']['processing']
    num_errored = batch_status["claude_batch_metadata"]['request_counts']['errored']
    num_expired = batch_status["claude_batch_metadata"]['request_counts']['expired']
    num_canceled = batch_status["claude_batch_metadata"]['request_counts']['canceled']
    num_total = num_succeeded + num_processing + num_errored + num_expired + num_canceled
    n = num_succeeded + num_errored + num_expired + num_canceled
    total = num_total
    tqdm_postfix = {
        "completed": num_succeeded,
        "processing": num_processing,
        "errored": num_errored,
        "expired": num_expired,
        "canceled": num_canceled
    }
    return n, total, tqdm_postfix, batch_status["status"]

def get_batch_result(batch_id, generate_kwargs={}):
    batch_status = check_batch_status(batch_id)
    if batch_status["status"] == "completed":
        output_path = batch_status["output_path"]
        results = []
        with open(output_path, "r") as f:
            raw_results = [json.loads(line) for line in f.readlines()]
        results = []
        for item in raw_results:
            if item["result"]["type"] == "succeeded":
                results.append(item["result"]['message']["content"][0]['text'])
            else:
                results.append(None)
        print("Batch requests status counts:", batch_status["claude_batch_metadata"]['request_counts'])
    else:
        results = None
    return results

def claude_batch_request(
    model_name:str,
    batch_messages:List[Union[str, List[str], List[dict]]],
    conv_system_msg:str=None,
    desc:str=None,
    detach:bool=False,
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
    if detach:
        return batch_result_id
    num_total = len(batch_messages)
    tqdm_bar = tqdm(total=num_total, desc=desc or "LLMEngine Batch Inference")
    while True:
        batch_status = check_batch_status(batch_result_id)
        assert batch_status is not None, f"Error: batch {batch_result_id} not found in Anthropic API"
        num_succeeded = batch_status["claude_batch_metadata"]['request_counts']['succeeded']
        num_processing = batch_status["claude_batch_metadata"]['request_counts']['processing']
        num_errored = batch_status["claude_batch_metadata"]['request_counts']['errored']
        num_expired = batch_status["claude_batch_metadata"]['request_counts']['expired']
        num_canceled = batch_status["claude_batch_metadata"]['request_counts']['canceled']
        num_total = num_succeeded + num_processing + num_errored + num_expired + num_canceled
        assert num_total == len(batch_messages), f"Error: total number of requests {num_total} does not match the number of requests {len(batch_messages)}"
        tqdm_bar.n = num_succeeded + num_errored + num_expired + num_canceled
        tqdm_bar.total = num_total
        # tqdm_bar.set_postfix(completed=num_completed, total=num_total, failed=num_failed)
        tqdm_bar.set_postfix(
            completed=num_succeeded,
            processing=num_processing,
            errored=num_errored,
            expired=num_expired,
            canceled=num_canceled
        )
        if batch_status["status"] in ["completed", "errored", "expired", "canceled"]:
            tqdm_bar.close()
            break
        elif batch_status["status"] == "in_progress":
            tqdm_bar.desc = "In Progress"
            tqdm_bar.refresh()
        else:
            tqdm_bar.desc = batch_status["status"]
            tqdm_bar.refresh()
        time.sleep(random.randint(5, 10))
    
    results = get_batch_result(batch_result_id)
    return results


if __name__ == "__main__":
    from icecream import ic
    ic(call_worker_claude(["Hello", "Hi, I am claude", "What did I ask in the last response?"], "claude-3-opus-20240229"))