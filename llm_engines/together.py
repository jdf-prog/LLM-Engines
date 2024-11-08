import os
import time
from .utils import with_timeout
together_client = None
def call_worker_together(messages, model_name, timeout:int=60, conv_system_msg=None, **generate_kwargs) -> str:
    from together import Together
    global together_client
    if together_client is None:
        together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"), timeout=timeout)

    if model_name.startswith("together_"):
        model_name = model_name.replace("together_", "")
    
    new_messages = []
    if conv_system_msg:
        new_messages.append({"role": "system", "content": conv_system_msg})
    for i, message in enumerate(messages):
        new_messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": message})
    
    stream = generate_kwargs.get("stream", False)
    if stream and "n" in generate_kwargs:
        generate_kwargs.pop("n")
        
    @with_timeout(timeout)
    def get_response():
        max_retry_for_unbound_local_error = 10
        retry_count = 0
        while True:
            try:
                completion = together_client.chat.completions.create(
                    model=model_name,
                    messages=new_messages,
                    **generate_kwargs,
                )
                break
            except UnboundLocalError as e:
                time.sleep(0.2)
                retry_count += 1
                if retry_count >= max_retry_for_unbound_local_error:
                    
                    raise e
                continue
        if not stream:
            if "logprobs" not in generate_kwargs:
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

def call_worker_together_completion(prompt:str, model_name, timeout:int=60, **generate_kwargs) -> str:
    from together import Together
    global together_client
    if together_client is None:
        together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"), timeout=timeout)

    if model_name.startswith("together_"):
        model_name = model_name.replace("together_", "")
    
    stream = generate_kwargs.get("stream", False)
    if stream and "n" in generate_kwargs:
        generate_kwargs.pop("n")
    @with_timeout(timeout)
    def get_response():
        completion = together_client.completions.create(
            model=model_name,
            prompt=prompt,
            **generate_kwargs,
        )
        if not stream:
            if len(completion.choices) > 1:
                return [c.text for c in completion.choices]
            else:
                return completion.choices[0].text
        else:
            def generate_stream():
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            return generate_stream()
    return get_response()