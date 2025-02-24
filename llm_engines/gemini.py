import os
import time
from typing import List
import google.ai.generativelanguage as glm
import google.generativeai as genai
from google.api_core.exceptions import ServiceUnavailable, ResourceExhausted
from .utils import with_timeout, decode_base64_image_url
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
# no image, multi-turn, do not use openai_generate, but can refer to it
def call_worker_gemini(messages:List[str], model_name, timeout:int=60, conv_system_msg=None, **generate_kwargs) -> str:
    # change messages to gemini format
    model = genai.GenerativeModel(model_name, system_instruction=conv_system_msg)
    
    new_messages = []
    role_map = {"user": "user", "assistant": "model"}
    for i, message in enumerate(messages):
        role = role_map[message["role"]]
        if isinstance(message["content"], str):
            new_messages.append({"role": role, "parts": [glm.Part(text=message["content"])]})
        elif isinstance(message["content"], list):
            parts = []
            for sub_message in message["content"]:
                if sub_message["type"] == "text":
                    parts.append(glm.Part(text=sub_message["text"]))
                elif sub_message["type"] == "image_url":
                    try:
                        image = decode_base64_image_url(sub_message["image_url"]['url'])
                    except Exception as e:
                        image = sub_message["image_url"]['url']
                    parts.append(image)
                else:
                    raise ValueError("Invalid message format")
            new_messages.append({"role": role, "parts": parts})
        else:
            raise ValueError("Invalid message format")
    
    stream = generate_kwargs.pop("stream", False)
    generation_config = genai.GenerationConfig(
        candidate_count=generate_kwargs.get("num_return_sequences", None),
        stop_sequences=generate_kwargs.get("stop", None),
        max_output_tokens=generate_kwargs.get("max_tokens", None),
        temperature=generate_kwargs.get("temperature", None),
        top_p=generate_kwargs.get("top_p", None),
        top_k=generate_kwargs.get("top_k", None),
        response_mime_type=generate_kwargs.get("response_mime_type", None),
        response_schema=generate_kwargs.get("response_schema", None),
    )
    request_options = genai.types.RequestOptions(
        timeout=timeout,
    )
    if "logprobs" in generate_kwargs:
        raise ValueError("logprobs is not supported in gemini")
    @with_timeout(timeout)
    def generate_content():
        return model.generate_content(new_messages, safety_settings=safety_settings, generation_config=generation_config, request_options=request_options, stream=stream)
    while True:
        try:
            response = generate_content()
            break
        except ServiceUnavailable as e:
            # sleep for a while and retry
            # print("ServiceUnavailable, retrying...")
            time.sleep(2)
            continue
        except ResourceExhausted as e:
            # sleep for a while and retry
            # print("ResourceExhausted, retrying...")
            time.sleep(10)
            continue
    try:
        if not stream:
            return response.text
        else:
            def generate_stream():
                for chunk in response:
                    yield chunk.text
            return generate_stream()
    except ValueError as e:
        print(f"Empty response from gemini due to {e}")
        return None

if __name__ == "__main__":
    from icecream import ic
    ic(call_worker_gemini(["Hello", "Hi, I am gemini", "What did I ask in the last response?"], "gemini-1.5-flash"))
