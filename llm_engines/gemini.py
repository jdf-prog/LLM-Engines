import os
from typing import List
import google.ai.generativelanguage as glm
import google.generativeai as genai
from .utils import with_timeout
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

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
    model = genai.GenerativeModel(model_name)
    
    new_messages = []
    if conv_system_msg:
        new_messages.append({"role": "system", "parts": [glm.Part(text=conv_system_msg)]})
    for i, message in enumerate(messages):
        new_messages.append({"role": "user" if i % 2 == 0 else "model", "parts": [glm.Part(text=message)]})
    
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
    @with_timeout(timeout)
    def generate_content():
        return model.generate_content(new_messages, safety_settings=safety_settings, generation_config=generation_config, request_options=request_options)
    response = generate_content()
    return response.text

if __name__ == "__main__":
    from icecream import ic
    ic(call_worker_gemini(["Hello", "Hi, I am gemini", "What did I ask in the last response?"], "gemini-1.5-flash"))