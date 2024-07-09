import os
from typing import List
import google.ai.generativelanguage as glm
import google.generativeai as genai
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
def call_worker_gemini(messages:List[str], model_name, conv_system_msg=None, **generate_kwargs) -> str:
    # change messages to gemini format
    model = genai.GenerativeModel(model_name)
    
    new_messages = []
    if conv_system_msg:
        new_messages.append({"role": "system", "parts": [glm.Part(text=conv_system_msg)]})
    for i, message in enumerate(messages):
        new_messages.append({"role": "user" if i % 2 == 0 else "model", "parts": [glm.Part(text=message)]})
    assert new_messages[-1]["role"] == "user", "The last message must be from the user"
    
    response = model.generate_content(new_messages, safety_settings=safety_settings, **generate_kwargs)
    return response.text

if __name__ == "__main__":
    from icecream import ic
    ic(call_worker_gemini(["Hello", "Hi, I am gemini", "What did I ask in the last response?"], "gemini-1.5-flash"))