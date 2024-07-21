from setuptools import setup, find_packages

setup(
    name='llm-engines',
    version='0.0.1',
    description='',
    author='Dongfu Jiang',
    author_email='dongfu.jiang@uwaterloo.ca',
    packages=find_packages(),
    url='https://github.com/jdf-progLLM-Engines',
    install_requires=[
        "fire",
        "openai",
        "google-generativeai",
        "accelerate",
        "transformers>=4.39.0",
        "torch",
        "Pillow",
        "torch",
        "tqdm",
        "numpy",
        "requests",
        "sentencepiece",
        "vllm==0.5.1", # 0.5.2 does not work, 0.5.1 is the latest version that works, can be updated if newer version is released and it works
        "together",
        "icecream",
        "prettytable",
        "sglang[all]",
        "mistralai",
        "anthropic",
        "flash-attn"
    ],
)
