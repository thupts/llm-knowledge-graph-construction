import os
from dotenv import load_dotenv
load_dotenv()

import openai

if os.getenv("AZURE_OPENAI_API_KEY"):
    openai.api_type = "azure"
    openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    deployment_name = "gpt-3.5-turbo"  # or your default

def chat_with_llm(prompt):
    response = openai.ChatCompletion.create(
        engine=deployment_name,  # 'engine' for Azure, 'model' for OpenAI
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
