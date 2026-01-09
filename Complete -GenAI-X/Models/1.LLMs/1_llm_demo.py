import os
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai = OpenAI(model="gpt-3.5-turbo", temperature=0.9)

result = openai.invoke("What is the capital of France?")
print(result)
