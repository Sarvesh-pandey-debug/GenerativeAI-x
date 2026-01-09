from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

import os
print("TOKEN:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

llm = HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task="text-generation",
    temperature=0.5
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of india?")
print(result.content)

