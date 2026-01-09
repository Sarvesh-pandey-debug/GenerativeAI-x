import os
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv

load_dotenv()

Model = ChatOpenAI(model="gpt-4", temperature=0.9, max_completion_tokens=10)

result = Model.invoke("tell me a story on cricketc")

print(result)
print(result.content)


