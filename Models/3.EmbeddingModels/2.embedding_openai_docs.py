from langchain_openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()


embedding_model = OpenAIEmbedding(model="text-embedding-3-larg", dimensions=32)

document = [
    "delhi is the capital of india",
    "paris is the capital of france",
    "berlin is the capital of germany",
    "tokyo is the capital of japan",
]


result = embedding_model.embed_documents(document)
print(str(result))

