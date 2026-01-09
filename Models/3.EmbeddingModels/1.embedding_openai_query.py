from langchain_openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()


embedding_model = OpenAIEmbedding(model="text-embedding-3-larg", dimensions=32)

result = embedding_model.embed_query("What is the capital of France?")
print(str(result))

