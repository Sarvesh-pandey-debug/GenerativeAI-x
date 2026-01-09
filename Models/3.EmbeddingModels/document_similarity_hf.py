from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
load_dotenv()
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


document = [
    "delhi is the capital of india",
    "paris is the capital of france",
    "berlin is the capital of germany",
    "tokyo is the capital of japan",
]



query = "What is the capital of japan?"

# Embed the documents
document_embeddings = embedding_model.embed_documents(document)

# Embed the query
query_embedding = embedding_model.embed_query(query)

# Compute cosine similarities
similarities = cosine_similarity([query_embedding], document_embeddings)[0]


index, score = sorted(list(enumerate(similarities)), key=lambda x: x[1])[-1]

print(query)

print(document[index])

print("similarities score is:", score)