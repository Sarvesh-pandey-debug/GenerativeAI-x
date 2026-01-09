from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


text = "What is the capital of France?"

vector  = embedding_model.embed_query(text)
print(str(vector))