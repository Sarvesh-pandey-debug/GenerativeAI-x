# from langchan_openai import ChatOpenAI
# from dotenv import load_dotenv
# import streamlit as st
# import os

# # Load environment variables
# load_dotenv()   

# llm = 

# while True:
#     user_input = input("user_input":)
#     if user_input == "exit":
#         break
#     model.incoke(user_input)




from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize HuggingFace LLM (chat-compatible model)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    huggingfacehub_api_token=hf_token,
    temperature=0.5,
    max_new_tokens=256
)

# Wrap with Chat interface
chat_model = ChatHuggingFace(llm=llm)

# CLI Chat loop
print("Hugging Face Chat (type 'exit' to quit)")

#  Initialize chat_history with SystemMessage
chat_history = [SystemMessage(content="You are a helpful assistant.")]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))

    response = chat_model.invoke(chat_history)

    chat_history.append(AIMessage(content=response.content))

    print("AI:", response.content.strip())

print("Chat History:", chat_history)
