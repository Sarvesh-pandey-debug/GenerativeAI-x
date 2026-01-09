# this code for openapi which is paid api 

# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# import streamlit as st 
# import os   
# # Load environment variables
# load_dotenv()       

# model = ChatOpenAI()

# st.header("Research Assistant with LangChain and OpenAI")
# user_input = st.text_input("Enter your query")
# if st.button("Submit"):
#     result = model.invoke(user_input)
#     st.write("### Response:", result.content)


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import streamlit as st
import os


load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    temperature=0.6,
    max_new_tokens=256
)

chat_model = ChatHuggingFace(llm=llm)


st.title("AI Assistant with Hugging Face")

user_input = st.text_input("Ask me something:")

if st.button("Submit"):
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content=user_input)
        ]
        response = chat_model.invoke(messages)
        st.write(response.content)
        
