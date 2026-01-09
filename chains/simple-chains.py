# This code demonstrates how to create a simple chain using LangChain with chatopenAi Paid api 

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv
# from langchain_core.output_parsers import StrOutputParser 
 
# load_dotenv()

# prompt = PromptTemplate(
#     template = "Write a detailed report on the {topic}?",
#     input_variables=["topic"]
# )

# model = ChatOpenAI()

# parser = StrOutputParser()

# chain = prompt | model | parser

# result = chain.invoke({"topic": "cricket"})
# print(result)



# This code demonstrates how to create a simple chain using LangChain with Hugging Face's Mixtral free API model

# First, make sure you have the necessary libraries installed:
# pip install langchain langchain-huggingface python-dotenv

from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser 

# Load environment variables from a .env file
load_dotenv()

# IMPORTANT: Make sure you have a .env file in the same directory
# with your Hugging Face API Token like this:
# HUGGINGFACEHUB_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Define the prompt template
prompt = PromptTemplate(
    template = "Write a detailed report on the {topic}?",
    input_variables=["topic"]
)

# 1. Initialize the Hugging Face Endpoint
# This object connects to the actual model on Hugging Face.
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.7,
    max_new_tokens=1024,
)

# 2. Initialize the ChatHuggingFace model
# This class acts as a wrapper around the endpoint to provide a chat interface.
# We pass the endpoint we created above as the `llm` argument.
model = ChatHuggingFace(llm=llm)

# Initialize the output parser
parser = StrOutputParser()

# Create the chain by piping the components together
chain = prompt | model | parser

# Invoke the chain with a topic
print("--- Generating report ---")
result = chain.invoke({"topic": "the history of cricket"})
print(result)


chain.get_graph().print_ascii()