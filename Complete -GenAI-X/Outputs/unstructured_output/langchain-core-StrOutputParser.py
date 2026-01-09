# # this code is using groq free API instead of OpenAI's paid API and in my case  
# # huggingface's API is not valid this this that why i am use groq API

# # this code also can use with huggingface free api 

# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain_core.output_parsers import StrOutputParser
# import os

# load_dotenv()

# llm = ChatGroq(
#     groq_api_key=os.getenv("GROQ_API_KEY"),
#     model_name="llama3-8b-8192", 
#     temperature=0.7,
#     max_tokens=512, 
# )

# # Prompt templates
# template1 = PromptTemplate(
#     template="Write a detailed report on the {topic}?",
#     input_variables=["topic"]
# )

# template2 = PromptTemplate(
#     template="Write 5 line summary of the following: {text}?",
#     input_variables=["text"]
# )

# parser = StrOutputParser()

# # Build chain using LangChain Expression Language
# report_chain = template1 | llm | parser |template2 | llm | parser

# report = report_chain.invoke({"topic": "climate change"})


# print(report)




# # this code using huggingface's free API 

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# Setup HuggingFaceEndpoint with conversational task
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="conversational",  
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

# Wrap in ChatHuggingFace to enable chat-like behavior
chat_model = ChatHuggingFace(llm=llm)

# Define prompts
template1 = PromptTemplate(
    template="Write a detailed report on the {topic}?",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write 5 line summary of the following: {text}?",
    input_variables=["text"]
)

parser = StrOutputParser()

# Use the chat_model instead of raw llm in the chain
report_chain = template1 | chat_model | parser | template2 | chat_model | parser

# Run the chains
report = report_chain.invoke({"topic": "climate change"})

print(report)

