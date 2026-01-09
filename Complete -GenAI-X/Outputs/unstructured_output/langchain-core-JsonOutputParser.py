
# # this code using huggingface's free API 

from xml.parsers.expat import model
from prompt_toolkit import prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

# Setup HuggingFaceEndpoint with conversational task
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",  
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

# Wrap in ChatHuggingFace to enable chat-like behavior
chat_model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="give me the name , age and city of the fictional persion \n {format_instruction}",
    input_variables=[],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }

)   


chain = template | chat_model | parser

prompt_text = chain.invoke({})

print(prompt_text)



