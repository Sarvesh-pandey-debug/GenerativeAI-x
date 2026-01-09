from operator import gt
from xml.parsers.expat import model
from prompt_toolkit import prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
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


class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City where the person lives")
    
    
PydanticOutputParser = PydanticOutputParser(pydantic_object=Person)

PromptTemplate = PromptTemplate(
    template="give me the name , age and city of the fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={
        "format_instruction": PydanticOutputParser.get_format_instructions()
    }
)

# prompt_text = PromptTemplate.invoke({"place": "indian"})
# model_response = chat_model.invoke(prompt_text)
# result = PydanticOutputParser.parse(model_response.content)
# print(result)

chain = PromptTemplate | chat_model | PydanticOutputParser
result = chain.invoke({"place": "indian"})  
print(result)