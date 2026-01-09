from xml.parsers.expat import model
from prompt_toolkit import prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
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


schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic "),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic "),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic ")

]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="give me 3 facts about the {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

# prompt_text = template.invoke({"topic": "climate change"})
# model_response = chat_model.invoke(prompt_text)
# response = parser.parse(model_response.content)
# print(response)

chain = template | chat_model | parser
result = chain.invoke({"topic": "climate change"})
print(result)


