from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a detailed report on the {topic}?",
    input_variables=["topic"]
)


prompt2 = PromptTemplate(
    template="generate the 5 pointer Summary of  the following \n {text} in one paragraph.",
    input_variables=["text"]
)



model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.7,
        max_new_tokens=1024,
    )
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

# Generate a report on cricket
print("--- Generating report ---")
result = chain.invoke({"topic": "the history of cricket"})
print(result)

chain.get_graph().print_ascii()