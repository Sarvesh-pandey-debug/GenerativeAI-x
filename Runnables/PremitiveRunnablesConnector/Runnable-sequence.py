from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parser import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence
load_dotenv()

prompt = PromptTemplate(
    template = "write a joke {topic}",
    input_variables = ["topic"]
)


model = ChatOpenAI()

parser = StrOutputParser()

chain = RunnableSequence(prompt , model , parser)

print(chain.invoke({"topic": "cats"}))




