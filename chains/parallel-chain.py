# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser 
# from langchain.schemas.rules import Runableparallel

 
# load_dotenv()

# model1 = ChatAnthropic(model_name="claude-2")
# model2 = ChatOpenAI()


# prompt1 = PromptTemplate(
#     template="Generate short and simple note following text \n {text}?",
#     input_variables=["text"]
# )   

# prompt2 = PromptTemplate(
#     template="Generate 5 questions based on the following text \n {text} ",
#     input_variables=["text"]
# )

# prompt3 = PromptTemplate(
#     template="Merge the provided notes and quiz into  in a single document \n Notes: {notes} \n Quiz: {quiz}",
#     input_variables=["notes", "quiz"]
# )

# parser = StrOutputParser()

# parallel_chain = Runableparallel({
#     "notes": prompt1 | model1 | parser,
#     "quiz": prompt2 | model2 | parser
# })

# merge_chain = prompt3 | model1 | parser

# chain = parallel_chain | merge_chain


# text = """The history of cricket dates back to the 16th century.
# It originated in England and has since become a popular sport worldwide. The game has evolved significantly over the years, with changes in rules, formats, and playing styles."""

# result = chain.invoke({
#     "text": text
# })

# print("--- Generating report ---")
# print(result)













from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Load environment variables from .env file
# Make sure you have a HUGGINGFACEHUB_API_TOKEN in your .env file
load_dotenv()

# Define the model from Hugging Face
# We'll use the same Mixtral model for all tasks
model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.7,
        max_new_tokens=1024,
    )
)

# Define the output parser
parser = StrOutputParser()

# Prompt to generate a short note from the text
prompt1 = PromptTemplate(
    template="Generate a short and simple note from the following text:\n{text}",
    input_variables=["text"]
)

# Prompt to generate 5 questions from the text
prompt2 = PromptTemplate(
    template="Generate 5 questions based on the following text:\n{text}",
    input_variables=["text"]
)

# Prompt to merge the notes and quiz into a single document
prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single, well-formatted document.\n\nNotes:\n{notes}\n\nQuiz:\n{quiz}",
    input_variables=["notes", "quiz"]
)

# Define a parallel chain that runs two chains at once:
# 1. "notes": Creates notes from the text
# 2. "quiz": Creates a quiz from the text
parallel_chain = RunnableParallel(
    notes=prompt1 | model | parser,
    quiz=prompt2 | model | parser,
)

# Define a final chain that takes the output of the parallel chains
# and merges them using the third prompt
final_chain = parallel_chain | prompt3 | model | parser


# The input text for the chain
text = """The history of cricket dates back to the 16th century.
It originated in England and has since become a popular sport worldwide. The game has evolved significantly over the years, with changes in rules, formats, and playing styles."""

# Invoke the final chain with the input text
result = final_chain.invoke({
    "text": text
})

# Print the final merged result
print("--- Generated Notes and Quiz ---")
print(result)


final_chain.get_graph().print_ascii()