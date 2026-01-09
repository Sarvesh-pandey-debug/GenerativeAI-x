from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)


model = ChatHuggingFace(llm=llm)

# First prompt -> detailed report
template1 = PromptTemplate(
    template = "Write a detailed report on the {topic}?",
    input_variables=["topic"]
)

# Second prompt -> summary
template2 = PromptTemplate(
    template = "Write 5 line summary of on the following  {text}?",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic": "climate change"})

result = model.invoke(prompt1)
if result is None or result.content is None:
    raise Exception("Model response is None. Possibly rate limited or API failure.")

prompt2 = template2.invoke({"text": result.content})
result1 = model.invoke(prompt2)

if result1 is None or result1.content is None:
    raise Exception("Model response is None. Possibly rate limited or API failure.")

print(result1.content)






# This code is updated to use Groq (free) instead of OpenAI's paid API or huggingface's API


# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# import os


# load_dotenv()

# llm = ChatGroq(
#     groq_api_key=os.getenv("GROQ_API_KEY"),
#     model_name="llama3-8b-8192", 
#     temperature=0.7,
#     max_tokens=512, 
# )

# # First prompt -> detailed report
# template1 = PromptTemplate(
#     template="Write a detailed report on the {topic}?",
#     input_variables=["topic"]
# )

# # Second prompt -> summary
# template2 = PromptTemplate(
#     template="Write 5 line summary of the following: {text}?",
#     input_variables=["text"]
# )

# # Invoke first prompt
# prompt1 = template1.invoke({"topic": "climate change"})
# result = llm.invoke(prompt1)

# if result is None or not hasattr(result, "content") or result.content is None:
#     raise Exception("Model response is None. Possibly rate limited or API failure.")

# # Invoke second prompt for summary
# prompt2 = template2.invoke({"text": result.content})
# result1 = llm.invoke(prompt2)

# if result1 is None or not hasattr(result1, "content") or result1.content is None:
#     raise Exception("Model response is None. Possibly rate limited or API failure.")

# print(result1.content)



