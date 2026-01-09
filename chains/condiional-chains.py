# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser 
# from langchain.schemas.rules import Runableparallel


# load_dotenv()

# model1 = ChatOpenAI()


# parser = StrOutputParser()

# prompt1 = PromptTemplate(
#     template="classify the sentiment of the following feedback  text into positive, negative, or neutral \n {feedback}?",
#     input_variables=["feedback"]
# )

# classifier_chain = prompt1 | model1 | parser

# result = classifier_chain.invoke({
#     "feedback": "I love using this product! It's amazing."
# })


# print("--- Sentiment Classification ---")
# print(result)


from numpy import positive
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

# Define the model from Hugging Face
model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.1,  
        max_new_tokens=128, 
    )
)

# Define your desired data structure using Pydantic
class FeedbackModel(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment of the feedback")

# Set up a parser
pydantic_parser = PydanticOutputParser(pydantic_object=FeedbackModel)

# Define the prompt template for sentiment classification
prompt = PromptTemplate(
    template="Classify the sentiment of the following feedback.\n{format_instructions}\nFeedback: {feedback}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
)

# Create the chain by linking the prompt, model, and parser
classifier_chain = prompt | model | pydantic_parser

prompt1 = PromptTemplate(
    template="write an apropriate responce for the negative feedback :\n{feedback}",
    input_variables=["feedback"]
)

prompt2 = PromptTemplate(
    template="write an apropriate responce for the positive feedback :\n{feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="write an apropriate responce for the neutral feedback :\n{feedback}",
    input_variables=["feedback"]
)   


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'negative', prompt1 | model),
    (lambda x: x.sentiment == 'positive', prompt2 | model),
    (lambda x: x.sentiment == 'neutral', prompt3 | model),
    RunnableLambda(lambda x: "could not classify the feedback")
)


chain = classifier_chain | branch_chain

result = chain.invoke({
    "feedback": "this product is not good but its ok."
})

print("--- Sentiment Classification ---")
print(result.content)

chain.get_graph().print_ascii()