# # this code is for openai paid api 


# from langchain_openai import  chatOpenAI
# from dotenv import load_dotenv
# from typing import TypedDict, Annotated,Optional
# from pydantic import BaseModel, Field , EmailStr
# load_dotenv()

# model = chatOpenAI()
# # schema 

# class Review(TypedDict):
#     summary: str = Field(..., description="A brief summary of the review")
#     sentiment: Literal["negative", "positive", "neutral"] = Field(..., description="A brief sentiment of the review, either negative, positive, or neutral")
#     pros: Optional[list[str]] = Field(default=None, description="Write down all pros in the list")
#     cons: Optional[list[str]] = Field(default=None, description="Write down all cons in the list")
#     name: Optional[str] = Field(default=None, description="Name of the person giving the review")

# #frunction calling
# structure_model = model.with_structured_output(Review)


# result = structure_model.invoke("I’ve been using this app for a couple of weeks, and overall it has been a great experience. The interface is clean and easy to use, and most of the core features work flawlessly. However, there are occasional lags when switching between sections, and I’d love to see more customization options. Still, it’s definitely worth trying out!")


# print(result)






# This code is updated to use Hugging Face (free) instead of OpenAI's paid API

# from langchain_community.chat_models import ChatHuggingFace
# from langchain_core.output_parsers import with_structured_output
# from dotenv import load_dotenv
# from typing import Optional, Literal
# from pydantic import BaseModel, Field
# import os

# # Load token
# load_dotenv()
# token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# # Define your output model
# class Review(BaseModel):
#     summary: str = Field(..., description="A brief summary of the review")
#     sentiment: Literal["negative", "positive", "neutral"]
#     pros: Optional[list[str]]
#     cons: Optional[list[str]]
#     name: Optional[str]

# # Wrap model with parser
# parser = with_structured_output(Review)

# # Use ChatHuggingFace (not HuggingFaceEndpoint!)
# llm = ChatHuggingFace(
#     repo_id="google/flan-t5-large",  # works!
#     huggingfacehub_api_token=token,
#     task="text2text-generation",
#     max_new_tokens=512,
# )

# # Chain LLM + parser
# chain = parser | llm

# # Test input
# input_text = """I've been using this app for a couple of weeks. It's user-friendly, fast, and gets frequent updates. Sometimes crashes though."""

# response = chain.invoke({
#     "input": f"""Extract the following from this review:
# - summary
# - sentiment (positive, neutral, negative)
# - pros (as list)
# - cons (as list)
# - name

# Review: "{input_text}"

# Respond ONLY with JSON."""
# })

# # Parsed response (Review model)
# print(response)

from pydantic import BaseModel
from langchain_core.output_parsers import with_structured_output

class MyData(BaseModel):
    name: str
    age: int

parser = with_structured_output(MyData)
print(parser)
