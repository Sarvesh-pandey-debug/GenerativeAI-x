# this code is for openai paid api 


from langchain_openai import  chatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Optional
load_dotenv()

model = chatOpenAI()
# schema 

class Review(TypedDict):
    
    summary:Annotated[str, "A breif summry of the review "] 
    sentiment: Annotated[str, "A breif sentiment of thr review eithr negative , possitive, neutral"]
    pros: Annotated[Optional[list[str]], "Write down all prons in the list"]
    cons: Annotated[Optional[list[str]], "Write down all cons in the list"]
      
#frunction calling 
structure_model = model.with_structured_output(Review)


result = structure_model.invoke("I’ve been using this app for a couple of weeks, and overall it has been a great experience. The interface is clean and easy to use, and most of the core features work flawlessly. However, there are occasional lags when switching between sections, and I’d love to see more customization options. Still, it’s definitely worth trying out!")


print(result)













# # this code is not work but in in this code use huggingface free api 


# from langchain_huggingface import ChatHuggingFace
# from dotenv import load_dotenv
# from typing import TypedDict
# load_dotenv()

# llm = ChatHuggingFace(
#     repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task="text-generation",
#     temperature=0.5
# )

# class Review(TypedDict):
    
#     summary:Annotated[str, "A breif summry of the review "] 
#     sentiment: Annotated[str, "A breif sentiment of thr review eithr negative , possitive, neutral"]
#     pros: Annotated[Optional[list[str]], "Write down all prons in the list"]
#     cons: Annotated[Optional[list[str]], "Write down all cons in the list"]
      
# #frunction calling 
# structure_model = llm.with_structured_output(Review)


# result = structure_model.invoke("I’ve been using this app for a couple of weeks, and overall it has been a great experience. The interface is clean and easy to use, and most of the core features work flawlessly. However, there are occasional lags when switching between sections, and I’d love to see more customization options. Still, it’s definitely worth trying out!")

# print("Summary:", result.summary)

