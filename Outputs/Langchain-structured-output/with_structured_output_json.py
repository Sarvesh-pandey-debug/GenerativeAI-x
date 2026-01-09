# this code is for openai paid api 


from langchain_openai import  chatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Optional
from pydantic import BaseModel, Field , EmailStr
load_dotenv()

model = chatOpenAI()
# schema 

json_schema = {
    "title": "Langchain Structured Output",
    "type": "object",
    "properties": {
        "name": {
        "type": "string",
        "description": "The name of the person"
        },
        "sentiment": {
        "type": "string",
        "enum": ["negative", "positive", "neutral"],
        "description": "The sentiment score of the review"
        },
        "summary": {
        "type": "string",
        "description": "A summary of the review"
        },
        "pros": {
        "type": ["array", "nullable"],
        "items": {
            "type": "string"
        },
        "description": "A list of pros for the app"
        
    },
        "cons": {
        "type": ["array", "null"],
        "items": {
            "type": "string"
        },  
        "description": "A list of cons for the app"
        }
    },
    "required": ["name", "sentiment", "summary"]
}



#frunction calling
structure_model = model.with_structured_output(json_schema)


result = structure_model.invoke("I’ve been using this app for a couple of weeks, and overall it has been a great experience. The interface is clean and easy to use, and most of the core features work flawlessly. However, there are occasional lags when switching between sections, and I’d love to see more customization options. Still, it’s definitely worth trying out!")


print(result)








from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
from huggingface_hub import login
import os
import json
import re

load_dotenv()

# Login to Hugging Face
login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Initialize HF model
model = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
)

# JSON schema
json_schema = {
    "title": "Langchain Structured Output",
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "The name of the person"},
        "sentiment": {
            "type": "string",
            "enum": ["negative", "positive", "neutral"],
            "description": "The sentiment score of the review"
        },
        "summary": {"type": "string", "description": "A summary of the review"},
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "A list of pros for the app"
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "A list of cons for the app"
        }
    },
    "required": ["name", "sentiment", "summary"]
}

# Review and instruction
review = """
I’ve been using this app for a couple of weeks, and overall it has been a great experience.
The interface is clean and easy to use, and most of the core features work flawlessly.
However, there are occasional lags when switching between sections, and I’d love to see more customization options.
Still, it’s definitely worth trying out!
"""

instruction = f"""
Extract the following fields from the review as a JSON object matching this schema:

{json.dumps(json_schema, indent=2)}

Only return a valid JSON object. Do not include any explanation or extra text.
"""

# Combine prompt
full_prompt = review + "\n\n" + instruction

# Invoke model
response = model.invoke(full_prompt)

# Try parsing JSON from response
try:
    json_text = re.search(r"\{.*\}", response, re.DOTALL).group()
    parsed = json.loads(json_text)
    print(parsed)
except Exception as e:
    print("Failed to parse JSON:", e)
    print("Raw response:", response)





