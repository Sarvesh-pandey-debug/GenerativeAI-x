# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# import streamlit as st 
# from langchain_core.prompts import PromptTemplate

# import os

# load_dotenv()  


# model = ChatOpenAI()

# st.header("Research Assistant with LangChain and OpenAI")

# Paper_input = st.selectbox("Select Research Paper name ",['Attention Is All You Need',
# 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
# 'GPT-3: Language Models are Few-Shot Learners','Diffusion Models Beat GANs on Image Synthesis'])

# style_input = Explanation_style = st.selectbox("Select Explation style",['Beginner Friendly',
# 'Intermediate',
# 'Advanced',
# 'Technical Code oriented',
# 'Mathematical Explanation'])


# length_input = st.selectbox("Select Length of Explanation",['Short',
# 'Medium',
# 'Long',])                                                            



# template = PromptTemplate(
# template = """Please summarize the research paper titled [paper_input] with the following specifications:
# Explanation Style: [style_input]
# Explanation Length: [length_input]

# 1. Mathematical Details:
#    - Include relevant mathematical equations if present in the paper.
#    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

# 2. Analogies:
#    - Use relatable analogies to simplify complex ideas.
#    - If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
#    """,

# inputs=["paper_input", "style_input", "length_input"]

# )
# prompt_template = load_prompt("prompt_template.json")

# prompt = template.format({  
#     paper_input=Paper_input,
#     style_input=style_input,
#     length_input=length_input
# })



# if st.button("Submit"):
#     result = model.invoke(prompt)
#     st.write("Response:", result.content)




from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import streamlit as st
import os

# Load token from .env
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Model Configured with Hugging Face Endpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    huggingfacehub_api_token=hf_token,
    temperature=0.5,
    max_new_tokens=512
)

chat_model = ChatHuggingFace(llm=llm)


st.title("📚 AI Research Assistant — Hugging Face (Mistral 7B)")

# User input
paper_input = st.selectbox("Select Research Paper", [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Diffusion Models Beat GANs on Image Synthesis"
])

style_input = st.selectbox("Explanation Style", [
    "Beginner Friendly", "Intermediate", "Advanced",
    "Technical Code oriented", "Mathematical Explanation"
])

length_input = st.selectbox("Explanation Length", [
    "Short", "Medium", "Long"
])

# Load prompt template
prompt_template = load_prompt("prompt_template.json")

# Chain the prompt → chat model
chain = prompt_template | chat_model

# On Submit
if st.button("Submit"):
    response = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })
    st.write("Response:", response.content)
