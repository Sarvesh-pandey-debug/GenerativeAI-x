from langchain.prompts import PromptTemplate

# PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],template_validation=True,
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:

Explanation Style: {style_input}
Explanation Length: {length_input}

1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain concepts using simple, intuitive code snippets where applicable.

2. Analogies:
   - Use relatable analogies to simplify complex ideas.
   - If certain information is not available, respond with: "Insufficient information available".

Ensure the summary is clear, accurate, and aligned with the provided style and length.
"""
)
prompt_template.save("prompt_template.json")
