from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful assistant.

Answer using ONLY the given context.

If answer not found, say:
I don't know

Keep answer clean and structured:
- Use short points
- Avoid long paragraphs
- If code is needed, write clean Python code
"""),

    ("human", """
Context:
{context}

Question:
{question}

Answer:
""")
])