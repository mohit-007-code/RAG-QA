from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert AI assistant for programming and technical questions.

STRICT RULES:
1. Use ONLY the provided context
2. If answer not found → say exactly: I don't know

3. If question is coding-related:
   - First give a short explanation
   - THEN provide code in a proper code block
   - ALWAYS use triple backticks like this:

```python
# your code here
     
4. Keep answers structured:
   - Use bullet points
   - Avoid long paragraphs
"""),

    ("human", """
Context:
{context}

Question:
{question}

Answer:
""")
])