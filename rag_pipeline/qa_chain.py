from langchain_ollama import ChatOllama
from rag_pipeline.prompt import prompt
from langchain_core.output_parsers import StrOutputParser


def build_chain(retriever):
    llm = ChatOllama(
        model="mistral"
    )   

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {
            "context": lambda x: format_docs(
                retriever.invoke(x["question"])
            ),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain