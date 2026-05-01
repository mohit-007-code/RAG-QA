from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from rag_pipeline.prompt import prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def build_qa_chain(retriever):
    llm = Ollama(
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