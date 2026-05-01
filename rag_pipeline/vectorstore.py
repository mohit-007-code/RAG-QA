import os
import shutil
from langchain.vectorstores import Chroma
def create_vectorstore(chunks, embedding, file_name):

    persist_dir = f"db/{file_name}"

    if os.path.exists(persist_dir):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding
        )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir
    )

    db.persist()
    return db