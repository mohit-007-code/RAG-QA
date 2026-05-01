import os
import shutil
from langchain.vectorstores import Chroma

def create_vectorstore(chunks, embedding):

    persist_dir = "db"

    # 🔥 critical fix
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir
    )

    db.persist()
    return db