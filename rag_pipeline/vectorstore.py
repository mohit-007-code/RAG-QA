import os
import shutil
from langchain_chroma import Chroma


def create_vectorstore(chunks, embedding, file_name):

    persist_dir = f"db/{file_name}"

    # ✅ Case 1: DB already exists → load it (FAST)
    if os.path.exists(persist_dir):
        try:
            db = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding
            )
            print("⚡ Loaded existing vector DB (no reprocessing)")
            return db

        except Exception as e:
            print(f"⚠️ Corrupted DB detected: {e}")
            print("🧹 Deleting and rebuilding DB...")
            shutil.rmtree(persist_dir)

    # ✅ Case 2: Create new DB
    print("⏳ Creating new vector DB (first time only)")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir
    )

    db.persist()

    print("✅ Vector DB created & saved")

    return db