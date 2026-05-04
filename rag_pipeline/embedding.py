from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 32, "normalize_embeddings": True}
    )
