def get_retriever(db):
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20}   
    )