def get_retriever(db):
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}   
    )