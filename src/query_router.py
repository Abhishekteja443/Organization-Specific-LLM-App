def route_query(query: str) -> str:
    q = query.lower()

    if any(x in q for x in ["hi", "hello", "thanks", "bye"]):
        return "chitchat"

    if any(x in q for x in ["where is", "full form", "what is", "who is"]):
        return "fact"

    if any(x in q for x in ["admission", "fees", "courses", "contact", "eligibility"]):
        return "rag"
    
    if any(x in q for x in [
        "what context do you have",
        "what data do you have",
        "what are you trained on",
        "what do you know about me"
    ]):
        return "meta"

    return "rag"  # default
