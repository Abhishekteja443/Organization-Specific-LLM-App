def route_query(query: str) -> str:
    q = query.lower()

    if any(q.startswith(x) for x in ["hi", "hello", "thanks", "bye"]):
        print("Routing to chitchat")
        return "chitchat"

    # if any(x in q for x in ["where is", "full form", "what is", "who is"]):
    #     print("Routing to fact")
    #     return "fact"

    elif any(x in q for x in ["admission", "fees", "courses", "contact", "eligibility"]):
        print("Routing to rag")
        return "rag"
    
    elif any(x in q for x in [
        "what context do you have",
        "what data do you have",
        "what are you trained on",
        "what do you know about me"
    ]):
        print("Routing to meta")
        return "meta"
    else:
        print("Routing to default rag")
        return "rag"  # default
