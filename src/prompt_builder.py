def build_system_prompt(route: str) -> str:
    base = """
You are an AI assistant for an organization.
Answer naturally, clearly, and professionally.
Do NOT mention internal tools, retrieval, FAISS, or reasoning.
"""

    if route == "chitchat":
        return base + "\nBe friendly and brief."

    if route == "fact":
        return base + "\nGive a short, direct factual answer."

    if route == "rag":
        return base + "\nUse the provided context silently if useful."

    return base