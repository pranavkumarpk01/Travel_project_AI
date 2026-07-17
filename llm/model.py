import os

from dotenv import load_dotenv

load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").lower()
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")


def get_llm(temperature: float = 0.2):
    if MODEL_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=MODEL_NAME, base_url=base_url, temperature=temperature)

    if MODEL_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required when MODEL_PROVIDER=gemini")
        return ChatGoogleGenerativeAI(
            model=MODEL_NAME, google_api_key=api_key, temperature=temperature
        )

    raise ValueError(
        f"Unknown MODEL_PROVIDER='{MODEL_PROVIDER}'. Use 'ollama' or 'gemini'."
    )


llm = None


def get_shared_llm():
    global llm
    if llm is None:
        llm = get_llm()
    return llm
