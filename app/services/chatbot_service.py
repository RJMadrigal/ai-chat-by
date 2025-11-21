import ollama
from app.memory.chroma_memory import ChromaMemory

class ChatbotService:
    def __init__(self):
        self.model = "llama3"
        self.memory = ChromaMemory()

    async def generate_answer(self, user_id: str, query: str) -> str:
        # Retrieve relevant memory (RAG)
        memory_docs = self.memory.search(user_id, query)
        context_text = "\n".join(memory_docs)

        prompt = f"""
        You are a helpful AI assistant.
        User memory:
        {context_text}

        User query: {query}
        """

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"]

        # Save to memory
        self.memory.add(user_id, query, answer)

        return answer
