import os
from mistralai import Mistral
from app.memory.chroma_memory import ChromaMemory
from dotenv import load_dotenv



load_dotenv()
print("API KEY:", os.getenv("MISTRAL_API_KEY"))  # Debug
 
class MistralService:
    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY no está configurada")
        
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-small-latest"  # Puedes cambiar el modelo
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

        response = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )

        answer = response.choices[0].message.content

        # Save to memory
        self.memory.add(user_id, query, answer)

        return answer

    async def generate_answer_stream(self, user_id: str, query: str):
        """Versión con streaming para respuestas en tiempo real"""
        memory_docs = self.memory.search(user_id, query)
        context_text = "\n".join(memory_docs)

        prompt = f"""
        You are a helpful AI assistant.
        User memory:
        {context_text}

        User query: {query}
        """

        full_response = ""
        
        stream = self.client.chat.stream(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        for chunk in stream:
            if chunk.data.choices[0].delta.content:
                content = chunk.data.choices[0].delta.content
                full_response += content
                yield content

        # Save to memory after stream completes
        self.memory.add(user_id, query, full_response)