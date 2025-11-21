from uuid import uuid4
import chromadb
import ollama

class ChromaMemory:
    def __init__(self):
        self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name="chat_memory",
            metadata={"hnsw:space": "cosine"}
        )

    def create_embedding(self, text: str):
        emb = ollama.embeddings(model="llama3", prompt=text)
        return emb["embedding"]

    def add(self, user_id: str, query: str, answer: str):
        document = f"Q: {query}\nA: {answer}"
        embedding = self.create_embedding(document)

        self.collection.add(
            ids=[f"{user_id}-{uuid4()}"],
            documents=[document],
            metadatas=[{"user_id": user_id}],
            embeddings=[embedding]
        )

    def search(self, user_id: str, query: str):
        query_emb = self.create_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=5,
            where={"user_id": user_id}
        )

        docs = results.get("documents", [[]])[0]
        return docs
