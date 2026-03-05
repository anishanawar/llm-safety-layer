from sentence_transformers import SentenceTransformer
import numpy as np


class SimpleRetriever:
    def __init__(self, kb_path="data/kb.txt"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        with open(kb_path, "r") as f:
            self.docs = [line.strip() for line in f.readlines() if line.strip()]

        self.embeddings = self.model.encode(self.docs, normalize_embeddings=True)

    def retrieve(self, query, top_k=2):
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        scores = np.dot(self.embeddings, query_embedding.T).squeeze()

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.docs[i] for i in top_indices]