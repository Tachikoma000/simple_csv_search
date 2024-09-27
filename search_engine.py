# search_engine.py

import faiss
import numpy as np
from embedder import get_embeddings
from config import INDEX_FILE_PATH, TOP_K

class SearchEngine:
    def __init__(self, embeddings):
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.embeddings = embeddings
        self.build_index()

    def build_index(self):
        """
        Add embeddings to the FAISS index.
        """
        self.index.add(self.embeddings)

    def save_index(self, file_path=INDEX_FILE_PATH):
        """
        Save the FAISS index to a file.
        """
        faiss.write_index(self.index, file_path)

    def load_index(self, file_path=INDEX_FILE_PATH):
        """
        Load the FAISS index from a file.
        """
        self.index = faiss.read_index(file_path)

    def search(self, query_embedding, top_k=TOP_K):
        """
        Perform a search for the query embedding.
        """
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return distances[0], indices[0]
