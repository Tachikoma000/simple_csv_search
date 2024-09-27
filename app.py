# app.py

from data_processor import load_data
from embedder import get_embeddings
from search_engine import SearchEngine
from config import TOP_K
import openai

def main():
    # Load data
    texts = load_data()

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = get_embeddings(texts)

    # Initialize search engine
    search_engine = SearchEngine(embeddings)

    # Save FAISS index (Optional)
    # search_engine.save_index()

    # Enter the search loop
    while True:
        query = input("Enter your search query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        # Get embedding for the query
        query_embedding = get_embeddings([query])[0]

        # Perform search
        distances, indices = search_engine.search(query_embedding)

        # Display results
        print("\nTop Results:")
        for idx, distance in zip(indices, distances):
            print(f"Score: {distance:.4f}")
            print(f"Text: {texts[idx]}\n")

if __name__ == '__main__':
    main()
