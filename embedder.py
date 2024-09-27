# embedder.py

import openai
import openai.error
from config import OPENAI_API_KEY, EMBEDDING_MODEL
import numpy as np
import time

# Set your OpenAI API key
openai.api_key = OPENAI_API_KEY

def get_embedding(text, model=EMBEDDING_MODEL):
    """
    Get the embedding for a single text string.
    """
    text = text.replace("\n", " ")
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    embedding = response['data'][0]['embedding']
    return embedding

def get_embeddings(texts, model=EMBEDDING_MODEL, batch_size=1000):
    """
    Generate embeddings for a list of texts using the new OpenAI API.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        success = False
        while not success:
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model=model
                )
                batch_embeddings = [data_point['embedding'] for data_point in response['data']]
                embeddings.extend(batch_embeddings)
                success = True
            except RateLimitError:
                print("Rate limit exceeded. Waiting for 60 seconds.")
                time.sleep(60)
            except openai.error.APIError as e:
                print(f"API Error: {e}")
                time.sleep(10)
    return np.array(embeddings).astype('float32')
