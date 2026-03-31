import faiss
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from config import OPENAI_API_KEY
 
client = OpenAI(api_key=OPENAI_API_KEY)
dimension = 1536
index = faiss.IndexFlatL2(dimension)

stored_chunks = []

def create_embeddings(chunks):

    embeddings = []

    for chunk in chunks:

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )

        vector = response.data[0].embedding

        embeddings.append(vector)
        stored_chunks.append(chunk)

    vectors = np.array(embeddings).astype("float32")

    index.add(vectors)

def search(query):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    query_vector = np.array([response.data[0].embedding]).astype("float32")

    D, I = index.search(query_vector, k=3)

    results = [stored_chunks[i] for i in I[0]]

    return results