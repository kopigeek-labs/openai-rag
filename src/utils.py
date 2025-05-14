import numpy as np
import base64
from typing import List

def batchify(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]

def cosine_similarity_manual(vec1, vec2):
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def find_similar_items(input_embedding, embeddings, threshold=0.5, top_k=3):
    similarities = [(index, cosine_similarity_manual(input_embedding, vec)) for index, vec in enumerate(embeddings)]
    filtered_similarities = [(index, sim) for index, sim in similarities if sim >= threshold]
    sorted_indices = sorted(filtered_similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_indices

def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode('utf-8')