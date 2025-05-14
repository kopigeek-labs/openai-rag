import os
import pandas as pd
import numpy as np
import json
import tiktoken
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
from typing import List
import base64

# Custom utility functions for embedding and similarity search
from .utils import batchify, cosine_similarity_manual, find_similar_items, encode_image_to_base64

# Load environment variables from .env file
load_dotenv()

client = OpenAI()
GPT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_COST_PER_1K_TOKENS = 0.00013

def get_embeddings(input: List[str]):
    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(10))
    def _get_embeddings(input):
        response = client.embeddings.create(
            input=input,
            model=EMBEDDING_MODEL
        ).data
        return [data.embedding for data in response]
    return _get_embeddings(input)


def embed_corpus(corpus: List[str], batch_size=64, num_workers=8, max_context_len=8191):
    encoding = tiktoken.get_encoding("cl100k_base")
    encoded_corpus = [encoded_article[:max_context_len] for encoded_article in encoding.encode_batch(corpus)]
    num_tokens = sum(len(article) for article in encoded_corpus)
    cost_to_embed_tokens = num_tokens / 1000 * EMBEDDING_COST_PER_1K_TOKENS
    print(f"num_articles={len(encoded_corpus)}, num_tokens={num_tokens}, est_embedding_cost={cost_to_embed_tokens:.2f} USD")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(get_embeddings, text_batch) for text_batch in batchify(corpus, batch_size)]
        with tqdm(total=len(encoded_corpus)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(batch_size)
        embeddings = []
        for future in futures:
            data = future.result()
            embeddings.extend(data)
        return embeddings

def generate_embeddings(df, column_name):
    descriptions = df[column_name].astype(str).tolist()
    embeddings = embed_corpus(descriptions)
    df['embeddings'] = embeddings
    print("Embeddings created successfully.")
    return df

def main():
    styles_filepath = "data/sample_clothes/sample_styles.csv"
    output_filepath = "data/sample_clothes/sample_styles_with_embeddings.csv"
    if not os.path.exists(styles_filepath):
        print(f"File {styles_filepath} not found.")
        return
    styles_df = pd.read_csv(styles_filepath, on_bad_lines='skip')
    print(styles_df.head())
    print(f"Opened dataset successfully. Dataset has {len(styles_df)} items of clothing.")
    styles_df = generate_embeddings(styles_df, 'productDisplayName')
    styles_df.to_csv(output_filepath, index=False)
    print(f"Embeddings successfully stored in {output_filepath}")

if __name__ == "__main__":
    main()