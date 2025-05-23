import os
import pandas as pd
import numpy as np
import json
import ast
import tiktoken
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
from typing import List
import base64
from pydantic import BaseModel


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

def analyze_image(image_base64, subcategories):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": """Given an image of an item of clothing, analyze the item and generate a JSON output with the following fields: "items", "category", and "gender". 
                           Use your understanding of fashion trends, styles, and gender preferences to provide accurate and relevant suggestions for how to complete the outfit.
                           The items field should be a list of items that would go well with the item in the picture. Each item should represent a title of an item of clothing that contains the style, color, and gender of the item.
                           The category needs to be chosen between the types in this list: {subcategories}.
                           You have to choose between the genders in this list: [Men, Women, Boys, Girls, Unisex]
                           Do not include the description of the item in the picture. Do not include the ```json ``` tag in the output.
                           
                           Example Input: An image representing a black leather jacket.

                           Example Output: {"items": ["Fitted White Women's T-shirt", "White Canvas Sneakers", "Women's Black Skinny Jeans"], "category": "Jackets", "gender": "Women"}
                           """,
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                },
                }
            ],
            }
        ]
    )
    # Extract relevant features from the response
    features = response.choices[0].message.content
    return features

## Implement Guardrails for the output of the matching outfits, using Responses API and Structured Output Parsing
class GuardrailMatchResponse(BaseModel):
    answer: str
    reason: str

def check_match(reference_image_base64, suggested_image_base64):
    response = client.responses.parse(
        model=GPT_MODEL,
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        """You will be given two images of two different items of clothing. 
                        Your goal is to decide if the items in the images would work in an outfit together. 
                        The first image is the reference item (the item that the user is trying to match with another item). 
                        You need to decide if the second item would work well with the reference item. 
                        The "answer" field must be either "yes" or "no", depending on whether you think the
                        items would work well together. The "reason" field must be a short explanation of your reasoning 
                        for your decision. Do not include the descriptions of the 2 images."""
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{reference_image_base64}",
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{suggested_image_base64}",
                },
            ]
        }],
        text_format=GuardrailMatchResponse,
        max_output_tokens=300,
    )

    features = response.output_parsed
    return features

def generate_combined_outfit(reference_image, paths, style_index=0):
    image_styles = [
        "High-Fashion Runway Illustration",
        "Retro 80s/90s anime",
    ]
    image_style = image_styles[style_index]
    prompt = f"""
    Keeping the original model outfit, add in new clothing accessories from the other images.\n
    The resulting look would be a combined outfit showing off combined items as a new look on one model.\n
    The new outfit must be a combination of the original outfit and the new items,\n 
    if there are more than one of the same items, just pick one, do not modify the item and keep its original design.\n
    Return the image in {image_style} style, with a dark background.\n
    Always include the full outfit look including shoes. The shoes must match and be the same as the reference images.
    """
    img = client.images.edit(
        image=[open(reference_image, "rb")] + [open(p, "rb") for p in paths],
        prompt=prompt,
        model="gpt-image-1",
        n=1,
        size="1024x1536",
        quality="medium",
        background="auto",
    )
    return img

# def main():
#     styles_filepath = "data/sample_clothes/sample_styles.csv"
#     output_filepath = "data/sample_clothes/sample_styles_with_embeddings.csv"
#     if not os.path.exists(styles_filepath):
#         print(f"File {styles_filepath} not found.")
#         return
#     styles_df = pd.read_csv(styles_filepath, on_bad_lines='skip')
#     print(styles_df.head())
#     print(f"Opened dataset successfully. Dataset has {len(styles_df)} items of clothing.")
#     styles_df = generate_embeddings(styles_df, 'productDisplayName')
#     styles_df.to_csv(output_filepath, index=False)
#     print(f"Embeddings successfully stored in {output_filepath}")

# if __name__ == "__main__":
#     main();
