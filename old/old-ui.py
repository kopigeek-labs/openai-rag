import gradio as gr
import json, ast, os, tempfile
import pandas as pd
from PIL import Image as PILImage
from src.utils import encode_image_to_base64, find_similar_items
from src.main import analyze_image, get_embeddings

def get_unique_subcategories():
    styles_filepath = "data/sample_clothes/sample_styles.csv"
    if not os.path.exists(styles_filepath):
        return []
    styles_df = pd.read_csv(styles_filepath, on_bad_lines='skip')
    return styles_df['articleType'].unique().tolist()

SUBCATEGORIES = get_unique_subcategories()
# print(f"There are {len(subcategories)} unique subcategories, such as:")
# print(unique_subcategories)

def find_matching_items_with_rag(df_items, item_descs):
    embeddings = df_items['embeddings'].tolist()
    similar_items = []
    for desc in item_descs:
        input_embedding = get_embeddings([desc])
        similar_indices = find_similar_items(input_embedding, embeddings, threshold=0.6)
        similar_items += [df_items.iloc[i] for i in similar_indices]
    return similar_items

def recommend_items_from_image(image_np):
    if image_np is None:
        return "Please upload an image.", []
    # 1) save to temp file & encode
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        PILImage.fromarray(image_np).save(tmp.name)
        encoded_image = encode_image_to_base64(tmp.name)

    try:
        analysis = analyze_image(encoded_image, subcategories)
        image_analysis = json.loads(analysis)
        item_descs = image_analysis['items']
        item_category = image_analysis['category']
        item_gender = image_analysis['gender']

        styles_filepath = "data/sample_clothes/sample_styles_with_embeddings.csv"
        if not os.path.exists(styles_filepath):
            return "Styles file not found.", None
        styles_df = pd.read_csv(styles_filepath, on_bad_lines='skip')
        styles_df['embeddings'] = styles_df['embeddings'].apply(ast.literal_eval)
        filtered_items = styles_df.loc[styles_df['gender'].isin([item_gender, 'Unisex'])]
        filtered_items = filtered_items[filtered_items['articleType'] != item_category]
        matching_items = find_matching_items_with_rag(filtered_items, item_descs)
        
        paths = []
        for item in matching_items:
            item_id = item['id'] if isinstance(item, pd.Series) else item.id
            image_path = f'../data/sample_clothes/sample_images/{item_id}.jpg'
            # dst_dir = os.path.join(os.getcwd(), "cache_images")
            # os.makedirs(dst_dir, exist_ok=True)
            # dst = os.path.join(dst_dir, f"{item_id}.jpg")
            # shutil.copy(image_path, dst)
            paths.append(image_path)

        return json.dumps(image_analysis, indent=2), paths
    except Exception as e:
        return f"Error: {str(e)}", None

description = "Upload a product image to analyze its style, category, and gender using GPT-4o-mini. Get recommended matching items with images."

iface = gr.Interface(
    fn=recommend_items_from_image,
    inputs=gr.Image(type="numpy", label="Upload Product Image"),
    outputs=[
      gr.Textbox(label="Analysis Output (JSON)"),
      gr.Gallery(label="Recommended Items")  
    ],    title="Clothing Image Analyzer & Recommender",
    description=description,
)

if __name__ == "__main__":
    iface.launch(
        allowed_paths=["/Users/weiyima/ml-dev/data/sample_clothes/sample_images"]
    )