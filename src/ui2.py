import gradio as gr
import json, ast, os, tempfile
import pandas as pd
from PIL import Image as PILImage
from src.utils import encode_image_to_base64, find_similar_items
from src.main import analyze_image, get_embeddings

IMAGES_DIR = os.path.abspath("data/sample_clothes/sample_images")

CSS = """
#rec_gallery .gallery-item img {
  width: 128px !important;
  height: 128px !important;
  object-fit: contain;
}
"""

def get_unique_subcategories():
    styles_filepath = "data/sample_clothes/sample_styles.csv"
    if not os.path.exists(styles_filepath):
        return []
    styles_df = pd.read_csv(styles_filepath, on_bad_lines='skip')
    return styles_df['articleType'].unique().tolist()

SUBCATEGORIES = get_unique_subcategories()
# print(f"There are {len(subcategories)} unique subcategories, such as:")
# print(unique_subcategories)

def analyze_step(image_np):
    """Step 1: analyze the image, stash the JSON, and return it."""
    if image_np is None:
        # keep the Recommend button hidden if nothing uploaded
        return "Please upload an image first.", None, gr.update(visible=False)
    # save & encode
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        PILImage.fromarray(image_np).save(tmp.name)
        b64 = encode_image_to_base64(tmp.name)
    raw = analyze_image(b64, SUBCATEGORIES)
    # show the Recommend button now
    return raw, raw, gr.update(visible=True)

def recommend_step(image_np, analysis_json):
    """Step 2: read the stashed JSON, run RAG, and return image paths."""
    if analysis_json is None:
        return ["Please run Analyze first!"]
    info = json.loads(analysis_json)
    descs = info['items']
    gender = info['gender']
    category = info['category']

    styles_fp = "data/sample_clothes/sample_styles_with_embeddings.csv"
    if not os.path.exists(styles_fp):
        return ["Styles file not found."]
    df = pd.read_csv(styles_fp, on_bad_lines='skip')
    df['embeddings'] = df['embeddings'].apply(ast.literal_eval)
    df = df[df['gender'].isin([gender, 'Unisex'])]
    df = df[df['articleType'] != category]

    # compute similarity
    emb_list = df['embeddings'].tolist()
    similar_indices = []
    for d in descs:
        ie = get_embeddings([d])
        similar_indices += find_similar_items(ie, emb_list, threshold=0.6)

    paths = []
    for idx in similar_indices:
        pid = df.iloc[idx]['id']
        # 2) build absolute path under IMAGES_DIR
        img_path = os.path.join(IMAGES_DIR, f"{pid}.jpg")
        paths.append(img_path)
    return paths or ["No matches found."]

## Gradio Interface
instructions_txt = """
## Step 1: Upload an image to analyze the product item's style, category, and gender. We'll recommened a few new items in a different category to match your outfit!
"""
recommend_txt = """
## Step 2: Show images of the matching items based on gpt4o-mini's visual analysis and recommendation.
"""

with gr.Blocks(css=CSS) as demo:
    gr.Markdown(instructions_txt)
    img = gr.Image(type="numpy", label="Upload Product Image")
    analyze_btn = gr.Button("Analyze the image!")
    analysis_txt = gr.Textbox(label="Analysis Output (JSON)")
    
    gr.Markdown(recommend_txt)
    recommend_btn = gr.Button("Search for Matching Items?", visible=False)
    gallery = gr.Gallery(
        label="Matching Items",
        elem_id="rec_gallery",
        columns=4
        )
    analysis_state = gr.State()

    # — Step 1 —
    analyze_btn.click(
        fn=analyze_step,
        inputs=[img],
        outputs=[analysis_txt, analysis_state, recommend_btn],
    )

    # — Step 2 —
    recommend_btn.click(
        fn=recommend_step,
        inputs=[img, analysis_state],
        outputs=[gallery]
    )

demo.launch(
    allowed_paths=["/Users/weiyima/ml-dev/data/sample_clothes/sample_images"]
)