import gradio as gr
import json, ast, os, tempfile
import pandas as pd
from PIL import Image as PILImage
from src.utils import encode_image_to_base64, find_similar_items
from src.main import analyze_image, get_embeddings

IMAGES_DIR = os.path.abspath("data/sample_clothes/sample_images")

CSS = """
/* overall page bg and font */
body, .block {
  background-color: #f5f7fa !important;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
/* header */
#header {
  background-color: #ffffff;
  padding: 1rem 2rem;
  border-bottom: 1px solid #e0e0e0;
}
#header h1 {
  color: #0d47a1;
  margin: 0;
  font-size: 1.8rem;
}
.subtitle {
  color: #555555;
  margin-top: 0.2rem;
  margin-bottom: 1.5rem;
}
/* buttons */
.gr-button {
  background-color: #0d47a1 !important;
  color: white !important;
  border-radius: 0.3rem !important;
  padding: 0.5rem 1rem !important;
  font-weight: 600 !important;
}
.gr-button:hover {
  background-color: #1565c0 !important;
}
/* card containers */
.card {
  background-color: #ffffff;
  border: 1px solid #e0e0e0;
  border-radius: 0.5rem;
  padding: 1rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  margin-bottom: 1rem;
}
/* gallery thumbnails */
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
        return "👉 Please upload an image first.", None, gr.update(visible=False)
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
        return ["⚠️ Run “Analyze” first!"]
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
        similar_indices += find_similar_items(ie, emb_list)

    paths = []
    for idx in similar_indices:
        pid = df.iloc[idx]['id']
        # 2) build absolute path under IMAGES_DIR
        img_path = os.path.join(IMAGES_DIR, f"{pid}.jpg")
        paths.append(img_path)
    return paths or ["No matches found."]

# —— Build the UI ——
with gr.Blocks(css=CSS) as demo:
    # HEADER
    gr.HTML("""
      <div id="header">
        <h1>👗 GPT4o-mini Clothing Style Analyzer & Recommender</h1>
        <p></p>
        <p class="subtitle"> 
        Step 1: Upload an image for analysis … 
        Step 2: Let the AI recommened a few new items to match the style!
        </p>
      </div>
    """)

    # === Step 1 card ===
    with gr.Row(elem_classes="card"):
        with gr.Column(scale=6):
            img = gr.Image(type="numpy", label="Upload Product Image")
            analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")
        with gr.Column(scale=6):
            analysis_txt = gr.Textbox(
                label="Analysis Output (JSON)",
                interactive=False, lines=8,
                placeholder="Your JSON will appear here…"
            )

    analysis_state = gr.State()
    recommend_btn   = gr.Button("🛍️ Show Matching Items", visible=False, variant="primary")

    # === Step 2 card ===
    with gr.Accordion("Step 2: Recommendations", open=False, elem_classes="card"):
        gallery = gr.Gallery(
            label="Matching Items",
            elem_id="rec_gallery",
            columns=4, height="300px"
        )

    # wiring
    analyze_btn.click(
        fn=analyze_step,
        inputs=[img],
        outputs=[analysis_txt, analysis_state, recommend_btn],
        show_progress=True
    )
    recommend_btn.click(
        fn=recommend_step,
        inputs=[img, analysis_state],
        outputs=[gallery],
        show_progress=True
    )

demo.launch(allowed_paths=[IMAGES_DIR])