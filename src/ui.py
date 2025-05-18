import gradio as gr
import json, ast, os, tempfile
import pandas as pd
import base64
import time
from PIL import Image as PILImage
from src.main import analyze_image, get_embeddings, check_match, encode_image_to_base64, find_similar_items, generate_combined_outfit

IMAGES_DIR = os.path.abspath("data/sample_clothes/sample_images")
OUTPUT_DIR = os.path.abspath("data/test")

# Load CSS from external file
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'static', 'styles.css')
    with open(css_path, 'r') as f:
        return f.read()

CSS = load_css()
# CSS = ""

# Get unique subcategories from the inventory dataset
def get_unique_subcategories():
    styles_filepath = "data/sample_clothes/sample_styles.csv"
    if not os.path.exists(styles_filepath):
        return []
    styles_df = pd.read_csv(styles_filepath, on_bad_lines='skip')
    return styles_df['articleType'].unique().tolist()

SUBCATEGORIES = get_unique_subcategories()

# Step 1 : User input, image analysis, feature extraction
def analyze_step(image_np):
    if image_np is None:
        return None, None, gr.update(visible=False)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        PILImage.fromarray(image_np).save(tmp.name)
        b64 = encode_image_to_base64(tmp.name)
    raw = analyze_image(b64, SUBCATEGORIES)
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"error": raw}
    return parsed, raw, gr.update(visible=True)

# Step 2 : Load Embeddings from Vector Store, Retrieve similar items, Load images
def recommend_step(analysis_json):
    if analysis_json is None:
        return [], gr.update(visible=False)
    info = json.loads(analysis_json)
    descs = info['items']
    gender = info['gender']
    category = info['category']

    styles_fp = "data/sample_clothes/sample_styles_with_embeddings.csv"
    if not os.path.exists(styles_fp):
        return [], gr.update(visible=False)
    df = pd.read_csv(styles_fp, on_bad_lines='skip')
    df['embeddings'] = df['embeddings'].apply(ast.literal_eval)
    df = df[df['gender'].isin([gender, 'Unisex'])]
    df = df[df['articleType'] != category]

    emb_list = df['embeddings'].tolist() # embedding from vector store
    similar_indices = []
    for d in descs:
        ie = get_embeddings([d]) # input query embedding
        similar_indices += [idx for idx, _ in find_similar_items(ie, emb_list)]
    similar_indices = list(dict.fromkeys(similar_indices))[:4] # limit 4 recommendations

    # Create image gallery items
    gallery_items = []
    for idx in similar_indices:
        pid = df.iloc[idx]['id']
        img_path = os.path.join(IMAGES_DIR, f"{pid}.jpg")
        if os.path.exists(img_path):
            caption = f"Product {pid}"
            gallery_items.append((img_path, caption))
    if not gallery_items:
        return [], gr.update(visible=False)
    return gallery_items, gr.update(visible=True)

# Step 3: Validate Outfit Matches
def validate_matches_step(image_np, gallery_paths):
    if image_np is None or not gallery_paths:
        return [], gr.update(visible=False)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        PILImage.fromarray(image_np).save(tmp.name)
        reference_b64 = encode_image_to_base64(tmp.name)
    validated = []
    reasons = []
    for img_path in gallery_paths:
        if isinstance(img_path, tuple):
            img_path = img_path[0]
        suggested_b64 = encode_image_to_base64(img_path)
        result = check_match(reference_b64, suggested_b64)
        answer = None
        reason = None
        if hasattr(result, 'answer'):
            answer = result.answer
            reason = getattr(result, 'reason', '')
        elif isinstance(result, dict):
            answer = result.get('answer', '')
            reason = result.get('reason', '')
        print(f"Validation for item: answer={answer}, reason={reason}")
        if answer and answer.lower() == "yes":
            validated.append(img_path)
            reasons.append(f"{reason}")
    if not validated:
        return [], gr.update(visible=False)
    gallery_items = list(zip(validated, reasons))
    return gallery_items, gr.update(visible=True)

# Step 4: Generate Outfit Image
def generate_outfit_step(image_np, gallery_paths):
    if image_np is None or not gallery_paths:
        return None
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        PILImage.fromarray(image_np).save(tmp.name)
        reference_image_path = tmp.name
    accessory_paths = [p[0] if isinstance(p, tuple) else p for p in gallery_paths]
    img = generate_combined_outfit(reference_image_path, accessory_paths)
    image_base64 = img.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    timestamp = int(time.time())
    output_path = f"data/test/output_{timestamp}.png"
    with open(output_path, "wb") as f:
        f.write(image_bytes)

    return output_path

# Gradio Interface
with gr.Blocks(css=CSS) as demo:
    gr.HTML("""
      <div id="header">
        <h1> 👗 Outfit Assistant - Prototype </h1>
        <p>Discover, recommend, and validate fashion looks with AI. Powered by GPT-4o-mini.</p>
      </div>
    """)
    gr.HTML("""
      <div class="stepper">
        <div class="step">1. Analyze your fit</div>
        <div class="step inactive">2. Recommend matching fits </div>
        <div class="step inactive">3. Validate the recommendation </div>
      </div>
    """)
    with gr.Row(elem_id="main-card"):
        with gr.Column(scale=6):
            gr.Markdown("""
            ### 1️⃣ Upload Product Image
            Upload a clothing product image for instant AI-powered analysis.
            """)
            img = gr.Image(type="numpy", label="Upload Product Image", elem_id="upload-image")
            analyze_btn = gr.Button("🔍 Analyze the Look", variant="primary")
        with gr.Column(scale=6):
            analysis_json = gr.JSON(
                label="Analysis Output (JSON)",
                show_indices=False,
                visible=True,
            )
    analysis_state = gr.State()
    with gr.Row(elem_id="main-card"):
        with gr.Column():
            gr.Markdown("""
            ### 2️⃣ AI Recommendation
            Click below to get AI-powered product recommendations matching your style.
            """)
            recommend_btn = gr.Button("🛍️ Recommend Matching Products", visible=False, variant="primary")
            gallery = gr.Gallery(
                label="Matching Items",
                elem_id="rec_gallery",
                columns=4, 
                height="320px",
                interactive=False,
            )
    with gr.Row(elem_id="main-card"):
        with gr.Column():
            gr.Markdown("""
            ### 3️⃣ Validate Outfit Matches
            Review the recommended matches and validate the best outfit combinations.
            """)
            validate_btn = gr.Button("✅ Validate Outfit Matches", visible=False, variant="primary")
            validated_gallery = gr.Gallery(
                label="Validated Matches",
                columns=2,
                height="320px",
                interactive=False,
            )
    with gr.Row(elem_id="main-card"):
        with gr.Column():
            gr.Markdown("""
            ### 4️⃣ Generate Combined Outfit Image
            Create a new look by combining the validated outfit items into a single image.
            """)
            output_img = gr.Image(label="Generated Outfit", elem_id="output-image", height=768, width=512)
            generate_btn = gr.Button("🎨 Generate Combined Outfit", visible=False, variant="primary")


# Wiring up the steps
    analyze_btn.click(
        fn=analyze_step,
        inputs=[img],
        outputs=[analysis_json, analysis_state, recommend_btn],
        show_progress=True
    )
    recommend_btn.click(
        fn=recommend_step,
        inputs=[analysis_state],
        outputs=[gallery,validate_btn],
        show_progress=True
    )
    validate_btn.click(
        fn=validate_matches_step,
        inputs=[img, gallery],
        outputs=[validated_gallery, generate_btn],
        show_progress=True
    )
    generate_btn.click(
        fn=generate_outfit_step,
        inputs=[img, validated_gallery],
        outputs=[output_img],
        show_progress=True
    )

demo.launch(allowed_paths=[IMAGES_DIR, OUTPUT_DIR])


