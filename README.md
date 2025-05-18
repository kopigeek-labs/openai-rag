# openai-rag

## Overview

**openai-rag** is an AI-powered outfit assistant that analyzes clothing images, recommends matching items, and validates outfit combinations using OpenAI's GPT-4o-mini and advanced embedding models. The app features a Gradio-based UI for seamless user interaction.

## Features

- **Image Analysis:** Upload a clothing product image and receive a structured JSON analysis (items, category, gender) using GPT-4o-mini.
- **AI Recommendations:** Retrieve and display visually and semantically similar items from a product catalog using vector embeddings and similarity search.
- **Outfit Validation:** Validate recommended matches with AI guardrails to ensure stylistic compatibility.
- **Interactive UI:** Step-by-step workflow with image upload, recommendations, and validation, all in a modern Gradio interface.

## How It Works

1. **Analyze Image:**
   - User uploads a product image.
   - The backend encodes the image and sends it to GPT-4o-mini for analysis.
   - The model returns a JSON with suggested matching items, category, and gender.
2. **Recommend Items:**
   - The app loads precomputed embeddings for the inventory.
   - It finds the most similar items (excluding same category/gender) using cosine similarity.
   - Top matches are displayed in a gallery.
3. **Validate Matches:**
   - The user can validate if recommended items truly match the uploaded product.
   - The system uses GPT-4o-mini to check compatibility and provides reasons for each validated match.

## Interactive Python Notebook

- `retailnext.ipynb`: A Jupyter notebook with step-by-step detailed code and explanations for core logic that is used to enable the Gradio App.

## Project Structure

- `src/main.py`: Core logic for embeddings, image analysis, and validation.
- `src/ui.py`: Gradio UI and workflow orchestration.
- `src/utils.py`: Utility functions for embeddings and image encoding.
- `src/static/styles.css`: Custom CSS for UI styling.
- `data/sample_clothes/`: Sample images and inventory CSVs.

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
