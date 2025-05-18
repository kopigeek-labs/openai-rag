# openai-rag

## Overview

This repo contains a demo "AI-powered Outfit Assistant" that analyzes clothing images, recommends matching items using OpenAI's GPT-4o-mini and Retrieval Augment Generation. The jupyter notebook shows the building blocks and the src folder contains an app and Gradio-based UI for seamless user interaction.

## Features

- **Image Analysis:** Upload a clothing product image and receive a structured JSON analysis (items, category, gender) using GPT-4o-mini.
- **AI Recommendations:** Retrieve and display semantically similar items from a product catalog using vector embeddings and similarity search.
- **Outfit Validation:** Validate recommended matches with AI guardrails to ensure stylistic compatibility.
- **Image Generation** Put the look together. Using the reference image and the newly recommended items for a new holistic look.
- **Interactive UI:** Step-by-step workflow with image upload, recommendations, and validation, all in a modern Gradio interface.

## Interactive Python Notebook

- `retailnext.ipynb`: A Jupyter notebook with step-by-step detailed code and explanations for core logic that is used to enable the Gradio App.

## Project Structure

- `src/main.py`: Core logic for embeddings, image analysis, and validation.
- `src/ui.py`: Gradio UI and workflow orchestration.
- `src/utils.py`: Utility functions for embeddings and image encoding.
- `src/static/styles.css`: Custom CSS for UI styling.

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
