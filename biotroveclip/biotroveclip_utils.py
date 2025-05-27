"""
🌿 BioTroveCLIP Utilities Module
Core functions for BioTroveCLIP model setup and predictions

Author: Nikita Gavrilov
"""

import torch
from PIL import Image
import open_clip
import numpy as np
import time


def setup_biotroveclip():
    """
    🔧 Set up BioTroveCLIP model with CPU optimizations

    Returns:
        tuple: (model, preprocess_val, tokenizer, device)
    """
    print("🚀 Setting up BioTroveCLIP for flower identification...")

    # CPU Optimizations for Ryzen 5 7600X
    torch.set_num_threads(12)  # 6 cores × 2 threads = 12
    torch.backends.mkldnn.enabled = True

    print(f"🔧 Optimized for AMD Ryzen 5 7600X: Using 12 threads")
    print(f"⚡ MKL-DNN optimization enabled: {torch.backends.mkldnn.enabled}")

    # Load BioTroveCLIP model and components
    print("📦 Loading BioTroveCLIP model...")
    print("   ⏳ Loading model components...")

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        'hf-hub:BGLab/BioTrove-CLIP'
    )
    print("   ✅ Model loaded")

    tokenizer = open_clip.get_tokenizer('hf-hub:BGLab/BioTrove-CLIP')
    print("   ✅ Tokenizer loaded")

    # Set up for CPU
    device = "cpu"
    model = model.to(device)
    model.eval()
    print("   ✅ Model moved to device and set to eval mode")

    print(f"💻 Device: {device}")
    print("✅ BioTroveCLIP setup complete!")

    return model, preprocess_val, tokenizer, device


def create_flower_species_prompts():
    """
    🌸 Create text prompts for flower species identification

    Returns:
        list: List of text prompts for BioTroveCLIP
    """
    flower_species = [
        "a photo of Bellis perennis",
        "a photo of Matricaria chamomilla",
        "a photo of Leucanthemum vulgare"
    ]

    print("🏷️ Flower species for identification:")
    for species in flower_species:
        print(f"   - {species}")

    return flower_species


def predict_single_image(model, preprocess_val, tokenizer, device, image_path, flower_species):
    """
    🤖 Get BioTroveCLIP prediction for a single image

    Args:
        model: BioTroveCLIP model
        preprocess_val: Image preprocessing function
        tokenizer: Text tokenizer
        device: Computing device ('cpu' or 'cuda')
        image_path: Path to image file
        flower_species: List of species prompts

    Returns:
        tuple: (predicted_species, confidence_score, all_probabilities) or (None, None, None) if error
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess_val(image).unsqueeze(0).to(device)

        # Prepare text prompts
        text_input = tokenizer(flower_species).to(device)

        # Run inference
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            logits_per_image = image_features @ text_features.T
            probabilities = torch.softmax(logits_per_image, dim=-1)

        # Get prediction
        probs_np = probabilities[0].cpu().numpy()
        predicted_idx = np.argmax(probs_np)
        predicted_species = flower_species[predicted_idx].replace("a photo of ", "")
        confidence_score = probs_np[predicted_idx]

        return predicted_species, confidence_score, probs_np

    except Exception as e:
        return None, None, None


def format_time(seconds):
    """
    ⏰ Format seconds into human-readable time

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"