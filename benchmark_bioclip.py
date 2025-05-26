import os
import torch
from PIL import Image
import open_clip
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # For a progress bar

# --- Configuration ---
CLASS_NAMES_FILE = "C:\\Users\\Nikita\\Projects\\ZeroShot\\flower_classes.txt"
GROUND_TRUTH_CSV = "C:\\Users\\Nikita\\Projects\\ZeroShot\\ground_truth_original.csv"  # Your CSV with 'flower_name' and 'file_path'

# BioCLIP Model Configuration
MODEL_NAME = "ViT-B-16"  # Example, ensure this matches the BioCLIP version you intend to use
# This is the standard Hugging Face identifier. If you had issues,
# you might try "hf-hub:imageomics/bioclip/pytorch_model.bin" after ensuring libraries are updated.
PRETRAINED_WEIGHTS = "hf-hub:imageomics/bioclip/pytorch_model.bin"

BATCH_SIZE = 16  # Adjust based on your GPU/CPU memory
USE_BFLOAT16 = True  # Set to True to attempt using bfloat16, False for float32


# --- End Configuration ---

def load_class_names(file_path):
    """Loads class names from a text file, one per line."""
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    return class_names


def load_ground_truth_from_custom_csv(csv_path):
    """
    Loads ground truth from your specific CSV format.
    Expects columns: 'flower_name' (as label) and 'file_path' (full path to image).
    Returns a list of dictionaries: [{'file_path': 'path/to/img.jpg', 'label': 'Species Name'}, ...]
    """
    if not os.path.exists(csv_path):
        print(f"Error: Ground truth CSV file not found at {csv_path}")
        return []
    try:
        df = pd.read_csv(csv_path)
        if 'flower_name' not in df.columns or 'file_path' not in df.columns:
            print(f"Error: CSV file {csv_path} must contain 'flower_name' and 'file_path' columns.")
            return []

        ground_truth_entries = []
        for _, row in df.iterrows():
            ground_truth_entries.append({
                'file_path': str(row['file_path']).strip(),
                'label': str(row['flower_name']).strip()
            })
        return ground_truth_entries
    except Exception as e:
        print(f"Error reading or processing CSV file {csv_path}: {e}")
        return []


def main():
    # --- Device Setup ---
    device_type_for_autocast = "cpu"  # Default
    actual_device_name = "cpu"

    try:
        import torch_directml
        if torch_directml.is_available() and torch_directml.device_count() > 0:
            device = torch_directml.device()
            actual_device_name = f"DirectML: {torch_directml.device_name(0)}"
            # DirectML primarily uses float16 for mixed precision.
            # Forcing bfloat16 might not be supported or effective.
            # We'll set device_type_for_autocast to 'cpu' or 'cuda' if applicable later,
            # DirectML doesn't use 'privateuseone' as device_type in torch.autocast.
            # For now, let's assume if DirectML is used, bfloat16 autocast might not apply as intended.
            # PyTorch's generic torch.autocast will need 'cpu' or 'cuda'.
            # If using DirectML, true bfloat16 via torch.autocast might be tricky.
            # We will print a warning if bfloat16 is requested with DML.
            device_type_for_autocast = "cpu"  # Placeholder, DirectML's interaction with torch.autocast for bfloat16 is nuanced
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            actual_device_name = f"CUDA: {torch.cuda.get_device_name(0)}"
            device_type_for_autocast = "cuda"
        else:
            device = torch.device("cpu")
            actual_device_name = "CPU"
            device_type_for_autocast = "cpu"
    except ImportError:
        print("torch_directml not found. Checking for CUDA or using CPU.")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            actual_device_name = f"CUDA: {torch.cuda.get_device_name(0)}"
            device_type_for_autocast = "cuda"
        else:
            device = torch.device("cpu")
            actual_device_name = "CPU"
            device_type_for_autocast = "cpu"

    print(f"Using device: {actual_device_name} (Pytorch device: {device})")

    enable_autocast_with_bfloat16 = USE_BFLOAT16
    if USE_BFLOAT16 and str(device.type) == "privateuseone":  # Heuristic for DirectML device object
        print(f"Warning: bfloat16 is specifically requested, but DirectML ({actual_device_name}) "
              "has limited or no native bfloat16 support via torch.autocast. "
              "Mixed precision might use float16 if supported, or operations might remain float32.")
        # Forcing device_type to 'cpu' for autocast with DML might not be what we want.
        # Autocast for DML is usually managed by DML itself if mixed precision is on.
        # For explicit bfloat16, it's safest to signal it might not work as expected.
        # We will still pass dtype=torch.bfloat16 to autocast, but its effect on DML is uncertain.
        # Let's assume device_type_for_autocast should match the actual device logic where possible.
        # If torch_directml.device() gives a device object, its type might be 'privateuseone'
        # torch.autocast expects 'cpu' or 'cuda'.
        # This part is tricky for DirectML & bfloat16. Let's try with device.type if not 'cpu' or 'cuda'.
        if device.type not in ['cpu', 'cuda']:
            print(
                f"Device type for autocast with DirectML ({device.type}) might not be directly supported by torch.autocast. "
                "Proceeding with caution.")
            # Fallback to CPU for autocast type if device.type is not 'cpu' or 'cuda' for safety with torch.autocast
            device_type_for_autocast = "cpu" if device.type != "cuda" else "cuda"

    # --- Model Loading ---
    print(f"Loading BioCLIP model: {MODEL_NAME} with pretrained weights: {PRETRAINED_WEIGHTS}...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED_WEIGHTS,
            device=device  # Pass the determined device here
        )
        model.eval()  # Set model to evaluation mode
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        print("Model loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print(
            "Please ensure 'open_clip_torch' and 'huggingface_hub' are up to date (`pip install --upgrade open_clip_torch huggingface_hub`).")
        print(
            f"If issues persist with '{PRETRAINED_WEIGHTS}', you could try a more specific path like 'hf-hub:imageomics/bioclip/pytorch_model.bin' if available for this model.")
        return  # Exit if model fails to load

    # --- Text Processing ---
    if not os.path.exists(CLASS_NAMES_FILE):
        print(f"Error: Class names file not found at {CLASS_NAMES_FILE}")
        return
    class_names = load_class_names(CLASS_NAMES_FILE)
    if not class_names:
        print("Error: No class names loaded.")
        return
    print(f"Loaded {len(class_names)} class names.")

    text_prompts = [f"a photo of a {name}" for name in class_names]
    # Tokenization is done on CPU then moved to device if needed by model.encode_text
    tokenized_text = tokenizer(text_prompts).to(device)

    print("Encoding text prompts...")
    # Use torch.autocast for mixed precision
    with torch.autocast(device_type=device_type_for_autocast, dtype=torch.bfloat16,
                        enabled=enable_autocast_with_bfloat16):
        with torch.no_grad():
            text_features = model.encode_text(tokenized_text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    print("Text features encoded.")

    # --- Ground Truth Loading ---
    ground_truth_entries = load_ground_truth_from_custom_csv(GROUND_TRUTH_CSV)
    if not ground_truth_entries:
        print(f"Error: No ground truth data loaded. Exiting.")
        return
    print(f"Loaded {len(ground_truth_entries)} ground truth entries.")

    # --- Image Processing and Prediction ---
    all_predictions = []
    all_true_labels_for_processed_images = []
    correct_predictions_count = 0

    for i in tqdm(range(0, len(ground_truth_entries), BATCH_SIZE), desc="Processing images"):
        batch_data_entries = ground_truth_entries[i:i + BATCH_SIZE]
        batch_images_pil = []
        current_batch_true_labels = []

        for entry in batch_data_entries:
            image_full_path, true_label = entry['file_path'], entry['label']
            if true_label not in class_names:
                # print(f"Warning: Label '{true_label}' for '{image_full_path}' not in class_names.txt. Skipping.")
                continue
            if not os.path.exists(image_full_path):
                # print(f"Warning: Image '{image_full_path}' not found. Skipping.")
                continue
            try:
                image = Image.open(image_full_path).convert("RGB")
                batch_images_pil.append(image)
                current_batch_true_labels.append(true_label)
            except Exception as e:
                # print(f"Warning: Could not load image {image_full_path}. Error: {e}")
                continue

        if not batch_images_pil: continue

        # Preprocessing is typically a CPU task, then tensor is moved to device
        batch_images_processed = torch.stack([preprocess(img) for img in batch_images_pil]).to(device)

        with torch.autocast(device_type=device_type_for_autocast, dtype=torch.bfloat16,
                            enabled=enable_autocast_with_bfloat16):
            with torch.no_grad():
                image_features = model.encode_image(batch_images_processed)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()

        top_k_probs, top_k_indices = logits_per_image.topk(1, dim=-1)

        for idx in range(len(batch_images_pil)):
            predicted_class_index = top_k_indices[idx][0].item()
            predicted_class_name = class_names[predicted_class_index]
            true_class_name_for_this_image = current_batch_true_labels[idx]

            all_predictions.append(predicted_class_name)
            all_true_labels_for_processed_images.append(true_class_name_for_this_image)
            if predicted_class_name == true_class_name_for_this_image:
                correct_predictions_count += 1

    # --- Results ---
    if not all_true_labels_for_processed_images:
        print("No valid images processed. Cannot calculate accuracy.")
        return

    total_successfully_processed_images = len(all_true_labels_for_processed_images)
    if total_successfully_processed_images > 0:
        top1_accuracy = accuracy_score(all_true_labels_for_processed_images, all_predictions)
        print(f"\n--- Results ---")
        print(f"Device used: {actual_device_name} (Pytorch device: {device})")
        print(f"bfloat16 autocast attempted: {enable_autocast_with_bfloat16}")
        print(f"Total images successfully processed: {total_successfully_processed_images}")
        print(f"Correct predictions: {correct_predictions_count}")
        print(f"Top-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy * 100:.2f}%)")


if __name__ == "__main__":
    # Before running, ensure libraries are up-to-date:
    # pip install --upgrade torch torchvision torchaudio open_clip_torch Pillow pandas scikit-learn tqdm huggingface_hub
    # For AMD GPU on Windows (DirectML): pip install torch-directml
    main()