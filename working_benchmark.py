"""
🌸 BioCLIP Flower Identification Benchmark with Progress Bars
Enhanced with tqdm progress tracking and detailed timing analysis

Author: Nikita Gavrilov
"""

import torch
from PIL import Image
import open_clip
import pandas as pd
import numpy as np
import os
import time
from pathlib import Path
import json
from datetime import datetime, timedelta

# Progress bar handling - safe import
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("📊 Note: tqdm not available. Using simple progress indicators.")


    # Simple fallback progress function
    def tqdm(iterable, **kwargs):
        return iterable


def setup_bioclip():
    """
    🔧 Set up BioCLIP model with CPU optimizations
    """
    print("🚀 Setting up BioCLIP for flower identification...")

    # CPU Optimizations for your Ryzen 5 7600X
    torch.set_num_threads(12)  # 6 cores × 2 threads = 12
    torch.backends.mkldnn.enabled = True

    print(f"🔧 Optimized for AMD Ryzen 5 7600X: Using 12 threads")
    print(f"⚡ MKL-DNN optimization enabled: {torch.backends.mkldnn.enabled}")

    # Load BioCLIP model and components with progress
    print("📦 Loading BioCLIP model...")

    # Create a simple progress indicator without tqdm for model loading
    print("   ⏳ Loading model components...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        'hf-hub:imageomics/bioclip-vit-b-16-inat-only'
    )
    print("   ✅ Model loaded")

    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-vit-b-16-inat-only')
    print("   ✅ Tokenizer loaded")

    # Set up for CPU
    device = "cpu"
    model = model.to(device)
    model.eval()
    print("   ✅ Model moved to device and set to eval mode")

    print(f"💻 Device: {device}")
    print("✅ BioCLIP setup complete!")

    return model, preprocess_val, tokenizer, device


def create_flower_species_prompts():
    """
    🌸 Create text prompts for your three flower species
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


def load_and_validate_ground_truth(csv_path, dataset_name):
    """
    📁 Load and validate ground truth CSV file with progress tracking
    """
    print(f"📊 Loading {dataset_name} ground truth data...")

    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Validate columns
    required_columns = ['file_path', 'flower_name']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"❌ Missing columns in {dataset_name}: {missing_columns}")

    # Check for missing values
    if df[required_columns].isnull().any().any():
        print(f"⚠️ Warning: Found missing values in {dataset_name}")
        df = df.dropna(subset=required_columns)
        print(f"📊 After removing missing values: {len(df)} images")

    # Validate image files exist with simple progress
    print(f"🔍 Validating image files exist...")
    missing_files = []

    for idx, row in df.iterrows():
        if not os.path.exists(row['file_path']):
            missing_files.append(row['file_path'])

        # Show progress every 500 files
        if (idx + 1) % 500 == 0 or (idx + 1) == len(df):
            print(f"   ✅ Checked {idx + 1}/{len(df)} file paths")

    if missing_files:
        print(f"⚠️ Warning: {len(missing_files)} image files not found in {dataset_name}")
        print("📋 First few missing files:")
        for file in missing_files[:5]:
            print(f"   - {file}")

        # Remove entries with missing files
        df = df[df['file_path'].apply(os.path.exists)]
        print(f"📊 After removing missing files: {len(df)} images")

    # Show dataset statistics
    print(f"📈 {dataset_name}: {len(df)} valid images")
    print(f"🌸 Species distribution:")
    species_counts = df['flower_name'].value_counts()
    for species, count in species_counts.items():
        print(f"   - {species}: {count} images")

    return df


def predict_single_image(model, preprocess_val, tokenizer, device, image_path, flower_species):
    """
    🤖 Get BioCLIP prediction for a single image
    Returns: (predicted_species, confidence_score, all_probabilities) or (None, None, None) if error
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


def process_dataset(model, preprocess_val, tokenizer, device, df, flower_species, dataset_name):
    """
    🔄 Process entire dataset with progress tracking
    """
    print(f"\n🔄 Processing {dataset_name}...")
    print(f"📊 Total images: {len(df)}")

    results = []
    correct_predictions = 0
    errors = 0

    start_time = time.time()

    # 🔍 DEBUG: Show what BioCLIP species prompts look like
    print(f"\n🔍 DEBUG: BioCLIP species prompts:")
    for i, species in enumerate(flower_species):
        clean_name = species.replace("a photo of ", "")
        print(f"   {i}: '{species}' → '{clean_name}'")

    # Process each image with progress updates
    for idx, row in df.iterrows():
        image_path = row['file_path']
        true_species = row['flower_name']

        # Get prediction
        predicted_species, confidence, all_probs = predict_single_image(
            model, preprocess_val, tokenizer, device, image_path, flower_species
        )

        # 🔍 DEBUG: Show first 3 predictions in detail
        if predicted_species is not None and idx < 3:
            print(f"\n🔍 DEBUG Image {idx + 1}:")
            print(f"   📁 File: {os.path.basename(image_path)}")
            print(f"   📋 Ground truth: '{true_species}' (length: {len(true_species)})")
            print(f"   🤖 BioCLIP predicted: '{predicted_species}' (length: {len(predicted_species)})")
            print(f"   🎯 Confidence: {confidence:.3f}")
            print(f"   ⚖️ Direct comparison: {predicted_species == true_species}")
            print(f"   🔤 Lowercase comparison: {predicted_species.lower() == true_species.lower()}")
            print(f"   📊 All probabilities:")
            for i, species in enumerate(flower_species):
                clean_species = species.replace("a photo of ", "")
                print(f"      {clean_species}: {all_probs[i]:.3f}")

            # Check for hidden characters
            print(f"   🔍 Character analysis:")
            print(f"      True species ASCII: {[ord(c) for c in true_species]}")
            print(f"      Predicted ASCII: {[ord(c) for c in predicted_species]}")

        if predicted_species is not None:
            # Check if prediction is correct
            is_correct = predicted_species == true_species
            if is_correct:
                correct_predictions += 1

            # Store detailed results
            result = {
                'image_path': image_path,
                'true_species': true_species,
                'predicted_species': predicted_species,
                'confidence': float(confidence),
                'is_correct': is_correct,
                'all_probabilities': all_probs.tolist()
            }
            results.append(result)
        else:
            errors += 1

        processed_count = idx + 1

        # Show progress every 100 images (but show first few immediately)
        if processed_count <= 5 or processed_count % 100 == 0 or processed_count == len(df):
            elapsed_time = time.time() - start_time
            avg_time_per_image = elapsed_time / processed_count
            remaining_images = len(df) - processed_count
            estimated_remaining_time = remaining_images * avg_time_per_image

            current_accuracy = (correct_predictions / len(results)) * 100 if results else 0

            print(f"\n✅ Processed {processed_count}/{len(df)} images ({processed_count / len(df) * 100:.1f}%)")
            print(f"   📊 Current accuracy: {current_accuracy:.1f}%")
            print(f"   ✅ Correct predictions so far: {correct_predictions}")
            print(f"   ❌ Errors so far: {errors}")
            if processed_count < len(df):
                print(f"   ⏱️ Estimated time remaining: {format_time(estimated_remaining_time)}")
                print(f"   ⚡ Speed: {1 / avg_time_per_image:.1f} images/minute")

    # Calculate final statistics
    total_processed = len(results)
    final_accuracy = (correct_predictions / total_processed) * 100 if total_processed > 0 else 0
    processing_time = time.time() - start_time

    print(f"\n✅ {dataset_name} complete!")
    print(f"📊 Processed: {total_processed}/{len(df)} images")
    print(f"🎯 Final accuracy: {final_accuracy:.2f}% ({correct_predictions}/{total_processed})")
    print(f"❌ Errors: {errors} images")
    print(f"⏱️ Processing time: {format_time(processing_time)}")
    print(f"⚡ Average per image: {processing_time / len(df):.2f}s")

    return results, final_accuracy, processing_time, errors


def analyze_results(filtered_results, unfiltered_results, filtered_accuracy, unfiltered_accuracy,
                    filtered_time, unfiltered_time, filtered_errors, unfiltered_errors):
    """
    📈 Analyze and compare results between datasets with timing info
    """
    print("\n" + "=" * 70)
    print("📈 EXPERIMENT RESULTS ANALYSIS")
    print("=" * 70)

    # Main comparison
    print(f"\n🌸 BioCLIP Flower Identification Benchmark Results:")
    print(f"📁 Filtered dataset accuracy:   {filtered_accuracy:.2f}%")
    print(f"📁 Unfiltered dataset accuracy: {unfiltered_accuracy:.2f}%")

    improvement = filtered_accuracy - unfiltered_accuracy
    print(f"📈 Improvement from data cleaning: {improvement:+.2f} percentage points")

    if improvement > 0:
        print(f"✅ Data cleaning IMPROVED performance by {improvement:.2f}%")
    elif improvement < 0:
        print(f"❌ Data cleaning DECREASED performance by {abs(improvement):.2f}%")
    else:
        print(f"➖ Data cleaning had NO effect on performance")

    # Timing analysis
    print(f"\n⏱️ Processing Time Analysis:")
    print(
        f"📁 Filtered dataset:   {format_time(filtered_time)} ({filtered_time / len(filtered_results):.2f}s per image)")
    print(
        f"📁 Unfiltered dataset: {format_time(unfiltered_time)} ({unfiltered_time / len(unfiltered_results):.2f}s per image)")
    total_time = filtered_time + unfiltered_time
    total_images = len(filtered_results) + len(unfiltered_results)
    print(f"🎯 Total experiment time: {format_time(total_time)} ({total_images} images)")

    # Error analysis
    print(f"\n❌ Error Analysis:")
    print(f"📁 Filtered dataset errors:   {filtered_errors}")
    print(f"📁 Unfiltered dataset errors: {unfiltered_errors}")
    total_errors = filtered_errors + unfiltered_errors
    print(f"🎯 Total errors: {total_errors}")

    if unfiltered_errors > filtered_errors:
        print(f"⚠️ Unfiltered dataset had {unfiltered_errors - filtered_errors} more processing errors")

    # Confidence analysis
    if filtered_results and unfiltered_results:
        filtered_confidences = [r['confidence'] for r in filtered_results if r['is_correct']]
        unfiltered_confidences = [r['confidence'] for r in unfiltered_results if r['is_correct']]

        if filtered_confidences and unfiltered_confidences:
            print(f"\n🎯 Confidence Analysis (for correct predictions):")
            print(f"📁 Filtered dataset avg confidence:   {np.mean(filtered_confidences):.3f}")
            print(f"📁 Unfiltered dataset avg confidence: {np.mean(unfiltered_confidences):.3f}")

            confidence_improvement = np.mean(filtered_confidences) - np.mean(unfiltered_confidences)
            print(f"📈 Confidence improvement: {confidence_improvement:+.3f}")

    # Species-wise analysis
    print(f"\n🌸 Per-species accuracy analysis:")

    species_list = ["Bellis perennis", "Matricaria chamomilla", "Leucanthemum vulgare"]

    for species in species_list:
        # Filtered accuracy for this species
        filtered_species = [r for r in filtered_results if r['true_species'] == species]
        filtered_correct = sum(1 for r in filtered_species if r['is_correct'])
        filtered_total = len(filtered_species)
        filtered_species_acc = (filtered_correct / filtered_total * 100) if filtered_total > 0 else 0

        # Unfiltered accuracy for this species
        unfiltered_species = [r for r in unfiltered_results if r['true_species'] == species]
        unfiltered_correct = sum(1 for r in unfiltered_species if r['is_correct'])
        unfiltered_total = len(unfiltered_species)
        unfiltered_species_acc = (unfiltered_correct / unfiltered_total * 100) if unfiltered_total > 0 else 0

        species_improvement = filtered_species_acc - unfiltered_species_acc

        print(f"   🌼 {species}:")
        print(f"     Filtered:   {filtered_species_acc:.1f}% ({filtered_correct}/{filtered_total})")
        print(f"     Unfiltered: {unfiltered_species_acc:.1f}% ({unfiltered_correct}/{unfiltered_total})")
        print(f"     Improvement: {species_improvement:+.1f}%")


def save_results_with_timing(filtered_results, unfiltered_results, filtered_accuracy, unfiltered_accuracy,
                             filtered_time, unfiltered_time, filtered_errors, unfiltered_errors):
    """
    💾 Save detailed results with timing information
    """
    print(f"\n💾 Saving results...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results with timing
    results_data = {
        'experiment_info': {
            'timestamp': timestamp,
            'total_experiment_time_seconds': filtered_time + unfiltered_time,
            'total_experiment_time_formatted': format_time(filtered_time + unfiltered_time),
            'filtered_accuracy': filtered_accuracy,
            'unfiltered_accuracy': unfiltered_accuracy,
            'improvement': filtered_accuracy - unfiltered_accuracy,
            'total_filtered_images': len(filtered_results),
            'total_unfiltered_images': len(unfiltered_results),
            'filtered_processing_time': filtered_time,
            'unfiltered_processing_time': unfiltered_time,
            'filtered_errors': filtered_errors,
            'unfiltered_errors': unfiltered_errors,
            'avg_time_per_image_filtered': filtered_time / len(filtered_results) if filtered_results else 0,
            'avg_time_per_image_unfiltered': unfiltered_time / len(unfiltered_results) if unfiltered_results else 0
        },
        'filtered_results': filtered_results,
        'unfiltered_results': unfiltered_results
    }

    results_filename = f"bioclip_benchmark_results_{timestamp}.json"
    with open(results_filename, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"💾 Detailed results saved to: {results_filename}")

    # Save summary CSV with timing
    summary_data = {
        'Dataset': ['Filtered', 'Unfiltered'],
        'Accuracy': [filtered_accuracy, unfiltered_accuracy],
        'Total_Images': [len(filtered_results), len(unfiltered_results)],
        'Correct_Predictions': [
            sum(1 for r in filtered_results if r['is_correct']),
            sum(1 for r in unfiltered_results if r['is_correct'])
        ],
        'Processing_Time_Seconds': [filtered_time, unfiltered_time],
        'Processing_Time_Formatted': [format_time(filtered_time), format_time(unfiltered_time)],
        'Avg_Time_Per_Image': [
            filtered_time / len(filtered_results) if filtered_results else 0,
            unfiltered_time / len(unfiltered_results) if unfiltered_results else 0
        ],
        'Errors': [filtered_errors, unfiltered_errors]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_filename = f"bioclip_benchmark_summary_{timestamp}.csv"
    summary_df.to_csv(summary_filename, index=False)

    print(f"📊 Summary saved to: {summary_filename}")


def main():
    """
    🎯 Main function - runs the complete benchmark experiment with progress tracking
    """
    print("🌸 BioCLIP Flower Identification Benchmark Experiment")
    print("🎯 Comparing performance on filtered vs unfiltered datasets")
    print("📊 Enhanced with progress bars and timing analysis")
    print("=" * 70)

    experiment_start_time = time.time()

    try:
        # Step 1: Set up BioCLIP
        model, preprocess_val, tokenizer, device = setup_bioclip()

        # Step 2: Create flower species prompts
        flower_species = create_flower_species_prompts()

        # Step 3: Load ground truth data
        print(f"\n📁 Loading ground truth data...")

        # 👇 UPDATE THESE PATHS TO YOUR ACTUAL CSV FILES
        filtered_csv = "C:\\Users\\Nikita\\Projects\\ZeroShot\\data\\ground_truth_filtered.csv"
        unfiltered_csv = "C:\\Users\\Nikita\\Projects\\ZeroShot\\data\\ground_truth_original.csv"

        filtered_df = load_and_validate_ground_truth(filtered_csv, "Filtered Dataset")
        unfiltered_df = load_and_validate_ground_truth(unfiltered_csv, "Unfiltered Dataset")

        # Step 4: Process both datasets with progress tracking
        print(f"\n🚀 Starting benchmark experiment...")

        # Process filtered dataset
        filtered_results, filtered_accuracy, filtered_time, filtered_errors = process_dataset(
            model, preprocess_val, tokenizer, device,
            filtered_df, flower_species, "Filtered Dataset"
        )

        # Process unfiltered dataset
        unfiltered_results, unfiltered_accuracy, unfiltered_time, unfiltered_errors = process_dataset(
            model, preprocess_val, tokenizer, device,
            unfiltered_df, flower_species, "Unfiltered Dataset"
        )

        # Step 5: Analyze results with timing
        analyze_results(
            filtered_results, unfiltered_results,
            filtered_accuracy, unfiltered_accuracy,
            filtered_time, unfiltered_time,
            filtered_errors, unfiltered_errors
        )

        # Step 6: Save results with timing data
        save_results_with_timing(
            filtered_results, unfiltered_results,
            filtered_accuracy, unfiltered_accuracy,
            filtered_time, unfiltered_time,
            filtered_errors, unfiltered_errors
        )

        # Experiment completion summary
        total_experiment_time = time.time() - experiment_start_time
        total_images = len(filtered_df) + len(unfiltered_df)

        print(f"\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"⏱️ Total experiment time: {format_time(total_experiment_time)}")
        print(f"📊 Total images processed: {total_images}")
        print(f"⚡ Overall average time per image: {total_experiment_time / total_images:.2f}s")
        print(f"🎯 Your research question answered!")

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        print("🔧 Please check your file paths and data format")
        raise


if __name__ == "__main__":
    main()