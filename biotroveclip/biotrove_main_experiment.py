"""
🌿 BioTroveCLIP Flower Identification Benchmark - Main Experiment
Clean, modular version using imported utilities

Author: Nikita Gavrilov
"""

# Core imports
import time
import os

# Your custom modules
from biotroveclip_utils import setup_biotroveclip, create_flower_species_prompts, predict_single_image, format_time
from biotroveclip_data_utils import load_and_validate_ground_truth
from biotroveclip_analysis_utils import analyze_results_basic, save_results_enhanced
from biotroveclip_validation_utils import create_comprehensive_report
import biotroveclip_config as config


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

    # 🔍 DEBUG: Show what BioTroveCLIP species prompts look like
    if dataset_name == "Filtered Dataset":  # Only show once
        print(f"\n🔍 DEBUG: BioTroveCLIP species prompts:")
        for i, species in enumerate(flower_species):
            clean_name = species.replace("a photo of ", "")
            print(f"   {i}: '{species}' → '{clean_name}'")

    # Process each image with progress updates
    for idx, row in df.iterrows():
        image_path = row['file_path']
        true_species = row['species']

        # Get prediction
        predicted_species, confidence, all_probs = predict_single_image(
            model, preprocess_val, tokenizer, device, image_path, flower_species
        )

        # 🔍 DEBUG: Show first few predictions in detail (only for first dataset)
        if predicted_species is not None and idx < config.DEBUG_SAMPLE_SIZE and dataset_name == "Filtered Dataset":
            print(f"\n🔍 DEBUG Image {idx + 1}:")
            print(f"   📁 File: {os.path.basename(image_path)}")
            print(f"   📋 Ground truth: '{true_species}' (length: {len(true_species)})")
            print(f"   🤖 BioTroveCLIP predicted: '{predicted_species}' (length: {len(predicted_species)})")
            print(f"   🎯 Confidence: {confidence:.3f}")
            print(f"   ⚖️ Direct comparison: {predicted_species == true_species}")
            print(f"   🔤 Lowercase comparison: {predicted_species.lower() == true_species.lower()}")

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

        # Show progress every N images
        if (processed_count <= 5 or
                processed_count % config.PROGRESS_UPDATE_INTERVAL == 0 or
                processed_count == len(df)):

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


def main():
    """
    🎯 Main function - runs the complete benchmark experiment
    """
    print("🌿 BioTroveCLIP Flower Identification Benchmark Experiment")
    print("🎯 Comparing performance on filtered vs unfiltered datasets")
    print("📊 Modular version with enhanced validation")
    print("=" * 70)

    # Validate configuration
    if not config.validate_config():
        print("❌ Configuration validation failed. Please fix errors and try again.")
        return

    # Print configuration summary
    config.print_config_summary()

    # Create output directory
    config.create_output_dir()

    experiment_start_time = time.time()

    try:
        # Step 1: Set up BioTroveCLIP
        model, preprocess_val, tokenizer, device = setup_biotroveclip()

        # Step 2: Create flower species prompts
        flower_species = create_flower_species_prompts()

        # Step 3: Load ground truth data
        print(f"\n📁 Loading ground truth data...")

        filtered_df = load_and_validate_ground_truth(config.FILTERED_CSV, "Filtered Dataset")
        unfiltered_df = load_and_validate_ground_truth(config.UNFILTERED_CSV, "Unfiltered Dataset")

        # Step 4: Process both datasets
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

        # Step 5: Basic analysis
        analyze_results_basic(
            filtered_results, unfiltered_results,
            filtered_accuracy, unfiltered_accuracy,
            filtered_time, unfiltered_time,
            filtered_errors, unfiltered_errors
        )

        # Step 6: Enhanced validation analysis
        print(f"\n🔍 Running enhanced validation analysis...")
        create_comprehensive_report(
            filtered_results, unfiltered_results,
            filtered_accuracy, unfiltered_accuracy
        )

        # Step 7: Save results
        save_results_enhanced(
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
        print(f"🎯 Enhanced validation completed!")
        print(f"📁 Results saved to: {config.OUTPUT_DIR}/")

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        print("🔧 Please check your configuration and data files")
        raise


if __name__ == "__main__":
    main()