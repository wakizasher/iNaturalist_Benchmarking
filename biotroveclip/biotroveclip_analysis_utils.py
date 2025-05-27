"""
📊 Analysis Utilities Module
Basic results analysis and reporting functions

Author: Nikita Gavrilov
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from biotroveclip_utils import format_time
import biotroveclip_config as config


def analyze_results_basic(filtered_results, unfiltered_results, filtered_accuracy, unfiltered_accuracy,
                          filtered_time, unfiltered_time, filtered_errors, unfiltered_errors):
    """
    📈 Basic analysis and comparison of results between datasets

    Args:
        filtered_results (list): Results from filtered dataset
        unfiltered_results (list): Results from unfiltered dataset
        filtered_accuracy (float): Accuracy on filtered dataset
        unfiltered_accuracy (float): Accuracy on unfiltered dataset
        filtered_time (float): Processing time for filtered dataset
        unfiltered_time (float): Processing time for unfiltered dataset
        filtered_errors (int): Number of errors in filtered dataset
        unfiltered_errors (int): Number of errors in unfiltered dataset
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

    for species in config.SPECIES_NAMES:
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


def calculate_confusion_matrix(results, species_names):
    """
    📊 Calculate confusion matrix for results

    Args:
        results (list): List of prediction results
        species_names (list): List of species names

    Returns:
        pd.DataFrame: Confusion matrix
    """
    # Create mapping from species name to index
    species_to_idx = {species: i for i, species in enumerate(species_names)}

    # Initialize confusion matrix
    n_species = len(species_names)
    confusion = np.zeros((n_species, n_species), dtype=int)

    # Fill confusion matrix
    for result in results:
        true_idx = species_to_idx[result['true_species']]
        pred_idx = species_to_idx[result['predicted_species']]
        confusion[true_idx][pred_idx] += 1

    # Convert to DataFrame for better display
    confusion_df = pd.DataFrame(
        confusion,
        index=species_names,
        columns=species_names
    )

    return confusion_df


def save_results_enhanced(filtered_results, unfiltered_results, filtered_accuracy, unfiltered_accuracy,
                          filtered_time, unfiltered_time, filtered_errors, unfiltered_errors):
    """
    💾 Save detailed results with timing information

    Args:
        filtered_results (list): Results from filtered dataset
        unfiltered_results (list): Results from unfiltered dataset
        filtered_accuracy (float): Accuracy on filtered dataset
        unfiltered_accuracy (float): Accuracy on unfiltered dataset
        filtered_time (float): Processing time for filtered dataset
        unfiltered_time (float): Processing time for unfiltered dataset
        filtered_errors (int): Number of errors in filtered dataset
        unfiltered_errors (int): Number of errors in unfiltered dataset
    """
    print(f"\n💾 Saving results...")

    # Save detailed results with timing
    results_data = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
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

    results_filename = config.get_output_filename("bioclip_benchmark_results", "json")
    results_path = f"{config.OUTPUT_DIR}/{results_filename}"

    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"💾 Detailed results saved to: {results_path}")

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
    summary_filename = config.get_output_filename("bioclip_benchmark_summary", "csv")
    summary_path = f"{config.OUTPUT_DIR}/{summary_filename}"

    summary_df.to_csv(summary_path, index=False)
    print(f"📊 Summary saved to: {summary_path}")


def print_quick_summary(filtered_accuracy, unfiltered_accuracy, filtered_time, unfiltered_time):
    """
    🎯 Print a quick summary of results

    Args:
        filtered_accuracy (float): Accuracy on filtered dataset
        unfiltered_accuracy (float): Accuracy on unfiltered dataset
        filtered_time (float): Processing time for filtered dataset
        unfiltered_time (float): Processing time for unfiltered dataset
    """
    improvement = filtered_accuracy - unfiltered_accuracy
    total_time = filtered_time + unfiltered_time

    print(f"\n🎯 QUICK SUMMARY:")
    print(f"📊 Filtered: {filtered_accuracy:.1f}% | Unfiltered: {unfiltered_accuracy:.1f}%")
    print(f"📈 Improvement: {improvement:+.1f} percentage points")
    print(f"⏱️ Total time: {format_time(total_time)}")

    if improvement > 0:
        print(f"✅ Data cleaning helped!")
    else:
        print(f"❌ Data cleaning didn't help much")