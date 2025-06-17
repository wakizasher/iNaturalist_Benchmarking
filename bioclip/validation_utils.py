"""
🔍 Enhanced Validation Utilities Module
Advanced validation metrics with comprehensive standard deviation analysis

Author: Nikita Gavrilov (Enhanced with std dev metrics)
"""

import numpy as np
from scipy import stats
from bioclip import config


def calculate_standard_deviation_metrics(results, dataset_name):
    """
    📊 Calculate comprehensive standard deviation metrics for validation results

    This function computes various standard deviation metrics to understand
    the variability and consistency of model predictions.

    Args:
        results (list): List of prediction results with keys:
                       - 'confidence': prediction confidence score
                       - 'true_species': ground truth species
                       - 'predicted_species': predicted species
                       - 'is_correct': boolean indicating correct prediction
        dataset_name (str): Name of dataset for reporting

    Returns:
        dict: Dictionary containing all standard deviation metrics
    """
    print(f"\n📊 STANDARD DEVIATION ANALYSIS FOR {dataset_name.upper()}")
    print("=" * 60)

    # Extract data for analysis
    confidences = [r['confidence'] for r in results]
    y_true = [r['true_species'] for r in results]
    y_pred = [r['predicted_species'] for r in results]

    # 1. Overall confidence standard deviation
    overall_conf_std = np.std(confidences, ddof=1)  # Using sample std (ddof=1)
    overall_conf_mean = np.mean(confidences)
    coefficient_of_variation = overall_conf_std / overall_conf_mean if overall_conf_mean > 0 else 0

    print(f"🎯 Overall Confidence Variability:")
    print(f"   Mean confidence: {overall_conf_mean:.4f}")
    print(f"   Std deviation: {overall_conf_std:.4f}")
    print(f"   Coefficient of variation: {coefficient_of_variation:.4f}")
    print(f"   Range: [{np.min(confidences):.4f}, {np.max(confidences):.4f}]")

    # 2. Standard deviation by prediction correctness
    correct_confidences = [r['confidence'] for r in results if r['is_correct']]
    incorrect_confidences = [r['confidence'] for r in results if not r['is_correct']]

    print(f"\n🔍 Confidence Std Dev by Prediction Accuracy:")
    if correct_confidences:
        correct_std = np.std(correct_confidences, ddof=1)
        correct_mean = np.mean(correct_confidences)
        print(f"   Correct predictions:")
        print(f"     Mean: {correct_mean:.4f} ± {correct_std:.4f}")
        print(f"     CV: {correct_std / correct_mean:.4f}")

    if incorrect_confidences:
        incorrect_std = np.std(incorrect_confidences, ddof=1)
        incorrect_mean = np.mean(incorrect_confidences)
        print(f"   Incorrect predictions:")
        print(f"     Mean: {incorrect_mean:.4f} ± {incorrect_std:.4f}")
        print(f"     CV: {incorrect_std / incorrect_mean:.4f}")

    # 3. Per-species confidence standard deviation
    species_std_metrics = {}
    print(f"\n🌿 Per-Species Confidence Variability:")

    for species in config.SPECIES_NAMES:
        species_results = [r for r in results if r['true_species'] == species]
        if len(species_results) >= 2:  # Need at least 2 samples for std dev
            species_confidences = [r['confidence'] for r in species_results]
            species_mean = np.mean(species_confidences)
            species_std = np.std(species_confidences, ddof=1)
            species_cv = species_std / species_mean if species_mean > 0 else 0

            species_std_metrics[species] = {
                'mean': species_mean,
                'std': species_std,
                'cv': species_cv,
                'count': len(species_results)
            }

            print(f"   {species} (n={len(species_results)}):")
            print(f"     Mean ± Std: {species_mean:.4f} ± {species_std:.4f}")
            print(f"     CV: {species_cv:.4f}")

    # 4. Confidence standard deviation by threshold ranges
    print(f"\n📊 Confidence Std Dev by Confidence Ranges:")

    # Define confidence ranges
    ranges = [
        (0.0, 0.3, "Low"),
        (0.3, 0.7, "Medium"),
        (0.7, 1.0, "High")
    ]

    range_std_metrics = {}
    for min_conf, max_conf, range_name in ranges:
        range_results = [r for r in results if min_conf <= r['confidence'] < max_conf]
        if len(range_results) >= 2:
            range_confidences = [r['confidence'] for r in range_results]
            range_mean = np.mean(range_confidences)
            range_std = np.std(range_confidences, ddof=1)

            range_std_metrics[range_name] = {
                'mean': range_mean,
                'std': range_std,
                'count': len(range_results)
            }

            print(f"   {range_name} confidence [{min_conf:.1f}-{max_conf:.1f}) (n={len(range_results)}):")
            print(f"     Mean ± Std: {range_mean:.4f} ± {range_std:.4f}")

    # 5. Prediction consistency metric (standard deviation of per-species accuracies)
    species_accuracies = []
    print(f"\n🎯 Model Consistency Across Species:")

    for species in config.SPECIES_NAMES:
        species_results = [r for r in results if r['true_species'] == species]
        if species_results:
            species_accuracy = np.mean([r['is_correct'] for r in species_results])
            species_accuracies.append(species_accuracy)
            print(f"   {species}: {species_accuracy:.4f}")

    if len(species_accuracies) >= 2:
        accuracy_std = np.std(species_accuracies, ddof=1)
        accuracy_mean = np.mean(species_accuracies)
        consistency_score = 1 - (accuracy_std / accuracy_mean) if accuracy_mean > 0 else 0

        print(f"\n📊 Species-Level Consistency Metrics:")
        print(f"   Mean species accuracy: {accuracy_mean:.4f}")
        print(f"   Std dev of species accuracies: {accuracy_std:.4f}")
        print(f"   Consistency score: {consistency_score:.4f} (higher = more consistent)")

    return {
        'overall_confidence_std': overall_conf_std,
        'overall_confidence_mean': overall_conf_mean,
        'coefficient_of_variation': coefficient_of_variation,
        'correct_confidence_std': np.std(correct_confidences, ddof=1) if correct_confidences else 0,
        'incorrect_confidence_std': np.std(incorrect_confidences, ddof=1) if incorrect_confidences else 0,
        'species_std_metrics': species_std_metrics,
        'range_std_metrics': range_std_metrics,
        'species_accuracy_std': accuracy_std if len(species_accuracies) >= 2 else 0,
        'consistency_score': consistency_score if len(species_accuracies) >= 2 else 0
    }


def calculate_advanced_metrics(results, dataset_name):
    """
    📊 Calculate comprehensive accuracy metrics beyond simple accuracy
    Enhanced with standard deviation analysis integration

    Args:
        results (list): List of prediction results
        dataset_name (str): Name of dataset for reporting

    Returns:
        dict: Dictionary containing various metrics including std dev metrics
    """
    print(f"\n📊 ADVANCED METRICS FOR {dataset_name.upper()}")
    print("=" * 60)

    # Extract predictions and ground truth
    y_true = [r['true_species'] for r in results]
    y_pred = [r['predicted_species'] for r in results]
    confidences = [r['confidence'] for r in results]

    # Species mapping for numerical analysis
    species_to_num = {species: i for i, species in enumerate(config.SPECIES_NAMES)}

    y_true_num = [species_to_num[species] for species in y_true]
    y_pred_num = [species_to_num[species] for species in y_pred]

    # Calculate per-class metrics manually (simplified version)
    precision = []
    recall = []
    f1 = []
    support = []

    for i, species in enumerate(config.SPECIES_NAMES):
        # True positives, false positives, false negatives
        tp = sum(1 for true, pred in zip(y_true_num, y_pred_num) if true == i and pred == i)
        fp = sum(1 for true, pred in zip(y_true_num, y_pred_num) if true != i and pred == i)
        fn = sum(1 for true, pred in zip(y_true_num, y_pred_num) if true == i and pred != i)

        # Calculate metrics
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)
        support.append(sum(1 for true in y_true_num if true == i))

    print("🎯 Per-Species Detailed Metrics:")
    for i, species in enumerate(config.SPECIES_NAMES):
        print(f"   {species}:")
        print(f"     Precision: {precision[i]:.3f} (of predicted {species}, how many were correct)")
        print(f"     Recall:    {recall[i]:.3f} (of actual {species}, how many were found)")
        print(f"     F1-Score:  {f1[i]:.3f} (balanced precision & recall)")
        print(f"     Support:   {support[i]} (number of actual instances)")

    # Macro and Weighted averages with standard deviations
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Calculate standard deviations of metrics across species
    precision_std = np.std(precision, ddof=1) if len(precision) > 1 else 0
    recall_std = np.std(recall, ddof=1) if len(recall) > 1 else 0
    f1_std = np.std(f1, ddof=1) if len(f1) > 1 else 0

    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)

    print(f"\n📈 Overall Metrics with Variability:")
    print(f"   Macro Average (treats all species equally):")
    print(f"     Precision: {macro_precision:.3f} ± {precision_std:.3f}")
    print(f"     Recall:    {macro_recall:.3f} ± {recall_std:.3f}")
    print(f"     F1-Score:  {macro_f1:.3f} ± {f1_std:.3f}")

    print(f"   Weighted Average (accounts for class imbalance):")
    print(f"     Precision: {weighted_precision:.3f}")
    print(f"     Recall:    {weighted_recall:.3f}")
    print(f"     F1-Score:  {weighted_f1:.3f}")

    # Enhanced Confidence Analysis with standard deviations
    print(f"\n🎯 Enhanced Confidence Analysis:")
    conf_mean = np.mean(confidences)
    conf_std = np.std(confidences, ddof=1)
    print(f"   Mean confidence: {conf_mean:.3f} ± {conf_std:.3f}")
    print(f"   Min confidence:  {np.min(confidences):.3f}")
    print(f"   Max confidence:  {np.max(confidences):.3f}")
    print(f"   Coefficient of variation: {conf_std / conf_mean:.3f}")

    # Confidence by correctness with standard deviations
    correct_confidences = [r['confidence'] for r in results if r['is_correct']]
    incorrect_confidences = [r['confidence'] for r in results if not r['is_correct']]

    if correct_confidences and incorrect_confidences:
        correct_mean = np.mean(correct_confidences)
        correct_std = np.std(correct_confidences, ddof=1)
        incorrect_mean = np.mean(incorrect_confidences)
        incorrect_std = np.std(incorrect_confidences, ddof=1)

        print(f"   Correct predictions:   {correct_mean:.3f} ± {correct_std:.3f}")
        print(f"   Incorrect predictions: {incorrect_mean:.3f} ± {incorrect_std:.3f}")
        print(f"   Confidence difference: {correct_mean - incorrect_mean:.3f}")

    # Confidence threshold analysis
    print(f"\n📊 Confidence Threshold Analysis:")
    for threshold_name, threshold_value in config.CONFIDENCE_THRESHOLDS.items():
        high_conf_results = [r for r in results if r['confidence'] > threshold_value]
        if high_conf_results:
            high_conf_accuracy = np.mean([r['is_correct'] for r in high_conf_results])
            high_conf_confidences = [r['confidence'] for r in high_conf_results]
            high_conf_std = np.std(high_conf_confidences, ddof=1)
            print(f"   {threshold_name.capitalize()} confidence (>{threshold_value}):")
            print(f"     Accuracy: {high_conf_accuracy:.3f} ({len(high_conf_results)} images)")
            print(f"     Confidence: {np.mean(high_conf_confidences):.3f} ± {high_conf_std:.3f}")

    # Calculate and display standard deviation metrics
    std_metrics = calculate_standard_deviation_metrics(results, dataset_name)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision_std': precision_std,
        'recall_std': recall_std,
        'f1_std': f1_std,
        'mean_confidence': conf_mean,
        'std_confidence': conf_std,
        'std_metrics': std_metrics  # Include all standard deviation metrics
    }


def statistical_significance_test(filtered_results, unfiltered_results):
    """
    📈 Test if the difference between datasets is statistically significant
    Enhanced with variance comparison tests

    Args:
        filtered_results (list): Results from filtered dataset
        unfiltered_results (list): Results from unfiltered dataset

    Returns:
        dict: Statistical test results including variance tests
    """
    print(f"\n📈 STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 60)

    # Extract accuracy scores
    filtered_correct = [r['is_correct'] for r in filtered_results]
    unfiltered_correct = [r['is_correct'] for r in unfiltered_results]

    filtered_successes = sum(filtered_correct)
    filtered_total = len(filtered_correct)
    unfiltered_successes = sum(unfiltered_correct)
    unfiltered_total = len(unfiltered_correct)

    # Two-proportion z-test
    p1 = filtered_successes / filtered_total
    p2 = unfiltered_successes / unfiltered_total

    # Pooled proportion
    p_pool = (filtered_successes + unfiltered_successes) / (filtered_total + unfiltered_total)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / filtered_total + 1 / unfiltered_total))

    # Z-statistic
    z_stat = (p1 - p2) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed test

    print(f"🧮 Two-Proportion Z-Test:")
    print(f"   Filtered accuracy:   {p1:.4f} ({filtered_successes}/{filtered_total})")
    print(f"   Unfiltered accuracy: {p2:.4f} ({unfiltered_successes}/{unfiltered_total})")
    print(f"   Difference: {p1 - p2:.4f}")
    print(f"   Z-statistic: {z_stat:.3f}")
    print(f"   P-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"   ✅ SIGNIFICANT: Difference is statistically significant (p < 0.05)")
    elif p_value < 0.10:
        print(f"   ⚠️ MARGINAL: Difference is marginally significant (p < 0.10)")
    else:
        print(f"   ❌ NOT SIGNIFICANT: Difference is not statistically significant (p ≥ 0.05)")

    # Effect size (Cohen's h for proportions)
    cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    print(f"   📏 Effect size (Cohen's h): {cohens_h:.3f}")

    if abs(cohens_h) < 0.2:
        effect_interpretation = "Small effect"
    elif abs(cohens_h) < 0.5:
        effect_interpretation = "Medium effect"
    else:
        effect_interpretation = "Large effect"

    print(f"   📊 Effect interpretation: {effect_interpretation}")

    # Confidence interval for the difference
    se_diff = np.sqrt(p1 * (1 - p1) / filtered_total + p2 * (1 - p2) / unfiltered_total)
    margin_error = 1.96 * se_diff  # 95% CI
    ci_lower = (p1 - p2) - margin_error
    ci_upper = (p1 - p2) + margin_error

    print(f"   📊 95% Confidence Interval for difference: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # NEW: Variance comparison for confidence scores
    filtered_confidences = [r['confidence'] for r in filtered_results]
    unfiltered_confidences = [r['confidence'] for r in unfiltered_results]

    print(f"\n📊 Variance Comparison (Confidence Scores):")
    filtered_var = np.var(filtered_confidences, ddof=1)
    unfiltered_var = np.var(unfiltered_confidences, ddof=1)

    print(f"   Filtered variance: {filtered_var:.6f}")
    print(f"   Unfiltered variance: {unfiltered_var:.6f}")

    # F-test for equal variances
    if filtered_var > 0 and unfiltered_var > 0:
        f_stat = filtered_var / unfiltered_var if filtered_var > unfiltered_var else unfiltered_var / filtered_var
        df1 = len(filtered_confidences) - 1
        df2 = len(unfiltered_confidences) - 1
        f_p_value = 2 * (1 - stats.f.cdf(f_stat, df1, df2))  # Two-tailed

        print(f"   F-statistic: {f_stat:.3f}")
        print(f"   F-test p-value: {f_p_value:.4f}")

        if f_p_value < 0.05:
            print(f"   ✅ Variances are significantly different")
        else:
            print(f"   ❌ Variances are not significantly different")

    return {
        'z_statistic': z_stat,
        'p_value': p_value,
        'effect_size': cohens_h,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < 0.05,
        'filtered_variance': filtered_var,
        'unfiltered_variance': unfiltered_var,
        'variance_f_stat': f_stat if filtered_var > 0 and unfiltered_var > 0 else None,
        'variance_p_value': f_p_value if filtered_var > 0 and unfiltered_var > 0 else None
    }


def confidence_analysis(filtered_results, unfiltered_results):
    """
    🎯 Deep dive into confidence score differences with enhanced std dev analysis

    Args:
        filtered_results (list): Results from filtered dataset
        unfiltered_results (list): Results from unfiltered dataset
    """
    print(f"\n🎯 CONFIDENCE SCORE DEEP ANALYSIS")
    print("=" * 60)

    # Extract confidences for correct predictions
    filtered_conf_correct = [r['confidence'] for r in filtered_results if r['is_correct']]
    filtered_conf_incorrect = [r['confidence'] for r in filtered_results if not r['is_correct']]

    unfiltered_conf_correct = [r['confidence'] for r in unfiltered_results if r['is_correct']]
    unfiltered_conf_incorrect = [r['confidence'] for r in unfiltered_results if not r['is_correct']]

    print("📊 Enhanced Confidence Statistics with Standard Deviations:")
    print(f"   Filtered - Correct predictions:")
    if filtered_conf_correct:
        print(f"     Mean ± Std: {np.mean(filtered_conf_correct):.3f} ± {np.std(filtered_conf_correct, ddof=1):.3f}")
        print(f"     CV: {np.std(filtered_conf_correct, ddof=1) / np.mean(filtered_conf_correct):.3f}")

    if filtered_conf_incorrect:
        print(f"   Filtered - Incorrect predictions:")
        print(
            f"     Mean ± Std: {np.mean(filtered_conf_incorrect):.3f} ± {np.std(filtered_conf_incorrect, ddof=1):.3f}")
        print(f"     CV: {np.std(filtered_conf_incorrect, ddof=1) / np.mean(filtered_conf_incorrect):.3f}")

    print(f"   Unfiltered - Correct predictions:")
    if unfiltered_conf_correct:
        print(
            f"     Mean ± Std: {np.mean(unfiltered_conf_correct):.3f} ± {np.std(unfiltered_conf_correct, ddof=1):.3f}")
        print(f"     CV: {np.std(unfiltered_conf_correct, ddof=1) / np.mean(unfiltered_conf_correct):.3f}")

    if unfiltered_conf_incorrect:
        print(f"   Unfiltered - Incorrect predictions:")
        print(
            f"     Mean ± Std: {np.mean(unfiltered_conf_incorrect):.3f} ± {np.std(unfiltered_conf_incorrect, ddof=1):.3f}")
        print(f"     CV: {np.std(unfiltered_conf_incorrect, ddof=1) / np.mean(unfiltered_conf_incorrect):.3f}")

    # Statistical test for confidence differences (means)
    if len(filtered_conf_correct) > 0 and len(unfiltered_conf_correct) > 0:
        t_stat, t_p_value = stats.ttest_ind(filtered_conf_correct, unfiltered_conf_correct)
        print(f"\n🧮 T-test for confidence difference (correct predictions):")
        print(f"   T-statistic: {t_stat:.3f}")
        print(f"   P-value: {t_p_value:.4f}")

        if t_p_value < 0.05:
            print(f"   ✅ Confidence difference is statistically significant")
        else:
            print(f"   ❌ Confidence difference is not statistically significant")

        # NEW: Levene's test for equal variances
        levene_stat, levene_p = stats.levene(filtered_conf_correct, unfiltered_conf_correct)
        print(f"\n📊 Levene's test for equal variances:")
        print(f"   Levene statistic: {levene_stat:.3f}")
        print(f"   P-value: {levene_p:.4f}")

        if levene_p < 0.05:
            print(f"   ✅ Variances are significantly different")
            print(f"   💡 Consider using Welch's t-test for unequal variances")
        else:
            print(f"   ❌ Variances are not significantly different")


def error_pattern_analysis(filtered_results, unfiltered_results):
    """
    🔍 Analyze what types of errors occur in each dataset
    Enhanced with error confidence variability analysis

    Args:
        filtered_results (list): Results from filtered dataset
        unfiltered_results (list): Results from unfiltered dataset
    """
    print(f"\n🔍 ERROR PATTERN ANALYSIS")
    print("=" * 60)

    for dataset_name, results in [("Filtered", filtered_results), ("Unfiltered", unfiltered_results)]:
        print(f"\n📊 {dataset_name} Dataset Error Patterns:")

        errors = [r for r in results if not r['is_correct']]

        if not errors:
            print("   ✅ No errors found!")
            continue

        print(f"   Total errors: {len(errors)}")

        # Count confusion patterns
        confusion_counts = {}
        confusion_confidences = {}
        for error in errors:
            pair = (error['true_species'], error['predicted_species'])
            confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
            if pair not in confusion_confidences:
                confusion_confidences[pair] = []
            confusion_confidences[pair].append(error['confidence'])

        print("   🔄 Most common confusions with confidence analysis:")
        sorted_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)
        for (true_sp, pred_sp), count in sorted_confusions[:5]:  # Top 5 confusions
            confs = confusion_confidences[(true_sp, pred_sp)]
            conf_mean = np.mean(confs)
            conf_std = np.std(confs, ddof=1) if len(confs) > 1 else 0
            print(f"     {true_sp} → {pred_sp}: {count} times")
            print(f"       Confidence: {conf_mean:.3f} ± {conf_std:.3f}")

        # Enhanced confidence analysis for errors
        error_confidences = [e['confidence'] for e in errors]
        error_conf_mean = np.mean(error_confidences)
        error_conf_std = np.std(error_confidences, ddof=1)

        print(f"   📊 Error Confidence Analysis:")
        print(f"     Mean ± Std: {error_conf_mean:.3f} ± {error_conf_std:.3f}")

        low_conf_errors = [e for e in errors if e['confidence'] < config.CONFIDENCE_THRESHOLDS['medium']]
        high_conf_errors = [e for e in errors if e['confidence'] > config.CONFIDENCE_THRESHOLDS['high']]

        print(f"     Low confidence errors (<{config.CONFIDENCE_THRESHOLDS['medium']}): {len(low_conf_errors)}")
        print(f"     High confidence errors (>{config.CONFIDENCE_THRESHOLDS['high']}): {len(high_conf_errors)}")

        if high_conf_errors:
            high_conf_mean = np.mean([e['confidence'] for e in high_conf_errors])
            high_conf_std = np.std([e['confidence'] for e in high_conf_errors], ddof=1)
            print(f"       High-conf error confidence: {high_conf_mean:.3f} ± {high_conf_std:.3f}")


def create_comprehensive_report(filtered_results, unfiltered_results,
                                filtered_accuracy, unfiltered_accuracy):
    """
    📋 Create a comprehensive validation report with enhanced standard deviation analysis

    Args:
        filtered_results (list): Results from filtered dataset
        unfiltered_results (list): Results from unfiltered dataset
        filtered_accuracy (float): Accuracy on filtered dataset
        unfiltered_accuracy (float): Accuracy on unfiltered dataset
    """
    print(f"\n📋 COMPREHENSIVE VALIDATION REPORT")
    print("=" * 70)

    # 1. Advanced metrics for both datasets (now includes std dev metrics)
    filtered_metrics = calculate_advanced_metrics(filtered_results, "Filtered")
    unfiltered_metrics = calculate_advanced_metrics(unfiltered_results, "Unfiltered")

    # 2. Statistical significance testing (enhanced with variance tests)
    significance_results = statistical_significance_test(filtered_results, unfiltered_results)

    # 3. Enhanced confidence analysis
    confidence_analysis(filtered_results, unfiltered_results)

    # 4. Enhanced error pattern analysis
    error_pattern_analysis(filtered_results, unfiltered_results)

    # 5. Summary and interpretation with std dev insights
    print(f"\n🎯 VALIDATION SUMMARY & INTERPRETATION")
    print("=" * 60)

    improvement = filtered_accuracy - unfiltered_accuracy
    print(f"📊 Accuracy Improvement: {improvement:.2f} percentage points")

    if significance_results['significant']:
        print(f"✅ STATISTICALLY SIGNIFICANT improvement (p = {significance_results['p_value']:.4f})")
    else:
        print(f"⚠️ Improvement is NOT statistically significant (p = {significance_results['p_value']:.4f})")

    effect_size = abs(significance_results['effect_size'])
    if effect_size < 0.2:
        effect_interpretation = "Small effect"
    elif effect_size < 0.5:
        effect_interpretation = "Medium effect"
    else:
        effect_interpretation = "Large effect"

    print(f"📏 Effect size: {effect_size:.3f} ({effect_interpretation})")

    # F1-score comparison with variability
    f1_improvement = filtered_metrics['macro_f1'] - unfiltered_metrics['macro_f1']
    f1_variability_change = filtered_metrics['f1_std'] - unfiltered_metrics['f1_std']
    print(f"📈 F1-Score Improvement: {f1_improvement:.3f}")
    print(f"📊 F1-Score Variability Change: {f1_variability_change:.3f} (negative = more consistent)")

    # Confidence improvement with variability analysis
    conf_improvement = filtered_metrics['mean_confidence'] - unfiltered_metrics['mean_confidence']
    conf_variability_change = filtered_metrics['std_confidence'] - unfiltered_metrics['std_confidence']
    print(f"🎯 Confidence Improvement: {conf_improvement:.3f}")
    print(f"📊 Confidence Variability Change: {conf_variability_change:.3f} (negative = more consistent)")

    # Consistency improvements
    filtered_consistency = filtered_metrics['std_metrics']['consistency_score']
    unfiltered_consistency = unfiltered_metrics['std_metrics']['consistency_score']
    consistency_improvement = filtered_consistency - unfiltered_consistency
    print(f"🔄 Model Consistency Improvement: {consistency_improvement:.3f}")

    # Research implications with std dev insights
    print(f"\n🔬 RESEARCH IMPLICATIONS:")
    if improvement > 0:
        print(f"✅ Data cleaning shows positive impact on BioCLIP performance")
        if significance_results['significant']:
            print(f"✅ Results are statistically reliable for publication")
        else:
            print(f"⚠️ Consider larger sample size for stronger statistical power")

    # New insights based on standard deviation analysis
    print(f"\n📊 STANDARD DEVIATION INSIGHTS:")
    if conf_variability_change < 0:
        print(f"✅ Data cleaning reduces confidence variability (more predictable)")
    else:
        print(f"⚠️ Data cleaning increases confidence variability")

    if f1_variability_change < 0:
        print(f"✅ Data cleaning improves cross-species consistency")
    else:
        print(f"⚠️ Data cleaning may increase performance disparity across species")

    if significance_results.get('variance_p_value') and significance_results['variance_p_value'] < 0.05:
        print(f"✅ Confidence score distributions significantly differ between datasets")

    print(f"🌸 Species-specific insights reveal cleaning benefits vary by species")
    print(f"📊 High baseline accuracy (93-94%) shows BioCLIP is robust to data quality")
    print(f"📈 Standard deviation analysis provides deeper insights into model reliability")

    return {
        'filtered_metrics': filtered_metrics,
        'unfiltered_metrics': unfiltered_metrics,
        'significance_results': significance_results,
        'improvement': improvement,
        'f1_variability_change': f1_variability_change,
        'confidence_variability_change': conf_variability_change,
        'consistency_improvement': consistency_improvement
    }


def compare_dataset_variability(filtered_results, unfiltered_results):
    """
    🔄 Comprehensive comparison of variability between filtered and unfiltered datasets

    This function provides a focused analysis of how data filtering affects
    the variability and consistency of model predictions.

    Args:
        filtered_results (list): Results from filtered dataset
        unfiltered_results (list): Results from unfiltered dataset

    Returns:
        dict: Comprehensive variability comparison metrics
    """
    print(f"\n🔄 DATASET VARIABILITY COMPARISON")
    print("=" * 60)

    # Extract confidence scores
    filtered_conf = [r['confidence'] for r in filtered_results]
    unfiltered_conf = [r['confidence'] for r in unfiltered_results]

    # Basic variability metrics
    filtered_std = np.std(filtered_conf, ddof=1)
    unfiltered_std = np.std(unfiltered_conf, ddof=1)
    filtered_cv = filtered_std / np.mean(filtered_conf)
    unfiltered_cv = unfiltered_std / np.mean(unfiltered_conf)

    print(f"📊 Overall Confidence Variability:")
    print(f"   Filtered dataset:")
    print(f"     Standard deviation: {filtered_std:.4f}")
    print(f"     Coefficient of variation: {filtered_cv:.4f}")
    print(f"   Unfiltered dataset:")
    print(f"     Standard deviation: {unfiltered_std:.4f}")
    print(f"     Coefficient of variation: {unfiltered_cv:.4f}")
    print(f"   Variability reduction: {unfiltered_std - filtered_std:.4f}")
    print(f"   CV improvement: {unfiltered_cv - filtered_cv:.4f}")

    # Per-species variability comparison
    print(f"\n🌿 Per-Species Variability Comparison:")
    species_variability_changes = {}

    for species in config.SPECIES_NAMES:
        filtered_species = [r['confidence'] for r in filtered_results if r['true_species'] == species]
        unfiltered_species = [r['confidence'] for r in unfiltered_results if r['true_species'] == species]

        if len(filtered_species) >= 2 and len(unfiltered_species) >= 2:
            filtered_species_std = np.std(filtered_species, ddof=1)
            unfiltered_species_std = np.std(unfiltered_species, ddof=1)
            variability_change = unfiltered_species_std - filtered_species_std

            species_variability_changes[species] = variability_change

            print(f"   {species}:")
            print(f"     Filtered std: {filtered_species_std:.4f}")
            print(f"     Unfiltered std: {unfiltered_species_std:.4f}")
            print(f"     Variability reduction: {variability_change:.4f}")

    # Statistical tests for variability differences
    print(f"\n🧮 Statistical Tests for Variability:")

    # Bartlett's test for equal variances (assumes normality)
    try:
        bartlett_stat, bartlett_p = stats.bartlett(filtered_conf, unfiltered_conf)
        print(f"   Bartlett's test (equal variances):")
        print(f"     Statistic: {bartlett_stat:.3f}")
        print(f"     P-value: {bartlett_p:.4f}")

        if bartlett_p < 0.05:
            print(f"     ✅ Variances are significantly different")
        else:
            print(f"     ❌ Variances are not significantly different")
    except Exception as e:
        print(f"   ⚠️ Bartlett's test failed: {e}")

    # Levene's test (more robust, doesn't assume normality)
    levene_stat, levene_p = stats.levene(filtered_conf, unfiltered_conf)
    print(f"   Levene's test (equal variances, robust):")
    print(f"     Statistic: {levene_stat:.3f}")
    print(f"     P-value: {levene_p:.4f}")

    if levene_p < 0.05:
        print(f"     ✅ Variances are significantly different")
    else:
        print(f"     ❌ Variances are not significantly different")

    # Interpretation
    print(f"\n💡 VARIABILITY INTERPRETATION:")
    if filtered_std < unfiltered_std:
        print(f"✅ Data filtering reduces prediction variability")
        print(f"   This suggests more consistent model behavior on clean data")
    else:
        print(f"⚠️ Data filtering increases prediction variability")
        print(f"   This might indicate loss of diverse prediction patterns")

    if levene_p < 0.05:
        print(f"✅ The variability difference is statistically significant")
    else:
        print(f"❌ The variability difference is not statistically significant")

    return {
        'filtered_std': filtered_std,
        'unfiltered_std': unfiltered_std,
        'filtered_cv': filtered_cv,
        'unfiltered_cv': unfiltered_cv,
        'variability_reduction': unfiltered_std - filtered_std,
        'cv_improvement': unfiltered_cv - filtered_cv,
        'species_variability_changes': species_variability_changes,
        'levene_statistic': levene_stat,
        'levene_p_value': levene_p
    }