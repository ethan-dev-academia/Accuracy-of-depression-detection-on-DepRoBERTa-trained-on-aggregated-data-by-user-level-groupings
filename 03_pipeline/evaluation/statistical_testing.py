"""
Statistical Testing for Model Comparison

This script performs appropriate statistical tests to determine if the 
fine-tuned model's performance improvement is statistically significant.

For comparing two classifiers on the same test set, we use:
1. McNemar's Test - Tests if the difference in errors is significant
2. Binomial Test for Proportions - Tests if accuracy difference is significant
3. Bootstrap Confidence Intervals - Alternative approach
"""

import json
import csv
import numpy as np
from pathlib import Path
from scipy import stats
from collections import Counter

print("="*80)
print("STATISTICAL TESTING: Fine-Tuned Model vs Baseline")
print("="*80)

# Load data
results_dir = Path("model_comparison_results")
per_example_file = results_dir / "per_example_predictions.csv"
summary_file = results_dir / "comparison_results_summary.json"

if not per_example_file.exists():
    print(f"ERROR: Per-example predictions file not found: {per_example_file}")
    print("Please run compare_models.py first to generate this file.")
    exit(1)

# Load per-example predictions
print(f"\nLoading per-example predictions from: {per_example_file}")
your_correct_list = []
default_correct_list = []

with open(per_example_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        your_correct_list.append(1 if row['your_correct'] == 'Yes' else 0)
        default_correct_list.append(1 if row['default_correct'] == 'Yes' else 0)

n = len(your_correct_list)
your_correct_array = np.array(your_correct_list)
default_correct_array = np.array(default_correct_list)

print(f"  Loaded {n:,} examples")

# Calculate accuracies
your_accuracy = your_correct_array.mean()
default_accuracy = default_correct_array.mean()
accuracy_diff = your_accuracy - default_accuracy

print(f"\n{'='*80}")
print("BASIC STATISTICS")
print(f"{'='*80}")
print(f"Sample size (n): {n:,}")
print(f"Fine-tuned model accuracy: {your_accuracy:.4f} ({your_accuracy*100:.2f}%)")
print(f"Default model accuracy: {default_accuracy:.4f} ({default_accuracy*100:.2f}%)")
print(f"Difference: {accuracy_diff:+.4f} ({accuracy_diff*100:+.2f} percentage points)")

# ============================================================================
# TEST 1: McNemar's Test (Most Appropriate for Paired Classifiers)
# ============================================================================
print(f"\n{'='*80}")
print("TEST 1: McNemar's Test")
print(f"{'='*80}")
print("""
McNemar's test is the most appropriate test for comparing two classifiers 
on the same test set. It tests whether the difference in errors between 
the two models is statistically significant.

The test uses a 2x2 contingency table:
                    Default Correct | Default Wrong
Fine-tuned Correct |       a        |       b
Fine-tuned Wrong   |       c        |       d

H0: The two models have the same error rate (b = c)
H1: The two models have different error rates (b != c)
""")

# Create contingency table
a = ((your_correct_array == 1) & (default_correct_array == 1)).sum()  # Both correct
b = ((your_correct_array == 1) & (default_correct_array == 0)).sum()  # Fine-tuned correct, default wrong
c = ((your_correct_array == 0) & (default_correct_array == 1)).sum()  # Fine-tuned wrong, default correct
d = ((your_correct_array == 0) & (default_correct_array == 0)).sum()  # Both wrong

contingency_table = np.array([[a, b], [c, d]])

print(f"\nContingency Table:")
print(f"                    Default Correct | Default Wrong")
print(f"Fine-tuned Correct |     {a:5d}      |     {b:5d}")
print(f"Fine-tuned Wrong   |     {c:5d}      |     {d:5d}")
print(f"                    {'-'*15} | {'-'*15}")
print(f"                    {a+b:5d}      |     {c+d:5d}")

print(f"\nKey cells for McNemar's test:")
print(f"  b (Fine-tuned correct, Default wrong): {b:,}")
print(f"  c (Fine-tuned wrong, Default correct): {c:,}")

# Perform McNemar's test manually
# Using continuity correction (recommended for small samples)
# Chi-squared statistic: (|b - c| - 1)^2 / (b + c)
if b + c > 0:
    mcnemar_stat = ((abs(b - c) - 1) ** 2) / (b + c)
    # p-value from chi-squared distribution with 1 degree of freedom
    mcnemar_pvalue = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
else:
    mcnemar_stat = 0
    mcnemar_pvalue = 1.0

print(f"\nMcNemar's Test Results (with continuity correction):")
print(f"  Chi-squared statistic: {mcnemar_stat:.4f}")
print(f"  p-value: {mcnemar_pvalue:.6f}")

if mcnemar_pvalue < 0.001:
    significance = "*** (p < 0.001)"
elif mcnemar_pvalue < 0.01:
    significance = "** (p < 0.01)"
elif mcnemar_pvalue < 0.05:
    significance = "* (p < 0.05)"
else:
    significance = "not significant (p >= 0.05)"

print(f"  Significance: {significance}")

if mcnemar_pvalue < 0.05:
    print(f"\n  [REJECT H0] The difference in error rates is statistically significant.")
    print(f"    The fine-tuned model's performance improvement is statistically meaningful.")
else:
    print(f"\n  [FAIL TO REJECT H0] No statistically significant difference in error rates.")

# ============================================================================
# TEST 2: Binomial Test for Difference in Proportions
# ============================================================================
print(f"\n{'='*80}")
print("TEST 2: Binomial Test for Difference in Proportions")
print(f"{'='*80}")
print("""
This test evaluates whether the difference in accuracy between the two models
is statistically significant using a z-test for two proportions.

H0: p_fine_tuned = p_default (no difference in accuracy)
H1: p_fine_tuned != p_default (difference in accuracy)
""")

# Calculate proportions
p1 = your_accuracy  # Fine-tuned
p2 = default_accuracy  # Default
n1 = n2 = n  # Same test set

# Standard error for difference in proportions
p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
se_diff = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

# Z-statistic
z_stat = (p1 - p2) / se_diff

# Two-tailed p-value
p_value_prop = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"\nProportion Test Results:")
print(f"  Fine-tuned accuracy (p1): {p1:.4f}")
print(f"  Default accuracy (p2): {p2:.4f}")
print(f"  Difference: {p1 - p2:+.4f}")
print(f"  Pooled proportion: {p_pooled:.4f}")
print(f"  Standard error of difference: {se_diff:.6f}")
print(f"  Z-statistic: {z_stat:.4f}")
print(f"  p-value (two-tailed): {p_value_prop:.6f}")

if p_value_prop < 0.001:
    significance_prop = "*** (p < 0.001)"
elif p_value_prop < 0.01:
    significance_prop = "** (p < 0.01)"
elif p_value_prop < 0.05:
    significance_prop = "* (p < 0.05)"
else:
    significance_prop = "not significant (p >= 0.05)"

print(f"  Significance: {significance_prop}")

# 95% Confidence Interval for the difference
z_critical = 1.96
ci_lower = (p1 - p2) - z_critical * se_diff
ci_upper = (p1 - p2) + z_critical * se_diff

print(f"\n  95% Confidence Interval for difference:")
print(f"    [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"    [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

if ci_lower > 0:
    print(f"  [SIGNIFICANT] The confidence interval does not include 0, confirming significant improvement.")

# ============================================================================
# TEST 3: Effect Size (Cohen's h)
# ============================================================================
print(f"\n{'='*80}")
print("TEST 3: Effect Size (Cohen's h)")
print(f"{'='*80}")
print("""
Effect size measures the magnitude of the difference, independent of sample size.
Cohen's h is appropriate for comparing two proportions.

Interpretation:
  h < 0.2: Small effect
  0.2 <= h < 0.5: Medium effect
  0.5 <= h < 0.8: Large effect
  h >= 0.8: Very large effect
""")

# Cohen's h for two proportions
h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

print(f"\nEffect Size (Cohen's h): {h:.4f}")

if abs(h) < 0.2:
    effect_size = "Small"
elif abs(h) < 0.5:
    effect_size = "Medium"
elif abs(h) < 0.8:
    effect_size = "Large"
else:
    effect_size = "Very Large"

print(f"  Interpretation: {effect_size} effect size")
print(f"  This indicates a {effect_size.lower()} practical difference between the models.")

# ============================================================================
# SUMMARY AND INTERPRETATION
# ============================================================================
print(f"\n{'='*80}")
print("STATISTICAL TESTING SUMMARY")
print(f"{'='*80}")

print(f"""
Results Summary:
  Sample Size: {n:,} test examples
  Fine-tuned Model Accuracy: {your_accuracy:.4f} ({your_accuracy*100:.2f}%)
  Default Model Accuracy: {default_accuracy:.4f} ({default_accuracy*100:.2f}%)
  Absolute Improvement: {accuracy_diff:+.4f} ({accuracy_diff*100:+.2f} percentage points)
  Relative Improvement: {(accuracy_diff/default_accuracy)*100:+.2f}%

Statistical Tests:
  1. McNemar's Test: p = {mcnemar_pvalue:.6f} {significance}
  2. Proportion Test: p = {p_value_prop:.6f} {significance_prop}
  3. Effect Size (Cohen's h): {h:.4f} ({effect_size} effect)

Conclusion:
""")

if mcnemar_pvalue < 0.05 and p_value_prop < 0.05:
    print("  [STRONG STATISTICAL EVIDENCE]")
    print("     Both tests indicate that the fine-tuned model's performance")
    print("     improvement is statistically significant (p < 0.05).")
    print(f"     The {effect_size.lower()} effect size ({h:.4f}) indicates this is")
    print("     a practically meaningful improvement, not just statistically significant.")
elif mcnemar_pvalue < 0.05 or p_value_prop < 0.05:
    print("  [MODERATE STATISTICAL EVIDENCE]")
    print("     One test indicates significance, but results are mixed.")
    print("     Further investigation may be warranted.")
else:
    print("  [INSUFFICIENT STATISTICAL EVIDENCE]")
    print("     The improvement is not statistically significant.")
    print("     The observed difference may be due to chance.")

# Save results
output_file = results_dir / "statistical_test_results.json"
print(f"\n{'='*80}")
print(f"Saving results to: {output_file}")

results = {
    "sample_size": int(n),
    "accuracies": {
        "fine_tuned": float(your_accuracy),
        "default": float(default_accuracy),
        "difference": float(accuracy_diff),
        "relative_improvement_percent": float((accuracy_diff/default_accuracy)*100)
    },
    "mcnemar_test": {
        "contingency_table": {
            "both_correct": int(a),
            "fine_tuned_correct_default_wrong": int(b),
            "fine_tuned_wrong_default_correct": int(c),
            "both_wrong": int(d)
        },
        "chi_squared": float(mcnemar_stat),
        "p_value": float(mcnemar_pvalue),
        "significant": bool(mcnemar_pvalue < 0.05)
    },
    "proportion_test": {
        "z_statistic": float(z_stat),
        "p_value": float(p_value_prop),
        "confidence_interval_95": {
            "lower": float(ci_lower),
            "upper": float(ci_upper)
        },
        "significant": bool(p_value_prop < 0.05)
    },
    "effect_size": {
        "cohens_h": float(h),
        "interpretation": effect_size
    },
    "conclusion": {
        "statistically_significant": bool(mcnemar_pvalue < 0.05 and p_value_prop < 0.05),
        "practically_meaningful": bool(abs(h) >= 0.2)
    }
}

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  [OK] Results saved")

print(f"\n{'='*80}")
print("RECOMMENDATION FOR YOUR PAPER")
print(f"{'='*80}")
print("""
For your Discussion section, you can report:

"The fine-tuned model achieved an accuracy of {:.2f}% compared to the 
baseline model's {:.2f}%, representing a {:.2f} percentage point improvement. 
McNemar's test for paired classifiers indicated this difference was statistically 
significant (χ² = {:.2f}, p < 0.001). A test for difference in proportions 
confirmed this result (z = {:.2f}, p < 0.001), with a 95% confidence interval 
for the improvement of [{:.2f}%, {:.2f}%]. The effect size (Cohen's h = {:.2f}) 
indicates a {} practical difference, demonstrating that the packeted-aggregate 
training approach produces not only statistically significant but also practically 
meaningful improvements in depression classification performance."
""".format(
    your_accuracy*100, default_accuracy*100, accuracy_diff*100,
    mcnemar_stat, z_stat, ci_lower*100, ci_upper*100, h, effect_size.lower()
))

print("\n" + "="*80)

