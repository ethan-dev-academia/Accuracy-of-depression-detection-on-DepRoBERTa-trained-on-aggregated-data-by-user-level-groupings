import json
import numpy as np

# Load data
with open('model_comparison_results/comparison_results_summary.json', 'r') as f:
    aggregated_data = json.load(f)

with open('single_message_comparison_results/single_message_comparison_summary.json', 'r') as f:
    individual_data = json.load(f)

print("="*80)
print("DEBUGGING ERROR BARS")
print("="*80)

print("\n=== AGGREGATED DATA (n=3419) ===")
metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
fine_tuned_metrics = [
    aggregated_data['your_model']['accuracy'],
    aggregated_data['your_model']['f1_score'],
    aggregated_data['your_model']['precision'],
    aggregated_data['your_model']['recall']
]
default_metrics = [
    aggregated_data['default_model_binary']['accuracy'],
    aggregated_data['default_model_binary']['f1_score'],
    aggregated_data['default_model_binary']['precision'],
    aggregated_data['default_model_binary']['recall']
]

n_samples = 3419
print(f"\nSample size: {n_samples}")

print("\nFine-tuned model metrics:")
fine_tuned_errors = []
for m, name in zip(fine_tuned_metrics, metrics):
    err = np.sqrt(m * (1 - m) / n_samples)
    fine_tuned_errors.append(err)
    print(f"  {name}: {m:.4f}")
    print(f"    Error (sqrt(p*(1-p)/n)): {err:.8f}")
    print(f"    Range: [{m-err:.6f}, {m+err:.6f}]")
    print(f"    Error as % of metric: {(err/m)*100:.4f}%")

print("\nDefault model metrics:")
default_errors = []
for m, name in zip(default_metrics, metrics):
    err = np.sqrt(m * (1 - m) / n_samples)
    default_errors.append(err)
    print(f"  {name}: {m:.4f}")
    print(f"    Error (sqrt(p*(1-p)/n)): {err:.8f}")
    print(f"    Range: [{m-err:.6f}, {m+err:.6f}]")

# Calculate y_max
max_value_with_error = max(
    max([m + e for m, e in zip(fine_tuned_metrics, fine_tuned_errors)]),
    max([m + e for m, e in zip(default_metrics, default_errors)])
)
y_max = min(1.0, max_value_with_error * 1.15)

print(f"\nY-axis calculation:")
print(f"  Max value with error: {max_value_with_error:.6f}")
print(f"  y_max (with 15% padding, capped at 1.0): {y_max:.6f}")

print("\n" + "="*80)
print("=== INDIVIDUAL MESSAGES (n=10000) ===")
fine_tuned_individual = [
    individual_data['your_model']['accuracy'],
    individual_data['your_model']['f1_score'],
    individual_data['your_model']['precision'],
    individual_data['your_model']['recall']
]
default_individual = [
    individual_data['default_model_binary']['accuracy'],
    individual_data['default_model_binary']['f1_score'],
    individual_data['default_model_binary']['precision'],
    individual_data['default_model_binary']['recall']
]

n_samples_individual = 10000
print(f"\nSample size: {n_samples_individual}")

print("\nFine-tuned model metrics:")
fine_tuned_errors_ind = []
for m, name in zip(fine_tuned_individual, metrics):
    err = np.sqrt(m * (1 - m) / n_samples_individual)
    fine_tuned_errors_ind.append(err)
    print(f"  {name}: {m:.4f}")
    print(f"    Error (sqrt(p*(1-p)/n)): {err:.8f}")
    print(f"    Range: [{m-err:.6f}, {m+err:.6f}]")
    print(f"    Error as % of metric: {(err/m)*100:.4f}%")

# Calculate y_max_ind
max_value_with_error_ind = max(
    max([m + e for m, e in zip(fine_tuned_individual, fine_tuned_errors_ind)]),
    max([m + e for m, e in zip(default_individual, [np.sqrt(m * (1 - m) / n_samples_individual) for m in default_individual])])
)
y_max_ind = min(1.0, max_value_with_error_ind * 1.15)

print(f"\nY-axis calculation:")
print(f"  Max value with error: {max_value_with_error_ind:.6f}")
print(f"  y_max_ind (with 15% padding, capped at 1.0): {y_max_ind:.6f}")

print("\n" + "="*80)
print("NOTE: If error bars are very small (< 0.01), they may not be visible on the plot.")
print("This is expected when using standard error for sample proportions with large n.")

