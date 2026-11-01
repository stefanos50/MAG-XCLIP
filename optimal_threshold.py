import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix

# Load predictions and ground truth
data = np.load("./preds.npy")  # Shape (N, 2), second column is probability for positive class
ground_truth = np.load("./true_labels.npy")

# Find best threshold
best_f1 = 0
best_threshold = 0
best_conf_matrix = None

thresholds = np.linspace(0, 1, 500)
for threshold in thresholds:
    predictions = (data > threshold).astype(int)
    f1 = f1_score(ground_truth, predictions, average='macro')
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_conf_matrix = confusion_matrix(ground_truth, predictions)

# Convert confusion matrix to accuracy (normalize by row sum)
conf_matrix_accuracy = best_conf_matrix.astype('float') / best_conf_matrix.sum(axis=1)[:, np.newaxis]

# Plot the confusion matrix with fixed resolution
plt.figure(figsize=(10, 8), dpi=120)  # dpi sets resolution (120 is typical; increase for higher quality)
sns.heatmap(conf_matrix_accuracy, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=['Normal','Abnormal'], yticklabels=['Normal','Abnormal'],
            annot_kws={"size": 30, "weight": "bold"},  # Numbers inside heatmap
            linewidths=1, linecolor='black')

plt.xlabel("Predicted Label", fontsize=28, fontweight='bold')
plt.ylabel("True Label", fontsize=28, fontweight='bold')
plt.xticks(fontsize=24, fontweight='bold')
plt.yticks(fontsize=24, fontweight='bold')
plt.title(f"Normalized Confusion Matrix\nBest Threshold={best_threshold:.3f}, F1={best_f1:.3f}",
          fontsize=32, fontweight='bold')

plt.tight_layout()  # Ensures nothing is cut off
plt.show()

# Print best results
print(f"Best Threshold: {best_threshold:.4f}")
print(f"Best F1-score: {best_f1:.4f}")
print("Confusion Matrix (Normalized by Row):")
print(conf_matrix_accuracy)

