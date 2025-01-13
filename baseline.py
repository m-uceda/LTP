from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_performance(all_labels, all_predictions, save_path="confusion_matrix.png") -> dict:
    """
    Evaluate the performance of a model on a test dataset using multiple metrics and plot a confusion matrix.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to preprocess the input data.
        test_dataset: The test dataset.

    Returns:
        dict: A dictionary containing the computed metrics and their values.
    """

    # Determine class names from the labels
    n_labels = len(set(all_labels))
    class_names = [str(i) for i in range(n_labels)]

    metrics = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, average="weighted"),
        "recall": recall_score(all_labels, all_predictions, average="weighted"),
        "f1": f1_score(all_labels, all_predictions, average="weighted")
    }

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, class_names, save_path=save_path)

    return metrics

def plot_confusion_matrix(all_labels, all_predictions, class_names, save_path=None):
    """
    Plot a confusion matrix for the model predictions.

    Args:
        all_labels: True labels of the test dataset.
        all_predictions: Predicted labels by the model.
        class_names: List of class names corresponding to the labels.
        save_path: Path to save the confusion matrix image (optional).

    Returns:
        None
    """
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

    plt.show()

# Load dataset
dataset = load_dataset("Karim-Gamal/SemEval-2018-Task-2-english-emojis")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Extract text and labels
train_texts = [example["sentence"] for example in train_dataset]
train_labels = [example["label"] for example in train_dataset]
test_texts = [example["sentence"] for example in test_dataset]
test_labels = [example["label"] for example in test_dataset]

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for simplicity
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Train Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)

# Make predictions
predictions = clf.predict(X_test)

# Evaluate the model
metrics = evaluate_performance(test_labels, predictions,save_path="confusion_matrix_EN_baseline.png")
print(metrics)
