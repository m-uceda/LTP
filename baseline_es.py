from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils_es import load_and_split_dataset, evaluate_performance
from exploratory_data_analysis import preprocess_es_data, get_mapping
from datasets import ClassLabel


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
# Split data in train, test
train_dataset, test_dataset = load_and_split_dataset("guillermoruiz/MexEmojis")
train_dataset = train_dataset.rename_column("text", "sentence")
test_dataset = test_dataset.rename_column("text", "sentence")

print(' --------- Train size:', len(train_dataset['sentence']))
print(' --------- Test size:', len(test_dataset['sentence']))

es_mapping = "es_mapping.txt"
emoji_mapping = get_mapping(es_mapping)

valid_labels = [emoji_mapping[i][1] for i in range(len(emoji_mapping))]
class_label = ClassLabel(names=valid_labels)

train_dataset = train_dataset.cast_column("label", class_label)
test_dataset = test_dataset.cast_column("label", class_label)

train_es_without_emojis, train_es_with_emojis = preprocess_es_data(train_dataset)
test_es_without_emojis, test_es_with_emojis = preprocess_es_data(test_dataset)

###### WITH EMOJIS
# Extract text and labels
train_texts = [example["sentence"] for example in train_es_with_emojis]
train_labels = [example["label"] for example in train_es_with_emojis]
test_texts = [example["sentence"] for example in test_es_with_emojis]
test_labels = [example["label"] for example in test_es_with_emojis]

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
metrics = evaluate_performance(test_labels, predictions,save_path="confusion_matrix_ES_baseline_with.png")
print('metrics WITH', metrics)

###### WITHOUT EMOJIS
# Extract text and labels
train_texts = [example["sentence"] for example in train_es_without_emojis]
train_labels = [example["label"] for example in train_es_without_emojis]
test_texts = [example["sentence"] for example in test_es_without_emojis]
test_labels = [example["label"] for example in test_es_without_emojis]

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
metrics = evaluate_performance(test_labels, predictions,save_path="confusion_matrix_ES_baseline_without.png")
print('Metrics WITHOUT', metrics)