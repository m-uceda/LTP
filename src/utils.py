from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoConfig

from datasets import Dataset, load_dataset, ClassLabel
from typing import Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def calculate_class_distribution(dataset, label_column="label", num_classes=20):
    """
    Calculate the class distribution in a dataset.

    Args:
        dataset: A HuggingFace Dataset object or any iterable with label data.
        label_column (str): The name of the column containing the labels.
        num_classes (int): The total number of classes.

    Returns:
        list: A list of length `num_classes` where each element represents 
              the number of occurrences of the corresponding label.
    """
    # Extract the labels from the dataset
    labels = dataset[label_column]
    
    # Count the occurrences of each label
    label_counts = Counter(labels)
    
    # Create a list with counts for each class (0 to num_classes - 1)
    class_distribution = [label_counts.get(i, 0) for i in range(num_classes)]
    
    return class_distribution

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # Compute class weights based on dataset distribution
        #class_distribution = [
        #    21.65, 10.51, 10.31, 5.38, 5.04, 4.72, 4.36, 3.67, 3.38, 3.30, 
        #    3.27, 3.08, 2.79, 2.74, 2.73, 2.72, 2.66, 2.59, 2.58, 2.50
        #]

        # Example Usage
        # Assuming `train_dataset` is a HuggingFace Dataset object with a 'label' column
        class_distribution = calculate_class_distribution(self.train_dataset)

        total = sum(class_distribution)
        #class_weights = [total / freq for freq in class_distribution]
        class_weights = [1 / freq for freq in class_distribution]

        # Normalize weights to avoid instability
        max_weight = max(class_weights)
        class_weights = [weight / max_weight for weight in class_weights]

        # Convert to tensor and ensure it matches model device
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(model.device)


        # Define loss function with class weights
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def load_model_and_tokenizer(model_name: str) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Load a pre-trained model and tokenizer.

    Args:
        model_name (str): The name of the pre-trained model to load.

    Returns:
        Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
            - tokenizer (AutoTokenizer): The tokenizer for the specified model.
            - model (AutoModelForSequenceClassification): The model to load.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, num_labels=20)
    config = AutoConfig.from_pretrained(model_name, num_labels=20)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)
    return tokenizer, model

def load_and_split_dataset(dataset_name: str) -> Tuple[Dataset, Dataset]:
    """
    Load a dataset and split it into training and testing datasets.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        Tuple[Dataset, Dataset]:
            - train_dataset (Dataset): The training dataset.
            - test_dataset (Dataset): The testing dataset.
    """
    dataset = load_dataset(dataset_name)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    return train_dataset, test_dataset

def tokenize(dataset: Dataset, tokenizer: AutoTokenizer) -> Tuple[Dataset, Dataset]:
    """
    Tokenize a dataset and split into train and validation sets.

    Args:
        dataset (Dataset): The dataset to tokenize.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenizing.

    Returns:
        Tuple[Dataset, Dataset]:
            - train_dataset (Dataset): Tokenized training dataset.
            - validate_dataset (Dataset): Tokenized validation dataset.
    """
    
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    # Convert label column to ClassLabel
    if not isinstance(dataset.features["label"], ClassLabel):
        dataset = dataset.cast_column("label", ClassLabel(num_classes=20))

    # Stratified split
    split = dataset.train_test_split(test_size=0.2, stratify_by_column='label')

    #split = dataset.train_test_split(test_size=0.2)
    #split = dataset.train_test_split(test_size=50000 / len(dataset['sentence']))

    train_dataset = split["train"].map(preprocess_function, batched=True)
    validate_dataset = split["test"].map(preprocess_function, batched=True)

    return train_dataset, validate_dataset

def get_trainer(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    tokenized_train: Dataset,
    tokenized_validate: Dataset
) -> Trainer:
    """
    Set up a Trainer for fine-tuning a model.

    Args:
        model (AutoModelForSequenceClassification): The pre-trained model to fine-tune.
        tokenized_train (Dataset): The tokenized training dataset.
        tokenized_validate (Dataset): The tokenized validation dataset.

    Returns:
        Trainer: A Trainer object.
    """
    training_args = TrainingArguments(
        output_dir="./results",          # Directory for model outputs
        evaluation_strategy="epoch",    # Evaluate at the end of each epoch
        learning_rate=2e-5,               # Learning rate
        per_device_train_batch_size=8,    # Batch size for training
        per_device_eval_batch_size=8,     # Batch size for evaluation
        num_train_epochs=3,               # Number of training epochs
        weight_decay=0.01                 # Weight decay for optimization
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validate,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    return trainer

def evaluate_performance(model, tokenizer, test_dataset) -> dict:
    """
    Evaluate the performance of a model on a test dataset using multiple metrics and plot a confusion matrix.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to preprocess the input data.
        test_dataset: The test dataset.

    Returns:
        dict: A dictionary containing the computed metrics and their values.
    """
    
    all_labels = []
    all_predictions = []

    for data in test_dataset:
        input_string = data['sentence']
        label = data['label']

        inputs = tokenizer(input_string, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_label = torch.argmax(logits, dim=1).item()

        all_labels.append(label)
        all_predictions.append(predicted_label)

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
    plot_confusion_matrix(all_labels, all_predictions, class_names, save_path="confusion_matrix.png")

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