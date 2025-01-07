from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, EvalPrediction
from datasets import Dataset, load_dataset
from evaluate import load
from peft import LoraConfig, TaskType, get_peft_model
from typing import List, Tuple, Dict, Callable, Optional, Any
import numpy as np

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoTokenizer, AutoModelForMaskedLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return model, tokenizer

def load_and_split_dataset(
        dataset_name: str, 
        test_size: float = .2, 
        validate_size: float = .5
)-> Tuple[Dataset, Dataset, Dataset]:
    """
    Loads a dataset using its name and performs a train-test split with a specified split.

    Args:
        dataset_name (str): The name of the dataset to load.

        test_size (float, default=.2): Size of the test set (in percent).
        validate_size (float, default=.5): Size of the validation set (percent of test set used).

    Returns:
        (Tuple[Dataset, Dataset, Dataset]):
            - (Dataset): The training dataset.
            - (Dataset): The test dataset
            - (Dataset): The validate dataset.
    """
    dataset = load_dataset(dataset_name)
    
    if "train" in dataset and "test" in dataset:
        train_set = dataset["train"]
        test_set = dataset["test"]
    else:
        train_test_split = dataset.train_test_split(test_size=test_size, seed=42)
        train_set = train_test_split["train"]
        test_set = train_test_split["test"]

    test_validate_split = test_set.train_test_split(test_size=validate_size, seed=42)
    test_set = test_validate_split["train"]
    validate_set = test_validate_split["test"]

    return train_set, test_set, validate_set

def tokenize_dataset(
        dataset: Dataset, 
        tokenizer: AutoTokenizer
) -> Dataset:
    """
    Tokenizes a dataset using the provided tokenizer.

    Args:
        dataset (Dataset): The dataset containing the raw text examples to be tokenized.
        tokenizer (AutoTokenizer): The tokenizer used for encoding the text.

    Returns:
        (Dataset): The tokenized dataset.
    """
    # Truncation True to ensure that tokens length fit the model's constraints
    # Padding to ensure all tokenized sequences are of of same length
    # Batched True to process multiple examples at once, enabling faster computation
    tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['sentence'], truncation=True, padding=True), batched=True)
    return tokenized_dataset

def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, Any]:
    """
    Computes accuracy for the model predictions using the `evaluate` library.

    Args:
        eval_preds (EvalPrediction): The predictions from the model evaluation.

    Returns:
        (Dict[str, Any]): A dictionary containing the accuracy metrics.
    """
    # Load the accuracy metric
    accuracy_metric = load("accuracy")

    preds, labels = eval_preds.predictions, eval_preds.label_ids

    # Assuming `preds` are logits, apply argmax to get predicted class indices
    logits = preds[0]
    predictions = np.argmax(logits, axis=-1)

    # Compute accuracy using the loaded metric
    results = accuracy_metric.compute(predictions=predictions.flatten(), references=labels.flatten())
    
    return results

def fine_tune_model_lora(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    eval_dataset: Dataset,
    training_arguments: TrainingArguments,
    lora_config: LoraConfig,
) -> None:
    """
    Fine-tune a translation model using the Low-Rank Adaptation method (LoRA).

    Args:
        model (AutoModelForMaskedLM): The pre-trained model to fine-tune.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        dataset (Dataset): The tokenized training dataset to use for fine-tuning.
        eval_dataset (Dataset): The tokenized evaluation dataset to use for fine-tuning.
        training_arguments (TrainingArguments): The arguments for the training process.
        lora_config (LoraConfig): The configuration for LoRA adaptation.
    """
    model = get_peft_model(model, lora_config)

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        eval_dataset=eval_dataset
    )
    # ,compute_metrics=lambda p: compute_metrics(p, tokenizer)

    trainer.train()
    trainer.save_model("./lora_fine_tuned_model")

def fine_tune_model_full(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    eval_dataset: Dataset,
    training_arguments: TrainingArguments
) -> None:
    """
    Fine-tune a translation model. This is a full parameter fine-tuning.

    Args:
        model (AutoModelForMaskedLM): The pre-trained model to fine-tune.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        dataset (Dataset): The tokenized training dataset to use for fine-tuning.
        eval_dataset (Dataset): The tokenized evaluation dataset to use for fine-tuning.
        training_arguments (TrainingArguments): The arguments for the training process.
    """
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        eval_dataset=eval_dataset
    )

     # ,compute_metrics=lambda p: compute_metrics(p, tokenizer)

    trainer.train()
    trainer.save_model("./full_fine_tuned_model")


def fine_tune_model(
    model: AutoModelForMaskedLM, 
    tokenizer: AutoTokenizer, 
    tokenized_train: Dataset,
    tokenized_test: Dataset
) -> None:
    """
    Fine tune model using both full and lora methods.

    Args:
        model (AutoModelForMaskedLM): The pre-trained model to fine-tune.

        tokenizer (AutoTokenizer): The tokenizer associated with the model.

        tokenized_train (Dataset): The tokenized training dataset to use for fine-tuning.

        tokenized_test (Dataset): The tokenized test dataset to use for fine-tuning.
    """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.4,
        task_type=TaskType.SEQ_CLS,
        target_modules="all-linear"
    )

    training_args = TrainingArguments(
        output_dir='./results', 
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=1
    )

    fine_tune_model_full(
        model=model,
        tokenizer=tokenizer,
        dataset=tokenized_train,
        eval_dataset=tokenized_test,
        training_arguments=training_args
    )

    fine_tune_model_lora(
        model=model,
        tokenizer=tokenizer,
        dataset=tokenized_train,
        eval_dataset=tokenized_test,
        training_arguments=training_args,
        lora_config=lora_config
    )

def evaluate_performance(fine_tuned_model_lora, tokenizer, tokenized_test) -> Dict[str, Any]:
    """
    Calculates the accuracy of the model on the test set.

    Args:
        fine_tuned_model_lora (PreTrainedModel): The fine-tuned model with LoRA applied.
        tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the data.
        tokenized_test (Dataset): The tokenized test dataset.

    Returns:
        dict: A dictionary containing the evaluation results, including accuracy.
    """
    trainer = Trainer(
        model=fine_tuned_model_lora,
        eval_dataset=tokenized_test,
        compute_metrics=lambda p: compute_metrics(p, tokenizer)
    )

    results = trainer.evaluate()
    return results