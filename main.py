from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils import load_model_and_tokenizer, load_and_split_dataset, get_trainer, tokenize, evaluate_performance


def main():
    """The main method of this script."""

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(
        model_name="google-bert/bert-base-multilingual-cased"
    )

    # Split data in train, test
    train_dataset, test_dataset = load_and_split_dataset("Karim-Gamal/SemEval-2018-Task-2-english-emojis")
    train_dataset = train_dataset

    # Tokenize data and extract validation set
    tokenized_train, tokenized_validate = tokenize(train_dataset, tokenizer)

    # Fine tune model (or retrieve by commenting next 4 lines)
    trainer = get_trainer(model, tokenizer, tokenized_train, tokenized_validate)
    trainer.train()
    model.save_pretrained("./fine-tuned-model-test")
    tokenizer.save_pretrained("./fine-tuned-model-test")

    fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("./fine-tuned-model-test")
    tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model-test")

    # Evaluate performance on test set
    performance = evaluate_performance(fine_tuned_model, tokenizer, test_dataset)

    print(performance)

if __name__ == "__main__":
    main()