from src.utils import (load_model_and_tokenizer, load_and_split_dataset, tokenize_dataset, fine_tune_model, evaluate_performance)
from transformers import AutoTokenizer, AutoModelForMaskedLM


def main():
    """The main method of this script."""
    # Preprocess data
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name="google-bert/bert-base-uncased"
    )

    # Split data in train, test, and validate sets
    train_set, test_set, validate_set = load_and_split_dataset(
        dataset_name="Karim-Gamal/SemEval-2018-Task-2-english-emojis"
    )
    
    # Tokenize data
    tokenized_train = tokenize_dataset(train_set, tokenizer)
    tokenized_test = tokenize_dataset(test_set, tokenizer)
    tokenized_validate = tokenize_dataset(validate_set, tokenizer)

    #performance = {}  # A dict keeping track of the accuracy scores for the model
    #performance = bleu_score(model, tokenizer, tokenized_test_context, performance, 'Context')

    # Fine tune model
    fine_tune_model(model, tokenizer, tokenized_train, tokenized_validate)

    fine_tuned_model_full = AutoModelForMaskedLM.from_pretrained("./full_fine_tuned_model")
    fine_tuned_model_lora = AutoModelForMaskedLM.from_pretrained("./lora_fine_tuned_model")

    # Evaluate performance
    performance_lora = evaluate_performance(fine_tuned_model_lora, tokenizer, tokenized_test)
    performance_full = evaluate_performance(fine_tuned_model_full, tokenizer, tokenized_test)

    print(performance_lora)
    print(performance_full)

if __name__ == "__main__":
    main()