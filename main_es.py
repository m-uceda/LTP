from src.utils_es import load_model_and_tokenizer, load_and_split_dataset, get_trainer, tokenize, evaluate_performance
from exploratory_data_analysis import preprocess_es_data, get_mapping
from datasets import ClassLabel

def main():
    """The main method of this script."""

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(
        model_name="google-bert/bert-base-multilingual-cased"
    )

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
    # Tokenize data and extract validation set
    tokenized_train, tokenized_validate = tokenize(train_es_with_emojis, tokenizer)

    # Fine tune model
    trainer = get_trainer(model, tokenizer, tokenized_train, tokenized_validate)
    trainer.train()
    
    # Evaluate performance on test set
    performance = evaluate_performance(trainer, tokenizer, test_es_with_emojis, save_path="confusion_matrix_es_with.png")

    print('Performance Es with emojis:', performance)

    ###### WITHOUT EMOJIS
    # Tokenize data and extract validation set
    tokenized_train, tokenized_validate = tokenize(train_es_without_emojis, tokenizer)

    # Fine tune model
    trainer = get_trainer(model, tokenizer, tokenized_train, tokenized_validate)
    trainer.train()

    # Evaluate performance on test set
    performance = evaluate_performance(trainer, tokenizer, test_es_without_emojis, save_path="confusion_matrix_es_without.png")

    print('Performance Es without emojis:', performance)


if __name__ == "__main__":
    main()