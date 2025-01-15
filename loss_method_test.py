
from src.utils import load_model_and_tokenizer, load_and_split_dataset, get_trainer, tokenize, evaluate_performance, get_subset
import os

def run_loss_test(
        model_name: str, 
        dataset_name: str,
        num_classes: int,
        file_name: str,
        mapping_file: str,
        performance_message: str,
        trainer_type: str,
        spanish_data_prep: str = None
):
    
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(
        model_name=model_name,
        num_labels=num_classes
    )

    # Split data in train, test
    train_dataset, test_dataset = load_and_split_dataset(dataset_name=dataset_name)
    subset_to_train = get_subset(train_dataset, size=900)
    subset_to_test = get_subset(test_dataset, size=100)

    # Tokenize data and extract validation set
    tokenized_train, tokenized_validate = tokenize(
        dataset=subset_to_train,
        tokenizer=tokenizer,
        num_classes=num_classes,
        test_size=100/900)

    # Fine tune model (or retrieve by commenting next 4 lines)
    trainer = get_trainer(model, tokenizer, tokenized_train, tokenized_validate, trainer_type)
    trainer.train()

    folder_path = "confusion matrices"
    os.makedirs(folder_path, exist_ok=True)
    sav_path = os.path.join(folder_path, file_name)

    # Evaluate performance on test set
    performance = evaluate_performance(
        model=model, 
        tokenizer=tokenizer, 
        test_dataset=subset_to_test,
        mapping_file=mapping_file,
        save_path=sav_path)

    # Open the file in append mode ('a') or create it if it doesn't exist
    with open("results_loss_tests.txt", 'a') as file:
        # Add a newline and then the text
        file.write(performance_message)
        file.write('\n')
        file.write(str(performance))
        file.write('\n\n')

def main():
    """The main method of this script."""

    # English (weighted loss)
    run_loss_test(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="Karim-Gamal/SemEval-2018-Task-2-english-emojis",
        num_classes=20,
        file_name="CM_english_subset_weightedloss.png",
        mapping_file="us_mapping.txt",
        performance_message="Performance English subset with weighted loss:",
        trainer_type="weighted"
        )
    
    # English (standard loss)
    run_loss_test(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="Karim-Gamal/SemEval-2018-Task-2-english-emojis",
        num_classes=20,
        file_name="CM_english_subset_standardloss.png",
        mapping_file="us_mapping.txt",
        performance_message="Performance English subset with standard loss:",
        trainer_type="standard"
        )
    
    # Spansih (not preprocessed)
    # Standard loss
    run_loss_test(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="guillermoruiz/MexEmojis",
        num_classes=12,
        file_name="CM_spanish_notprep_subset_standardloss.png",
        mapping_file="es_mapping.txt",
        performance_message="Performance Spanish subset (not prep) with standard loss:",
        trainer_type="standard",
        spanish_data_prep="not preprocessed"
        )
    
    # Weighted loss
    run_loss_test(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="guillermoruiz/MexEmojis",
        num_classes=12,
        file_name="CM_spanish_notprep_subset_weightedloss.png",
        mapping_file="es_mapping.txt",
        performance_message="Performance Spanish subset (not prep) with weighted loss:",
        trainer_type="weighted",
        spanish_data_prep="not preprocessed"
        )

    # Spansih (preprocessed with emojis)
    # Standard loss
    run_loss_test(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="guillermoruiz/MexEmojis",
        num_classes=12,
        file_name="CM_spanish_emojis_subset_standardloss.png",
        mapping_file="es_mapping.txt",
        performance_message="Performance Spanish subset (with emojis) with standard loss:",
        trainer_type="standard",
        spanish_data_prep="with emojis"
        )

    # Weighted loss
    run_loss_test(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="guillermoruiz/MexEmojis",
        num_classes=12,
        file_name="CM_spanish_emojis_subset_weightedloss.png",
        mapping_file="es_mapping.txt",
        performance_message="Performance Spanish subset (with emojis) with weighted loss:",
        trainer_type="weighted",
        spanish_data_prep="with emojis"
        )

    # Spansih (preprocessed without emojis)
    # Standard loss
    run_loss_test(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="guillermoruiz/MexEmojis",
        num_classes=12,
        file_name="CM_spanish_noemojis_subset_standardloss.png",
        mapping_file="es_mapping.txt",
        performance_message="Performance Spanish subset (no emojis) with standard loss:",
        trainer_type="standard",
        spanish_data_prep="no emojis"
        )

    # Weighted loss
    run_loss_test(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="guillermoruiz/MexEmojis",
        num_classes=12,
        file_name="CM_spanish_noemojis_subset_weightedloss.png",
        mapping_file="es_mapping.txt",
        performance_message="Performance Spanish subset (no emojis) with weighted loss:",
        trainer_type="weighted",
        spanish_data_prep="no emojis"
        )
    

if __name__ == "__main__":
    main()