from src.utils import load_train_and_evaluate

def main():
    """The main method of this script."""
    train_subset_size=9000
    test_subset_size=1000

    # English (standard loss)
    load_train_and_evaluate(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="Karim-Gamal/SemEval-2018-Task-2-english-emojis",
        num_classes=20,
        file_name="CM_english_subset_standardloss.png",
        mapping_file="us_mapping.txt",
        performance_message="Performance English subset with standard loss:",
        trainer_type="standard",
        train_subset_size=train_subset_size,
        test_subset_size=test_subset_size
        )
    
    # English (weighted loss)
    load_train_and_evaluate(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="Karim-Gamal/SemEval-2018-Task-2-english-emojis",
        num_classes=20,
        file_name="CM_english_subset_weightedloss.png",
        mapping_file="us_mapping.txt",
        performance_message="Performance English subset with weighted loss:",
        trainer_type="weighted",
        train_subset_size=train_subset_size,
        test_subset_size=test_subset_size
        )

    # Spansih (not preprocessed)
    # Standard loss
    load_train_and_evaluate(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="guillermoruiz/MexEmojis",
        num_classes=12,
        file_name="CM_spanish_notprep_subset_standardloss.png",
        mapping_file="es_mapping.txt",
        performance_message="Performance Spanish subset (not prep) with standard loss:",
        trainer_type="standard",
        train_subset_size=train_subset_size,
        test_subset_size=test_subset_size,
        spanish_data_prep="not preprocessed"
        )
    
    # Weighted loss
    load_train_and_evaluate(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="guillermoruiz/MexEmojis",
        num_classes=12,
        file_name="CM_spanish_notprep_subset_weightedloss.png",
        mapping_file="es_mapping.txt",
        performance_message="Performance Spanish subset (not prep) with weighted loss:",
        trainer_type="weighted",
        train_subset_size=train_subset_size,
        test_subset_size=test_subset_size,
        spanish_data_prep="not preprocessed"
        )
    

if __name__ == "__main__":
    main()