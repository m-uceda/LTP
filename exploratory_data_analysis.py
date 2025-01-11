from datasets import load_dataset
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import re

# I can't currently figure out how to get a font that actually shows the emojis nicely.
rcParams['font.family'] = 'Segoe UI Emoji'


def get_mapping(file_path):
    file = open(file_path, "r", encoding="utf-8").readlines()
    mapping = []
    for line in file:
        mapping.append(line.split())
    return np.array(mapping)


def map_emojis(mapping, label):
    if isinstance(label, (int, np.integer)):
        return mapping[label][1]
    else:
        return label


def load_data(file_path):
    data = load_dataset(file_path)
    return data


def change_to_pandas(data):
    df = pd.DataFrame(data['train'])
    return df


def basic_info(df, language):
    print(f"\n----- BASIC INFO {language.upper()} -----")
    print(df.info)
    print(df.columns)


def class_distribution(df, emoji_map, language):
    print(f"\n----- CLASS DISTRIBUTION {language.upper()} -----")

    # Prints the percentages of the class distribution
    class_dist = df["label"].value_counts(normalize=True)
    for label, proportion in class_dist.items():
        print(f"{map_emojis(emoji_map, label)}: {proportion * 100:.2f}%")

    # Plots the instance count of each class
    class_count = df["label"].value_counts()
    em_labels = [map_emojis(emoji_map, label) for label in class_count.index]
    plt.barh(em_labels, class_count.values)
    plt.title(f'Class Distribution, {language}')
    plt.ylabel('Class')
    plt.xlabel('Instance Count')
    plt.show()


def character_count_by_class(df, emoji_map, column_name, language):
    print(f"\n----- CHARACTER COUNT {language.upper()}-----")
    df['char_count'] = df[column_name].apply(len)
    print(f"Range: {df['char_count'].min():.2f} to {df['char_count'].max():.2f}")
    print(
        f"Character Count mean = {df['char_count'].mean():.2f}, median = {df['char_count'].median()}, and standard deviation = {df['char_count'].std():.2f}")

    df_sorted = df.sort_values(by='label')
    groups = [group['char_count'].values for name, group in df_sorted.groupby('label')]
    sorted_labels = sorted(df['label'].unique())
    em_labels = [map_emojis(emoji_map, label) for label in sorted_labels]

    plt.boxplot(groups, tick_labels=em_labels)
    plt.title(f'Character Count by Class, {language}')
    plt.xlabel('Class')
    plt.ylabel('Number of Characters')
    plt.show()


def character_count_comparison(df_en, df_es):
    plt.boxplot((df_en['char_count'], df_es['char_count']),
                tick_labels=('English', 'Mexican Spanish'))  # maybe use this to compare the datasets
    plt.title('Character Count')
    plt.xlabel("Corpus")
    plt.ylabel('Number of Characters')
    plt.show()


def word_count_by_class(df, emoji_map, column_name, language):
    print(f"\n----- WORD COUNT {language.upper()}-----")
    df['word_count'] = df[column_name].apply(lambda x: len(x.split()))
    print(f"Range: {df['word_count'].min():.2f} to {df['word_count'].max():.2f}")
    print(
        f"Word Count mean = {df['word_count'].mean():.2f}, median = {df['word_count'].median()}, and standard deviation = {df['word_count'].std():.2f}")

    df_sorted = df.sort_values(by='label')
    groups = [group['word_count'].values for name, group in df_sorted.groupby('label')]
    sorted_labels = sorted(df['label'].unique())
    em_labels = [map_emojis(emoji_map, label) for label in sorted_labels]

    plt.boxplot(groups, tick_labels=em_labels)
    plt.title(f'Word Count by Class, {language}')
    plt.xlabel('Class')
    plt.ylabel('Number of Words')
    plt.show()


def word_count_comparison(df_en, df_es):
    plt.boxplot((df_en['word_count'], df_es['word_count']),
                tick_labels=('English', 'Mexican Spanish'))  # maybe use this to compare the datasets
    plt.title('Word Count')
    plt.ylabel('Number of Words')
    plt.xlabel("Corpus")
    plt.show()


def exploratory_analysis(en_dataframe, es_dataframe):
    en_mapping = "us_mapping.txt"
    emoji_mapping = get_mapping(en_mapping)

    basic_info(en_dataframe, 'English')
    basic_info(es_dataframe, 'Spanish')

    class_distribution(en_dataframe, emoji_mapping, 'English')
    class_distribution(es_dataframe, False, 'Mexican Spanish')

    character_count_by_class(en_dataframe, emoji_mapping, 'sentence', 'English')
    character_count_by_class(es_dataframe, False, 'text', 'Mexican Spanish')
    character_count_comparison(en_dataframe, es_dataframe)

    word_count_by_class(en_dataframe, emoji_mapping, 'sentence', 'English')
    word_count_by_class(es_dataframe, False, 'text', 'Mexican Spanish')
    word_count_comparison(en_dataframe, es_dataframe)


def remove_emojis(text):
    # Define the emoji pattern
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def remove_tags(ds):
    # Take away the location tag and take away placeholders for emoijs, URLs, etc.
    ds["text"] = re.sub("^.+ _GEO|_[a-zA-Z]{3}", "", ds["text"])
    # Remove emojis
    ds["text"] = remove_emojis(ds["text"])
    # Make sure there are no double whitespaces or whitespaces in the beginning or end
    ds["text"] = re.sub(" {2,}", " ", ds["text"])
    ds["text"] = re.sub("^ | $", "", ds["text"])
    return ds


def preprocess_es_data(ds):
    print(f"\n----- ORIGINAL MEXICAN SPANISH DATA -----")
    print(f"First Row: {ds['train'][0]}")
    print(f"First Row: {ds['train'][162]}")
    print(f"First Row: {ds['train'][27238]}")
    print(f"First Row: {ds['train'][27283]}")
    print(f"Second Row: {ds['train'][36139]}")

    ds['train'] = ds['train'].map(remove_tags)

    print(f"\n----- MODIFIED MEXICAN SPANISH DATA -----")
    print(f"First Row: {ds['train'][0]}")
    print(f"First Row: {ds['train'][162]}")
    print(f"First Row: {ds['train'][27238]}")
    print(f"First Row: {ds['train'][27283]}")
    print(f"Second Row: {ds['train'][36139]}")
    return ds


if __name__ == '__main__':
    en_data_path = "Karim-Gamal/SemEval-2018-Task-2-english-emojis"
    es_data_path = "guillermoruiz/MexEmojis"

    dataframe_en = change_to_pandas(load_data(en_data_path))
    dataset_es = load_data(es_data_path)
    dataframe_es = change_to_pandas(dataset_es)

    exploratory_analysis(dataframe_en, dataframe_es)
    modified_dataset_es = preprocess_es_data(dataset_es)
    modified_dataframe_es = change_to_pandas(modified_dataset_es)
    exploratory_analysis(dataframe_en, modified_dataframe_es)