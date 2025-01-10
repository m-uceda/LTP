from datasets import load_dataset
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

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
    print(f"Character Count mean = {df['char_count'].mean():.2f}, median = {df['char_count'].median()}, and standard deviation = {df['char_count'].std():.2f}")

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
    plt.boxplot((df_en['char_count'], df_es['char_count']), tick_labels=('English', 'Mexican Spanish'))  # maybe use this to compare the datasets
    plt.title('Character Count')
    plt.xlabel("Corpus")
    plt.ylabel('Number of Characters')
    plt.show()


def word_count_by_class(df, emoji_map, column_name, language):
    print(f"\n----- WORD COUNT {language.upper()}-----")
    df['word_count'] = df[column_name].apply(lambda x: len(x.split()))
    print(f"Range: {df['word_count'].min():.2f} to {df['word_count'].max():.2f}")
    print(f"Word Count mean = {df['word_count'].mean():.2f}, median = {df['word_count'].median()}, and standard deviation = {df['word_count'].std():.2f}")

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
    plt.boxplot((df_en['word_count'], df_es['word_count']), tick_labels=('English', 'Mexican Spanish'))  # maybe use this to compare the datasets
    plt.title('Word Count')
    plt.ylabel('Number of Words')
    plt.xlabel("Corpus")
    plt.show()


if __name__ == '__main__':
    en_mapping = "us_mapping.txt"
    emoji_mapping = get_mapping(en_mapping)

    en_data_path = "Karim-Gamal/SemEval-2018-Task-2-english-emojis"
    es_data_path = "guillermoruiz/MexEmojis"

    en_dataframe = load_data(en_data_path)
    es_dataframe = load_data(es_data_path)

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