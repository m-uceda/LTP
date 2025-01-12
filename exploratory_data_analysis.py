from datasets import load_dataset
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import re

# I can't currently figure out how to get a font that actually shows the emojis nicely.
rcParams['font.family'] = 'Segoe UI Emoji'


def load_data(file_path):
    data = load_dataset(file_path)
    return data


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


def character_count_by_class(df, emoji_map, language):
    print(f"\n----- CHARACTER COUNT {language.upper()}-----")
    df['char_count'] = df['sentence'].apply(len)
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


def word_count_by_class(df, emoji_map, language):
    print(f"\n----- WORD COUNT {language.upper()}-----")
    df['word_count'] = df['sentence'].apply(lambda x: len(x.split()))
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


def plot_comparison(df_en, df_es_1, df_es_2, df_es_3, label, type):
    plt.boxplot((df_en[label], df_es_1[label], df_es_2[label], df_es_3[label]),
                tick_labels=('English', 'Mexican Spanish Unprocessed', "Mexican Spanish without Emojis",
                             "Mexican Spanish with Emojis"))
    plt.title(f"{type} Count")
    plt.ylabel(f"Number of {type}s")
    plt.xlabel(f"Corpus")
    plt.show()


def individual_analysis(df, version, mapping):
    basic_info(df, version)
    class_distribution(df, mapping, version)
    character_count_by_class(df, mapping, version)
    word_count_by_class(df, mapping, version)


def exploratory_analysis(en_df, es_df_original, es_df_without_emojis, es_df_with_emojis):
    en_mapping = "us_mapping.txt"
    emoji_mapping = get_mapping(en_mapping)

    individual_analysis(en_df, "English", emoji_mapping)
    individual_analysis(es_df_original, "Mexican Spanish Unprocessed", False)
    individual_analysis(es_df_without_emojis, "Mexican Spanish without Emojis", False)
    individual_analysis(es_df_with_emojis, "Mexican Spanish with Emojis", False)

    plot_comparison(en_df, es_df_original, es_df_without_emojis, es_df_with_emojis, 'word_count', "Word")
    plot_comparison(en_df, es_df_original, es_df_without_emojis, es_df_with_emojis, 'char_count', "Character")


def remove_emojis_regex(text):
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


def remove_emojis(ds):
    ds["sentence"] = remove_emojis_regex(ds["sentence"])
    return ds


def remove_tags(ds):
    # Take away the location tag and take away placeholders for emoijs, URLs, etc.
    ds["sentence"] = re.sub("^.+ _GEO|_[a-zA-Z]{3}", "", ds["sentence"])

    # Make sure there are no double whitespaces or whitespaces in the beginning or end
    ds["sentence"] = re.sub(" {2,}", " ", ds["sentence"])
    ds["sentence"] = re.sub("^ | $", "", ds["sentence"])
    return ds


def preprocess_es_data(ds):
    ds_without = ds
    # Remove emojis
    ds_without = ds_without.map(remove_emojis)
    # Remove placeholders
    ds = ds.map(remove_tags)
    ds_without = ds_without.map(remove_tags)

    print(f"with: {ds['train'][27283]}")
    print(f"Without: {ds_without['train'][27283]}")
    return ds_without, ds


if __name__ == '__main__':
    # load data
    en_data_path = "Karim-Gamal/SemEval-2018-Task-2-english-emojis"
    es_data_path = "guillermoruiz/MexEmojis"
    dataset_en = load_data(en_data_path)
    dataset_es = load_data(es_data_path)
    dataset_es = dataset_es.rename_column("text", "sentence")

    dataset_es_without_emojis, dataset_es_with_emojis = preprocess_es_data(dataset_es)
    # copy until here

    exploratory_analysis(change_to_pandas(dataset_en), change_to_pandas(dataset_es), change_to_pandas(dataset_es_without_emojis), change_to_pandas(dataset_es_with_emojis))
