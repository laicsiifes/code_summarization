import os
import pandas as pd


def read_corpus_csv(train_file_path=None, valid_file_path=None,
                    test_file_path=None, sample_size=-1):
    train_data = read_csv_file(train_file_path, sample_size)
    valid_data = read_csv_file(valid_file_path, sample_size)
    test_data = read_csv_file(test_file_path, sample_size)
    return train_data, valid_data, test_data


def read_csv_file(file_path, sample_size=-1):
    if file_path is not None and os.path.exists(file_path):
        if sample_size > 0:
            df = pd.read_csv(file_path, sep='\t', na_filter=False, nrows=sample_size)
        else:
            df = pd.read_csv(file_path, sep='\t', na_filter=False)
        tokens = df['code'].tolist()
        descriptions = df['desc'].tolist()
        return tokens, descriptions
    return None
