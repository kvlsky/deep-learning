import pandas as pd

def get_data(path: str):
    files = []
    labels = []
    f = pd.read_csv(path, header=None, delim_whitespace=True)
    for idx, cl in zip(f[0], f[1]):
        files.append(idx)
        labels.append(cl)
    return files, labels

def get_test_data(path: str):
    files = []
    f = pd.read_csv(path, header=None, delim_whitespace=True)
    for idx in f[0]:
        files.append(idx)
    return files


