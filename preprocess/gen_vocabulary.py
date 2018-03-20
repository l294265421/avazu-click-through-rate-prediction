import pandas as pd

from data.raw_data import base_dir

def gen_vocabulary_file_path(base_dir, category, column_name):
    return base_dir + 'vocabulary\\' + category + '_' + column_name

for category in ['site', 'app']:
    train_filename = 'train_{category}.csv'.format(category=category)
    df = pd.read_csv(base_dir + train_filename, nrows=3)
    for column_name in df.columns:
        if 'id' == column_name:
            continue
        vocabulary = pd.read_csv(base_dir + train_filename, usecols=[column_name])[column_name].unique()
        vocabulary_file_path = gen_vocabulary_file_path(base_dir, category, column_name)
        with open(vocabulary_file_path, 'w') as w:
            for word in vocabulary:
                w.write(str(word) + '\n')


