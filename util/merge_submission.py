import pandas as pd

base_dir = r'D:\document\program\ml\machine-learning-databases\kaggle\Click-Through Rate Prediction\\'

app = pd.read_csv(base_dir + 'app_submit.csv')
site = pd.read_csv(base_dir + 'site_submit.csv')

merge = pd.concat([app, site], axis=0)
merge.to_csv(base_dir + 'merge_submit.csv', index=False)