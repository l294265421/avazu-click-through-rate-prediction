from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
GradientBoostingClassifier)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn import *
from data.raw_data import *
import bisect
from xgboost import XGBClassifier

FIELDS = ['id',
 'click',
 'hour',
 'C1',
 'banner_pos',
 'device_id',
 'device_ip',
 'device_model',
 'device_type',
 'device_conn_type',
 'C14',
 'C15',
 'C16',
 'C17',
 'C18',
 'C19',
 'C20',
 'C21']
NEW_FIELDS = FIELDS+['pub_id','pub_domain','pub_category','device_id_count','device_ip_count','user_count','smooth_user_hour_count','user_click_histroy']

typemap = {
'id':str,
 'click':int,
 'hour':str,
 'C1':str,
 'banner_pos':str,
 'device_id':str,
 'device_ip':str,
 'device_model':str,
 'device_type':str,
 'device_conn_type':str,
 'C14':str,
 'C15':str,
 'C16':str,
 'C17':str,
 'C18':str,
 'C19':str,
 'C20':str,
 'C21':str,
 'pub_id':str,
'pub_domain':str,
'pub_category':str,
'device_id_count':int,
'device_ip_count':int,
'user_count':int,
'smooth_user_hour_count':int,
'user_click_histroy':str
}

CATEGORICAL = [
 'C1',
 'banner_pos',
 'device_id',
 'device_ip',
 'device_model',
 'device_type',
 'device_conn_type',
 'C14',
 'C15',
 'C16',
 'C17',
 'C18',
 'C19',
 'C20',
 'C21',
 'pub_id',
 'pub_domain',
 'pub_category',
 'user_click_histroy']
NUMEIRCAL = ['device_id_count','device_ip_count','user_count','smooth_user_hour_count']

label_column = 'click'
feature_columns = NEW_FIELDS[3:]

for category in [[train_app_path, test_app_path, gbdt_train_app_path, gbdt_test_app_path],
                 [train_site_path, test_site_path, gbdt_train_site_path, gbdt_test_site_path]]:
    train_data_gbdt = pd.read_csv(category[0], dtype=typemap)
    train_data_gbdt_y = train_data_gbdt[label_column]
    train_data_gbdt_x = train_data_gbdt[feature_columns]

    test_data_gbdt = pd.read_csv(category[1], dtype=typemap)
    test_data_gbdt_x = test_data_gbdt[feature_columns]

    for column in CATEGORICAL:
        train_data_gbdt_x[column] = train_data_gbdt_x[column].fillna('')
        test_data_gbdt_x[column] = test_data_gbdt_x[column].fillna('')

        lbl = LabelEncoder()
        lbl.fit(train_data_gbdt_x[column])

        test_data_gbdt_x[column] = test_data_gbdt_x[column].map(lambda s: 'unknown' if s not in lbl.classes_ else s)

        le_classes = lbl.classes_.tolist()
        bisect.insort_left(le_classes, 'unknown')
        lbl.classes_ = le_classes

        train_data_gbdt_x[column] = lbl.transform(train_data_gbdt_x[column])
        test_data_gbdt_x[column] = lbl.transform(test_data_gbdt_x[column])

    # 弱分类器的数目
    n_estimator = 30
    # 调用GBDT分类模型。
    gd = XGBClassifier(max_depth=3, n_estimators=n_estimator, n_jobs=8, nthread=8)
    gbdt_train_sample_num = round(len(train_data_gbdt_x) / 2)
    '''使用X_train训练GBDT模型，后面用此模型构造特征'''
    gd.fit(train_data_gbdt_x[:gbdt_train_sample_num], train_data_gbdt_y[:gbdt_train_sample_num])

    train_features = gd.apply(train_data_gbdt_x[gbdt_train_sample_num:])
    train_features = train_features.astype(int)
    train_df = pd.DataFrame(train_features)
    train_df['click'] = train_data_gbdt_y[gbdt_train_sample_num:].reset_index(drop=True)
    train_df.to_csv(category[2], index=False)

    test_features = gd.apply(test_data_gbdt_x)
    test_features = test_features.astype(int)
    test_df = pd.DataFrame(test_features)
    test_df['click'] = 0
    test_df.to_csv(category[3], index=False)