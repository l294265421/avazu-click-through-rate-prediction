from algorithm.wide_deep.wide_deep_wrapper import WideDeepClassifier

_CSV_COLUMNS = ['id', 'click', 'hour', 'C1', 'banner_pos', 'device_id', 'device_ip',
       'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16',
       'C17', 'C18', 'C19', 'C20', 'C21', 'pub_id', 'pub_domain',
       'pub_category', 'device_id_count', 'device_ip_count', 'user_count',
       'smooth_user_hour_count', 'user_click_histroy']

_CSV_COLUMN_DEFAULTS = [[''], [0], ['00000000'], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''], [''],
                        [''], [0.0], [0.0], [0.0],
                        [0.0], ['']]

fields = ['C1', 'banner_pos', 'device_id', 'device_ip',
       'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16',
       'C17', 'C18', 'C19', 'C20', 'C21', 'pub_id', 'pub_domain',
       'pub_category', 'device_id_count', 'device_ip_count', 'user_count',
       'smooth_user_hour_count', 'user_click_histroy']

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

CROSS = []

category = 'site'
train_filename = 'train_{category}.csv'.format(category=category)
test_filename = 'test_{category}.csv'.format(category=category)
classifier = WideDeepClassifier(category, _CSV_COLUMNS, _CSV_COLUMN_DEFAULTS, NUMEIRCAL, CATEGORICAL, train_filename, test_filename, cross=CROSS, train_epochs=1, batch_size=128, mode_type='deep')
classifier.fit()
classifier.predict()
