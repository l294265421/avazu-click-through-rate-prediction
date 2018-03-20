import pandas as pd
def wc(file_path):
    count = -1
    with open(file_path, 'rU') as f:
        for count, line in enumerate(open(file_path, 'rU')):
            pass
    count += 1
    return count

def head(file_path, n):
    return pd.read_csv(file_path, iterator=True).get_chunk(n)

def sample_submission_id(file_path):
    return [i for i in pd.read_csv(file_path, skip_blank_lines=True, header=0)['id']]

def unique_value(file_path, line_index):
    s = set()
    with open(file_path) as f:
        f.readline()
        for line in f:
            if line.strip():
                elements = line.split(',')
                s.add(elements[line_index])
    return list(s)

def split_file(file_path, n):
    for i in range(n):
        dot_index = file_path.rfind('.')
        target_file_path = file_path[:dot_index]
        target_file_path += '_' + str(i)
        target_file_path += file_path[dot_index:]
        with open(target_file_path, 'w', encoding='utf-8') as target_file:
            with open(file_path, 'r', encoding='utf-8') as src_file:
                header = src_file.readline()
                target_file.write(header)
                count = 0
                for line in src_file:
                    if count % n == i:
                        target_file.write(line)
                    count += 1

if __name__ == '__main__':
    base_dir = r'D:\document\program\ml\machine-learning-databases\kaggle\Click-Through Rate Prediction\\'
    # train_file = base_dir + 'train.csv'
    # test_file = base_dir + 'test.csv'
    # sampleSubmission = base_dir + 'sampleSubmission.csv'
    # print(wc(base_dir + 'train.csv'))
    # print(wc(base_dir + 'test.csv'))
    # print(wc(r'D:\Users\liyuncong\PycharmProjects\avazu-click-through-rate-prediction\util\common_util.py'))
    # train_head_n = head(train_file, 50000)
    # train_head_n.to_csv(base_dir + 'train_head.csv', index=False)
    # test_head_n = head(test_file, 5000)
    # test_head_n.to_csv(base_dir + 'test_head.csv', index=False)
    # sampleSubmission_head = head(sampleSubmission, 5000)
    # sampleSubmission_head.to_csv(base_dir + 'sampleSubmission_head.csv', index=False)
    # wide_result_head = head(base_dir + 'va.r0.site.sp', 5)
    # print(wide_result_head)
    # wide_result_head.to_csv(base_dir + 'tr.r0.app.new.csv_head.csv', index=False)
    # count = 0
    # with open(train_file) as t:
    #     for line in t:
    #         parts = line.split(',')
    #         if '1000009418151094273' == parts[0]:
    #             count += 1

    # print(count)
    # unique_C1 = unique_value(base_dir + 'train_head.csv', 3)
    # for element in unique_C1:
    #     print(element)
    # split_file(base_dir + 'tr.r0.site.sp', 2)