"""
Author: Zhiqiang
Date: 2020-12-24
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed



def readRawData(curdir, filePath):
    def parallelProcess(id, data, results):

        results['id'].append(id)
        results['text'].append(data.loc[data['id'] == id]['text'].values[0])
        tmp = np.zeros(shape=(12,)).astype(int)
        users = []
        for _, row in data.iterrows():
            tmp[row['label'] - 1] += 1
            if row['user'] not in users:
                users.append(row['user'])
        for i in range(len(tmp)):
            results[str(i + 1)].append(tmp[i])
        results['count'].append(len(users))

        return results

    def formatRawData(data):

        results = {'id': [], 'text': [], 'count': []}
        for i in range(len(set(data['label'].values.tolist()))):
            results[str(i + 1)] = []

        tmp = Parallel(n_jobs=10)(
            delayed(parallelProcess)(id, data.groupby('id').get_group(id), results) for id in
            list(set(data['id'].values.tolist())))

        for sentence in tmp:
            for key in results.keys():
                results[key].extend(sentence[key])

        return pd.DataFrame(data=results)

    data = pd.read_csv(filePath)

    formatted = formatRawData(data)
    savePath = curdir + '/data/' + 'formatted_dataset.csv'
    formatted.to_csv(savePath, encoding='utf-8', index=False)


def groundTruth(data, filePath):
    ground = pd.read_csv(data, encoding='utf-8')
    pbar = tqdm(len(ground))
    header = ground.columns.tolist()[3:]
    for index, row in ground.iterrows():
        if row['count'] > 2:
            threshold = 0.6
            if (row['1'] + row['2'])/row['count'] > threshold:
                # if row['1'] > row['2']:
                #     ground.loc[index, '1'] = 1
                #     ground.loc[index, '2'] = 0
                # else:
                #     ground.loc[index, '1'] = 0
                #     ground.loc[index, '2'] = 1
                ground.loc[index, '1'] = 1 if row['1'] > row['2'] else 0
                ground.loc[index, '2'] = 0 if row['1'] > row['2'] else 1
                for item in header[2:]:
                    ground.loc[index, item] = 0
            else:
                ground.loc[index, '1'] = 0
                ground.loc[index, '2'] = 0
                for item in header[2:]:
                    if row[item] != 0:
                        ground.loc[index, item] = 1
        else:
            if row['1'] + row['2'] >= 2:
                ground.loc[index, '1'] = 1
                for item in header[1:]:
                    ground.loc[index, item] = 0
            else:
                ground.loc[index, '1'] = 0
                ground.loc[index, '2'] = 0
                for item in header[2:]:
                    if row[item] != 0:
                        ground.loc[index, item] = 1
        pbar.update(1)
    pbar.close()
    savePath = filePath + '/data/ground_truth.csv'
    ground.to_csv(savePath, index=False, encoding='utf-8')


def dataClean(data):
    # Discard --> remove all        V
    # None --> remove partial of records V

    # def rmvPunct(sentence):
    #     rule = re.compile(r"\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]")
    #     return rule.sub('', sentence)
    #
    # def rmBlank(sentence):
    #     return [word for word in sentence if word != ' ']

    def countWords(sentence):
        words = sentence.split(' ')
        return [word for word in words if word.strip() != '']

    # def drouptout(x, level, seed=None):
    #
    #     if level < 0. or level >= 1:
    #         raise ValueError('Dropout level must be in interval [0, 1.].')
    #     if seed is None:
    #         seed = np.random.randint(1, 10e6)
    #
    #     x = np.array(x)
    #
    #     retain_prob = 1. - level
    #     random_tensor = np.random.binomial(n=1, p=retain_prob, size=x.shape)
    #     x *= random_tensor
    #     #print(np.count_nonzero(x))
    #     return [i for i in x if i != 0]

    # delete 'count'
    data.drop(['count'], axis=1, inplace=True)
    total = len(data)
    # delete all discard data and the column
    data.drop(data[data['1'] == 1].index, inplace=True)
    data.drop(['1'], axis=1, inplace=True)
    print('Discard {} records, and retain {} samples'.format(total-len(data), len(data)))

    # delete some records with NONE
    # 删除的标准：
    # 1. 少于10个字
    # 2. 随机删除
    rmIndex = []
    pbar = tqdm(len(data))
    for index, row in data.iterrows():
        #words = rmBlank(rmvPunct(row['text']))
        if row['2'] != 1:
            continue
        words = countWords(row['text'])
        if len(words) < 10:
            rmIndex.append(index)
        pbar.update(1)
    pbar.close()
    data.drop(rmIndex, inplace=True)
    print("Deleting {} records with NONE label that consists of less than 10 letters, remain {} records".format(len(rmIndex), len(data)))


    # rename columns
    columns = {'2': 'None', '3':'Calendar', '4': 'Camera', '5': 'Contacts', '6': 'Location', '7': 'Mircophone',
               '8': 'Phone', '9': 'SMS', '10': 'Call_Log', '11': 'Storage', '12': 'Sensors'}
    data.rename(columns = columns, inplace=True)
    # save data to a new csv
    data.to_csv('data/clean_data.csv', index=False)
    return data

def validateData(data):
    print("Validating the data......")
    for index, row in data.iterrows():
        if row['None'] == 1:
            if 1 in row[3:].tolist():
                print(row['id'])
        else:
            if 1 not in row[3:].tolist():
                print(row['id'])

def resetID(data):
    num = 0
    print('Resetting index of dataset.....')
    for index, row in data.iterrows():
        data.loc[index, 'id'] = num
        num += 1
    return data

