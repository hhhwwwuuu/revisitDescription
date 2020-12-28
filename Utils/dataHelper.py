'''
Author: Zhiqiang
Date: 2020-12-24
'''

import pandas as pd
from tqdm import tqdm
import numpy as np
#from multiprocessing import Pool
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
            results[str(i+1)].append(tmp[i])
        results['count'].append(len(users))

        return results


    def formatRawData(data):

        results = {'id': [], 'text': [], 'count': []}
        for i in range(len(set(data['label'].values.tolist()))):
            results[str(i + 1)] = []


        tmp = Parallel(n_jobs=10)(
            delayed(parallelProcess)(id, data.groupby('id').get_group(id), results) for id in list(set(data['id'].values.tolist())))



        for sentence in tmp:
            for key in results.keys():
                results[key].extend(sentence[key])

        return pd.DataFrame(data= results)


    data = pd.read_csv(filePath)

    formatted = formatRawData(data)
    savePath = curdir +'/data/'+ 'formatted_dataset.csv'
    formatted.to_csv(savePath, encoding='utf-8', index=False)

def groundTruth(data, filePath):
    ground = pd.read_csv(data, encoding='utf-8')
    pbar = tqdm(len(ground))
    header = ground.columns.tolist()[3:]
    for index, row in ground.iterrows():
        if row['count'] > 2:
            threshold = 0.6
            for item in header:
                if (row[item]/row['count']) < threshold:
                    ground.loc[index, item] = 0
                else:
                    ground.loc[index, item] = 1
        else:
            if row['1'] + row['2'] != 0:
                ground.loc[index, '1'] = 1
                for item in header:
                    ground.loc[index, item] = 0
            else:
                ground.loc[index, '1'] = 0
                ground.loc[index, '2'] = 0
                for item in header[2:]:
                    if row[item] == 0:
                        continue
                    else:
                        ground.loc[index, item] = 1
        pbar.update(1)
    pbar.close()
    savePath = filePath + '/data/ground_truth.csv'
    ground.to_csv(savePath, index= False, encoding='utf-8')

    # ground = pd.read_csv(data, encoding='utf-8')
    #
    # # 三个人标注的情况下，少数服从多数
    # print(ground[ground['count'] >= 3].index)

def dataClean(data):
    # TODO: remove negative sentences from dataset
    # Discard --> remove all
    # None --> remove partial of records
    pass
