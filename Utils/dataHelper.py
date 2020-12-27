'''
Author: Zhiqiang
Date: 2020-12-24
'''

import pandas as pd
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool


def readRawData(curdir, filePath):
    def parallelProcess():
        pass

    def formatRawData(data):
        pbar = tqdm(len(data))
        ids = list(set(data['id'].values.tolist()))
        results = {'id':[], 'text': [], 'count':[]}
        for i in range(len(set(data['label'].values.tolist()))):
            results[str(i+1)]=[]
        for id in ids:
            results['id'].append(id)
            results['text'].append(data.loc[data['id']==id]['text'].values[0])
            tmp = np.zeros(shape=(12,)).astype(int)
            users = []
            for index, row in data.groupby('id').get_group(id).iterrows():
                #results['user'+str(row['user'])] = row['label']
                tmp[row['label']-1]+=1
                if row['user'] not in users:
                    users.append(row['user'])
            for i in range(len(tmp)):
                results[str(i+1)].append(tmp[i])
            results['count'].append(len(users))
            pbar.update(1)
        pbar.close()

        return pd.DataFrame(data= results)



    data = pd.read_csv(filePath)
    formatted = formatRawData(data)
    savePath = curdir +'/data/'+ 'formatted_dataset.csv'
    formatted.to_csv(savePath, encoding='utf-8', index=False)