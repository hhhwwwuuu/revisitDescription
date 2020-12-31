'''
Author: Zhiqiang
Date: 2020-12-28
This library is used for data augmentation.
    - Backtranslation
    - Auto-generate the possitive sentences based on verb and noun
'''

from BackTranslation import BackTranslation
import numpy as np
import pandas as pd
from tqdm import tqdm

class DataAugment():
    def __init__(self, url = 'https://translate.google.com/'):
        self.url = url
        self.trans = BackTranslation()


    def translate(self, data, src='en', tmp='zh-cn'):
        '''
            only back-translate the positive smaples
            :param data: DataFrame. --> labelled dataset
            :return: back-translated data
            '''

        # init a dict
        translated = {key: [] for key in data.columns.tolist()}

        id = np.max(data['id'].values)
        # translated the positive smaples
        pbar = tqdm(len(data))
        for index, row in data.iterrows():
            pbar.update(1)
            if row['None'] == 1:
                continue
            else:
                id += 1
                for key, value in translated.items():
                    # update id and text
                    if key == 'id':
                        translated[key].append(id)
                    elif key == 'text':
                        try:
                            translated[key].append(self.trans.translate(row['text'], src, tmp).result_text)
                        except:
                            translated[key].append(row[key])
                    else:
                        # copy the labels
                        translated[key].append(row[key])
        pbar.close()

        # concate data frame
        tmp_frame = pd.DataFrame(translated)
        frames = [data, tmp_frame]
        result = pd.concat(frames)
        return result


    def POStagging(self, data):
        pass








