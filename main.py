
from Utils import dataHelper as dh
import os
import pandas as pd
import numpy as np
from Utils.augmentation import Augmenter
import stanza



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #dh.readRawData(os.getcwd(), 'data/final_annotation.csv')
    # dh.groundTruth('data/formatted_dataset.csv', os.getcwd())
    #print(dh.head(5))
    # data = pd.read_csv('data/ground_truth.csv')
    # data = dh.dataClean(data)
    # dh.validateData(data)
    # data = dh.resetID(data)
    # data.to_csv('data/clean_data.csv', index=False, encoding='utf-8')
    #
    #
    # data = pd.read_csv('data/clean_data.csv', encoding='utf-8')
    # da = Augmenter()
    # result = da.translate(data)
    # print('Starting Back-Translation.....')
    # result.to_csv('data/backtranslated_dataset.csv', index=False)

    #stanza.download(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

    da = Augmenter()
    da.thesaurus("Record your voice, press play and listen.")
