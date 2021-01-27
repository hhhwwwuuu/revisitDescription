
from Utils import dataHelper as dh
import os
import pandas as pd
import numpy as np
import stanza
import random
from nltk.corpus import wordnet as wn
import tensorflow as tf
from Utils import Augmenter


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


    """
    数据增强
    1. 合并其他数据集
    2. back-translation
    3. thesaurus
    """

    #da.thesaurus("You can use it as a note or memo recorder, voice recorder, voice recorder, or even record the entire conversation")

    # data = pd.read_csv('data/clean_data.csv')
    # data = da.merge(data)
    #
    # print('Starting Back-Translation.....')
    # result = result = da.translate(data)
    # result.to_csv('data/backtranslated_dataset.csv', index=False)

    data = pd.read_csv('data/clean_data.csv')


    da = Augmenter()
    print('Starting Replace synonyms.....')
    thesaurus = da.thesaurus_verb(data)

    print('Starting Back-Translation.....')
    result = da.translate(thesaurus)
    #print('Starting Replace synonyms.....')
    #thesaurus = da.thesaurus_verb(result)
    result.to_csv('data/thesaurus_back_0112.csv', index=False)

    #da = Augmenter()
    # doc = da.nlp("Read calendar events - access is required for the calendar event counter.")
    # ids = [1, 7]
    # word_list = [word for sent in doc.sentences for word in sent.words]
    # noun_map = {}
    # for verb in ids:
    #     noun_map[word_list[verb - 1].lemma] = list(set(da.search_noun([verb], word_list)))
    #
    # print(noun_map)




