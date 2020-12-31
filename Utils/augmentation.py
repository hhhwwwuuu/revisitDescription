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
import stanza
from nltk.corpus import wordnet as wn
from nltk.parse.stanford import StanfordParser
import nltk
import pandas as pd

class Augmenter():
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


    def _POStagging(self, sentence):
        #TODO: analyze the POS of sentence, pick the verb and noun
        # For Thesaurus
        pass


    def thesaurus(self, data):
        # nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
        # doc = nlp(data)
        # print(
        #     *[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for
        #       sent in doc.sentences for word in sent.words], sep='\n')
        # print(*[
        #     f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head - 1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}\txpos:{word.xpos}'
        #     for sent in doc.sentences for word in sent.words], sep='\n')
        print(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(data))))
        #print(nltk.parse(data))






    def merge(self, data):
        #TODO:merge our dataset with more positive samples from other dataset
        # load other dataset
        def handleACNet(id, result, data):
            # remove irrelevant columns
            columns = ['SETTOMGS', 'TAKS', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15',
                       'Unnamed: 16', 'Unnamed: 17']
            data.drop(columns=columns, inplace=True)
            id = result['id'][-1] if len(result) != 0 else id
            header = data.columns.tolist()[2:]
            for index, row in data.iterrows():
                if 1 in row[2:]:
                    id += 1
                    result['id'].append(id)
                    result['text'].append(row['sentence'])
                    # TODO: 需要根据我们自己数据集的header进行重新赋值！！！！
                    for permission in header:
                        if row[permission] == 1:
                            result[permission].append(1)
                        else:
                            result[permission].append(0)



        def handleWHYPER():
            pass


        whyper_Calendar = pd.read_excel('data/dataset/Read_Calendar.xls')
        whyper_Contacts = pd.read_excel('data/dataset/Read_Contacts.xls')
        whyper_Audio = pd.read_excel('data/dataset/Record_Audio.xls')
        acnet = pd.read_excel('data/dataset/ACNet.xlsx')


        # create the dict for saving formatted data
        id = np.max(data['id'].values)
        results = {key: [] for key in data.columns.values}
        results = handleACNet(id, results, acnet)










