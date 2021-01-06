'''
Author: Zhiqiang
Date: 2020-12-28
This library is used for data augmentation.
    - Backtranslation
    - Auto-generate the possitive sentences based on verb and noun
'''

from BackTranslation import BackTranslation
import numpy as np
# import pandas as pd
from tqdm import tqdm
import stanza
from nltk.corpus import wordnet as wn
# import nltk
import pandas as pd
from stanza.server import CoreNLPClient


import os

os.environ['STANFORD_PARSER'] = "H:\Google Drive\Implementation\revisitDescription\data\parser\stanford-parser-full-2020-11-17"

class Augmenter():
    def __init__(self, url = 'https://translate.google.com/'):
        self.url = url
        self.trans = BackTranslation()
        self.permissions = ['None', 'Calendar', 'Camera', 'Contacts', 'Location', 'Mircophone', 'Phone',
                            'SMS', 'Call_Log', 'Storage', 'Sensors']
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse',
                                   tokenize_no_ssplit = True)

        self.client = CoreNLPClient(timeout=30000, memory='8G')
        #stanza.install_corenlp()
        #stanza.download_corenlp_models(model='english', version='4.2.0')



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


    def _tag(self, tag):
        if tag.startswith('NN'):
            return wn.NOUN
        if tag.startswith('VB'):
            return wn.VERB


    def noun_phrases(self,_client, _text, _annotators=None):
        pattern = 'NP'
        matches = _client.tregex(_text,pattern,annotators=_annotators)
        print("\n".join(
            ["\t" + sentence[match_id]['spanString'] for sentence in matches['sentences'] for match_id in sentence]))
        for sentence in matches['sentences']:
            print(sentence)




    def thesaurus(self, text):
        """

        :param data:
        :return:
        """
        # generate the syntax tree
        doc = self.nlp(text)

        print(*[
            f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head - 1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}\txpos:{word.xpos}'
            for sent in doc.sentences for word in sent.words], sep='\n')

        # change the verb and noun with synonyms
        for sent in doc.sentences:
            self.noun_phrases(self.client, sent.text, _annotators="tokenize,ssplit,pos,lemma,parse")
            for word in sent.words:
                if word.xpos.startswith('VB') or word.xpos.startswith('NN'):
                    print(word.text, self.synonyms(word.text, self._tag(word.xpos)))


        # match the noun phrases
        # with CoreNLPClient(timeout=30000, memory='16G') as client:
        #     englishText = data
        #     print('---')
        #     print(englishText)
        #     self.noun_phrases(client, englishText, _annotators="tokenize,ssplit,pos,lemma,parse")


    def synonyms(self, word, tag):
        syn = []
        for synset in wn.synsets(word, pos=tag):
            for l in synset.lemmas():
                syn.append(l.name())
        syn = list(set(syn))
        return syn







    def merge(self, data):
        """

        :param data: required. Our labeled data.
        :return:
        """
        #TODO:merge our dataset with more positive samples from other dataset
        # load other dataset
        def handleACNet(id, result, data):
            """

            :param id:
            :param result:
            :param data:
            :return:
            """
            # remove irrelevant columns
            columns = ['SETTINGS', 'TASKS', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15',
                       'Unnamed: 16', 'Unnamed: 17']
            data.drop(columns=columns, inplace=True)
            data.dropna(how='any', inplace=True)

            header = data.columns.tolist()[2:]
            pbar = tqdm(len(data))
            for index, row in data.iterrows():
                if 1 in row[2:].tolist():
                    id += 1
                    result['id'].append(id)
                    result['text'].append(row['sentence'])

                    for permission in self.permissions:
                        if permission.upper() in header:
                            if row[permission.upper()]==1:
                                result[permission].append(1)
                            else:
                                result[permission].append(0)
                        else:
                            result[permission].append(0)
                pbar.update(1)
            pbar.close()
            return id, result



        def handleWHYPER(id, result, permission, data):
            """

            :param id:
            :param result:
            :param data:
            :return:
            """
            # deleting the useless colomns
            rmColomns = [col for col in data.columns.tolist() if col.startswith('Unnamed')]
            data.drop(columns=rmColomns, inplace=True)
            data.dropna(how='any', inplace=True)

            header = data.columns.tolist()
            pbar = tqdm(len(data))
            for index, row in data.iterrows():
                pbar.update(1)
                if row[header[1]] in [1,2,3]: # 1,2,3 indicate positive samples in WHYPER dataset
                    id += 1
                    result['id'].append(id)
                    result['text'].append(row[header[0]])

                    for per in self.permissions:
                        if permission.lower() == per.lower() and row[header[1]] == 1:
                            result[per].append(1)
                        else:
                            result[per].append(0)
            pbar.close()

            return id, result


        whyper_Calendar = pd.read_excel('data/dataset/Read_Calendar.xls')
        whyper_Contacts = pd.read_excel('data/dataset/Read_Contacts.xls')
        whyper_Audio = pd.read_excel('data/dataset/Record_Audio.xls')
        acnet = pd.read_excel('data/dataset/ACNet.xlsx')
        whyper_data = {'Calendar': whyper_Calendar, 'Contacts': whyper_Contacts, 'Mircophone': whyper_Audio}


        # create the dict for saving formatted data
        id = np.max(data['id'].values)
        results = {key:[] for key in data.columns.tolist()}

        print("Extracting positive smaples from AC-Net ......")
        id, results = handleACNet(id, results, acnet)


        print("Extracting positive samples from WHYPER ......")
        for permission, whyper in whyper_data.items():
            id, results = handleWHYPER(id, results, permission, whyper)


        # combine the original dataset with new extra data from previous works
        posData = pd.DataFrame(results)

        frames = [data, posData]
        result = pd.concat(frames)

        #result.to_csv('data/combined_clean_data.csv', index=False)
        return result











