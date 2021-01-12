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
# from stanza.server import CoreNLPClient
from nltk.corpus import stopwords
from itertools import product
from nltk.stem import WordNetLemmatizer
from collections import Counter
import json


import os

os.environ['STANFORD_PARSER'] = "H:\Google Drive\Implementation\revisitDescription\data\parser\stanford-parser-full-2020-11-17"

class Augmenter():
    def __init__(self, url = 'https://translate.google.com/'):
        self.url = url
        self.trans = BackTranslation()
        self.permissions = ['None', 'Calendar', 'Camera', 'Contacts', 'Location', 'Mircophone', 'Phone',
                            'SMS', 'Call_Log', 'Storage', 'Sensors']
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse',
                                   tokenize_no_ssplit = True)

        #self.client = CoreNLPClient(timeout=30000, memory='8G')
        #stanza.install_corenlp()
        #stanza.download_corenlp_models(model='english', version='4.2.0')
        self.stop = stopwords.words('english')
        self.wnl = WordNetLemmatizer()

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

    def thesaurus(self, data):
        """
        replace the verb of positive sentences to generate more equivalent sematic sentence
        1. separate the positive samples
        2. extract verb from labelled sentence
        3. replace verb to generate new sentence
        4. combine the new sentence with original labels
        :param data: pd.DataFrame type.
        :return: pd.DataFrame with new sentences
        """
        # extact all positve samples
        #posSamples = data.loc[data['None']==0]

        # create a dict to store the result of thesaurus
        id = np.max(data['id'].values)
        results = {key:[] for key in data.columns.tolist()}
        pbar = tqdm(len(data))
        for index, row in data.iterrows():
            pbar.update(1)
            # get the new sentences
            if row['None'] == 1:
                continue
            sentences = self.sentenceRefactor(row['text'])
            #print(row['id'])
            # merge all new sentences into result
            for sentence in sentences:
                id += 1
                results['id'].append(id)
                results['text'].append(sentence)
                # append labels
                for key in data.columns.tolist()[2:]:
                    results[key].append(row[key])
            #print("....{}".format(row['id']))

        pbar.close()
        newThesaurus = pd.DataFrame(results)
        frames = [data, newThesaurus]
        return pd.concat(frames)

    def sentenceRefactor(self, text):
        """

        :param text:
        :return:
        """
        # generate the syntax tree
        doc = self.nlp(text)


        # print(*[
        #     f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head - 1].text if word.head > 0 else "root"}\tlemma: {word.lemma}\txpos:{word.xpos}'
        #     for sent in doc.sentences for word in sent.words], sep='\n')

        '''
        pick out all replacable verbs
        '''
        syn = {}
        for sent in doc.sentences:
            #self.noun_phrases(self.client, sent.text, _annotators="tokenize,ssplit,pos,lemma,parse")
            for word in sent.words:
                if word.xpos.startswith('VB') and not self.isStopwords(word.text):
                    # remove duplicate
                    # skip only one synonyms
                    voca = self.synonyms(word.lemma)
                    if len(voca) > 1 and not self.duplicateSynonyms(syn, word.lemma):
                        syn[word] = voca
        #print(*[f'{word.text}: {result[word]}' for word in result.keys()])

        '''
        re-construct the new sentence with synonyms
        1. 如果verb多于1个，进行排列组合
        2. 逐个插入 造句
        '''
        arg = tuple(item for item in syn.values())
        res = [list(x) for x in list(product(*arg))]
        # print(res)
        # print(len(res))

        #augSize = len(res)
        """
        init a empty list to save the thesaurus result
        len(res) is the number of new sentences
        """

        sentences = [''] * len(res)
        # print(sentences)
        for sent in doc.sentences:
            for word in sent.words:
                #print(word.text)
                if word in syn:
                    sentences = [sent+' '+w.pop(0) if len(sent)!=0 else sent+w.pop(0) for w, sent in zip(res, sentences)]
                else:
                    sentences = [sent+' '+ word.text if word.deprel!='punct' else sent+word.text for sent in sentences]
        sentences = [sent.lstrip() for sent in sentences]

        #print(sentences)

        return sentences

    def duplicateSynonyms(self, syn, word):
        #print(syn)
        for l in syn.keys():
            if l.lemma == word:
                return True
        return False

    def synonyms(self, word):
        """
        collect synonyms, but only pick single word rather than phrase
        :param word:
        :param tag:
        :return:
        """
        syn = []
        tmp = wn.synsets(word, pos=wn.VERB)
        # print(word)
        if len(tmp) == 0:
            return []
        for l in tmp[0].lemmas():
            if '_' not in l.name():
                syn.append(l.name())
        # for synset in wn.synsets(word, pos=tag):
        #     #print(synset)
        #     for l in synset.lemmas():
        #         if '_' not in l.name():
        #             syn.append(l.name())
        syn = list(set(syn))
        if word in syn:
            syn.remove(word)
        return syn

    def isStopwords(self, word):
        return True if word in self.stop else False

    def merge(self, data):
        """

        :param data: required. Our labeled data.
        :return:
        """
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

    def extract_common_verbs(self, permission, data):
        """
        extract verb for each labels
        :param data:
        :return:
        """
        print("Extracting common verbs from {} label......".format(permission))
        tmp = []
        for index, row in data.iterrows():
            doc = self.nlp(row['text'])
            tmp.extend([self.wnl.lemmatize(word.text, pos='v').lower() for sent in doc.sentences for word in sent.words
                   if word.xpos.startswith('VB') and not self.isStopwords(word.text)])
        result = self._counter(tmp)
        print('{}: {}'.format(permission, result))
        return result



    def _counter(self, nums):
        return [word[0] for word in Counter(nums).most_common(10)]

    def thesaurus_verb(self, data):
        """
        keyword-based replace
        1. extracting the most common verbs from sentences and pick their noun
        2. if the verb exists in the sentence, replace it with other synonyms when following noun also appears in the sentence
        :param data:
        :return:
        """
        #TODO: 提取频率最大的动词 及其后面所附带的名词或名词词组
        # 当替换verb时候， 须确认同义词中是否含有相同的noun
        # 因为storage和carmera 被lable的句子可能大量重合！导致替换词的时候 替换了不匹配的名词
        commons = {} # save the common verbs and their nouns for result analysis
        thesaurus_result = [data]
        pbar = tqdm(len(self.permissions[1:]))
        id = np.max(data['id'])
        for permission in self.permissions[1:]:
            pbar.update(1)
            # common verbs
            common_verbs = self.extract_common_verbs(permission, data.groupby(permission).get_group(1))
            noun_map = {}
            # extracting noun based common verbs for one permission
            for index, row in data.groupby(permission).get_group(1).iterrows():
                doc = self.nlp(row['text'])

                tmp = self.extract_noun(common_verbs, doc)
                # merge tmp into noun_map
                for key, value in tmp.items():
                    if key in noun_map:
                        noun_map[key].extend([v for v in value if v not in noun_map[key] and v.isalpha()])
                    else:
                        noun_map[key] = value

            # store the common verbs and noun
            commons[permission] = noun_map
            ''' Replace!!!
            if the sentence has any common verb, 
            replace the verb if the following nouns also appear in that list of nouns.
            Otherwise, skip 
            '''
            #TODO:
            # 1. 确定可替换的动词
            # 2. 替换并插入到原始数据
            id, tmp_frame = self.replace_verb(id, noun_map, data.groupby(permission).get_group(1))
            thesaurus_result.append(tmp_frame)
        pbar.close()
        common_verb = json.dumps(commons)
        with open('data/common_verbs.json', 'w') as f:
            f.write(common_verb)
            f.close()
        return pd.concat(thesaurus_result)




    def extract_noun(self, common_verbs, doc):
        """
        search relevant noun based on verbs
        1. obtain the id of verbs if there are common verbs in this sentence
        2. extract noun based on verbs
        :param common_verbs:
        :param doc:
        :return:
        """
        noun_map = {}
        # extracting common verbs id in sentence
        verbs = [word.id for sent in doc.sentences for word in sent.words
                 if self.wnl.lemmatize(word.lemma, 'v').lower() in common_verbs]
        # word list of sentence
        word_list = [word for sent in doc.sentences for word in sent.words]

        if len(verbs) == 0:
            return {}
        # search noun for verbs
        for verb in verbs:
            noun_map[self.wnl.lemmatize(word_list[verb-1].lemma, 'v').lower()] = list(set(self.search_noun([verb],
                                                                                                           word_list)))
        return noun_map

    def search_noun(self, ids, words):
        """
        using the id of verbs to search relevant noun

        :param ids: list. the ids of verbs
        :param words: list of word obejct of sentences.
        :return: a list of relevant words
        """
        result = []
        noun_ids = []
        for id in ids:
            for word in words:
                if word.head == id and word.xpos.startswith('NN'):
                    result.append(self.wnl.lemmatize(word.lemma, 'n').lower())
                    noun_ids.append(word.id)
        if len(noun_ids) == 0:
            return result
        else:
            result.extend(self.search_noun(noun_ids, words))
            return result

    def replace_verb(self, num, noun_map, data):
        """

        :param num: the max index of DataFrame
        :param noun_map: the mapping among nouns and verbs for current permission
        :param data: a set of data for current label
        :return: id and
        """

        def _generate_sentences(word_id, verbs, doc):
            '''
            replace verb whose id is word_id
            :param word_id:
            :param verbs:
            :param doc:
            :return:
            '''
            results = ['']*len(verbs)
            for sent in doc.sentences:
                for word in sent.words:
                    if word.id == word_id:
                        results = [text + ' ' + verb if len(text)!=0 else verb for text, verb in zip(results, verbs)]
                    else:
                        results = [text + ' ' + word.text if word.deprel!='punct' else text+word.text for text in results]
            results = [sent.lstrip() for sent in results]

            return results


        results = {key:[] for key in data.columns.tolist()}

        for index, row in data.iterrows():
            doc = self.nlp(row['text'])

            verbs = {self.wnl.lemmatize(word.lemma, 'v').lower(): word.id
                     for sent in doc.sentences for word in sent.words if word.xpos.startswith('VB')}
            word_list = [word for sent in doc.sentences for word in sent.words]

            if len(set(list(verbs.keys())) & set(list(noun_map.keys()))) > 0:
                #sentence has at least one common verb
                for verb, id in verbs.items():
                    if verb in noun_map:
                        # determine which verbs can be replaced
                        nouns = self.search_noun([id], word_list)
                        if len(nouns) != 0:
                            replaced_verbs = []
                            for key in noun_map.keys():
                                if len(set(nouns) & set(noun_map[key])) > 0 and verb != key:
                                    replaced_verbs.append(key)
                            # generate new sentences with other verbs
                            sentences = _generate_sentences(id, replaced_verbs, doc)

                            #insert sentences into dict
                            for sent in sentences:
                                num += 1
                                results['id'].append(num)
                                results['text'].append(sent)
                                for per in data.columns.tolist()[2:]:
                                    results[per].append(row[per])
        results = pd.DataFrame(results)
        return num, results



















