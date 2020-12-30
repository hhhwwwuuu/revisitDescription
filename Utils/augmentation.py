'''
Author: Zhiqiang
Date: 2020-12-28
This library is used for data augmentation.
    - Backtranslation
    - Auto-generate the possitive sentences based on verb and noun
'''

from BackTranslation import BackTranslation


class DataAugment():
    def __init__(self, url = 'https://translate.google.com/'):
        self.url == url
        self.trans = BackTranslation()


    def translate(self, data, src='en', tmp='zh-cn'):
        '''
            only back-translate the positive smaples
            :param data: DataFrame. --> labelled dataset
            :return: back-translated data
            '''
        # TODO: translate the sentence to generate more positive samples if the sample is labeled on any permission.








