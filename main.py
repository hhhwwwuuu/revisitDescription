
from Utils import dataHelper as dh
import os




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dh.readRawData(os.getcwd(), 'data/final_annotation.csv')
    print(dh.head(5))
