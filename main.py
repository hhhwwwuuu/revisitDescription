
from Utils import dataHelper as dh
import os
import pandas as pd
import numpy as np




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #dh.readRawData(os.getcwd(), 'data/final_annotation.csv')
    dh.groundTruth('data/formatted_dataset.csv', os.getcwd())
    #print(dh.head(5))
    data = pd.read_csv('data/ground_truth.csv')
    data = dh.dataClean(data)
    data = dh.validateData(data)

