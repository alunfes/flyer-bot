import catboost as cb
from catboost import Pool
from numba import jit
import pickle
import numpy as np
import pandas as pd


class CatModel:
    @jit
    def generate_data(self, df: pd.DataFrame, test_size=0.2):
        dff = df
        dff['future_side'] = dff['future_side'].map({'no': 0, 'buy': 1, 'sell': 2, 'both': 3}).astype(int)
        dff = dff.drop(['dt', 'open', 'high', 'low', 'close', 'size'], axis=1)
        size = int(round(df['future_side'].count() * (1 - test_size)))
        train_x = dff.drop('future_side', axis=1).iloc[:size]
        train_y = dff['future_side'].iloc[:size]
        test_x = dff.drop('future_side', axis=1).iloc[size:]
        test_y = dff['future_side'].iloc[size:]
        print('train length='+str(len(train_y)))
        print('test length='+str(len(test_y)))
        return train_x, test_x, train_y, test_y

    @jit
    def generate_bot_pred_data(self, df:pd.DataFrame):
        if 'future_side' in df.columns:
            return df.drop(['dt', 'open', 'high', 'low', 'close', 'size', 'future_side'], axis=1)
        else:
            return df.drop(['dt', 'open', 'high', 'low', 'close', 'size'], axis=1)


    def read_dump_model(self, path):
        with open(path, mode='rb') as f:
            return pickle.load(f)


    @jit
    def calc_accuracy(self, predict, test_y):
        num = len(predict)
        matched = 0
        y = np.array(test_y)
        for i in range(len(predict)):
            if predict[i] == y[i]:
                matched += 1
        return float(matched) / float(num)
