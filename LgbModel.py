import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from numba import jit
import numpy as np
import pandas as pd
import pickle


class LgbModel:
    @jit
    def generate_data(self, df: pd.DataFrame, test_size=0.2):
        dff = df
        dff['future_side'] = dff['future_side'].map({'no': 0, 'buy': 1, 'sell': 2, 'both': 3}).astype(int)
        #dff = dff.drop(['dt', 'open', 'high', 'low', 'close', 'size'], axis=1)
        dff = dff.drop(['dt', 'size'], axis=1)
        size = int(round(dff['future_side'].count() * (1 - test_size)))
        train_x = dff.drop('future_side', axis=1).iloc[0:size]
        train_y = dff['future_side'].iloc[0:size]
        test_x = dff.drop('future_side', axis=1).iloc[size:]
        test_y = dff['future_side'].iloc[size:]
        print('train length=' + str(len(train_y)))
        print('test length=' + str(len(test_y)))
        return train_x, test_x, train_y, test_y

    @jit
    def generate_bot_pred_data(self, df: pd.DataFrame):
        if 'future_side' in df.columns:
            return df.drop(['dt', 'size', 'future_side'], axis=1)
        else:
            return df.drop(['dt', 'size'], axis=1)

    @jit
    def train(self, train_x, train_y):
        print('training data description')
        print('train_x:', train_x.shape)
        print('train_y:', train_y.shape)
        train = lgb.Dataset(train_x, train_y)
        lgbm_params = {
            'objective': 'multiclass',
            'num_class': 4,
        }
        model = lgb.train(lgbm_params, train)
        return model

    def load_model(self, path):
        with open('./Model/lgb_model.dat', 'rb') as f:
            return pickle.load(f)

    def prediction(self, model, test_x, pred_kijun):
        prediction = []
        pval = model.predict(test_x, num_iteration=model.best_iteration)
        for p in pval:
            if p[1] > pred_kijun and (p[0] < 1 - pred_kijun and p[2] < 1 - pred_kijun and p[3] < 1 - pred_kijun):
                prediction.append(1)
            elif p[2] > pred_kijun and (p[0] < 1 - pred_kijun and p[1] < 1 - pred_kijun and p[3] < 1 - pred_kijun):
                prediction.append(2)
            elif p[3] > pred_kijun and (p[0] < 1 - pred_kijun and p[1] < 1 - pred_kijun and p[2] < 1 - pred_kijun):
                prediction.append(3)
            else:
                prediction.append(0)
        return prediction


    def prediction2(self, model, test_x):
        prediction = []
        pval = model.predict(test_x, num_iteration=model.best_iteration)
        for p in pval:
            prediction.append(p.argmax())
        return prediction

    @jit
    def calc_buysell_accuracy(self, predictions, test_y):
        num = predictions.count(1) + predictions.count(2)
        matched = 0
        y = np.array(test_y)
        for i in range(len(predictions)):
            if predictions[i] == 1 and y[i] == 1 or predictions[i] == 2 and y[i] == 2:
                matched += 1
        return float(matched) / float(num)



