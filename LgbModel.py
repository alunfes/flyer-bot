import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from numba import jit
import numpy as np
import pandas as pd
import pickle

import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split

from numba import jit
import numpy as np
import pandas as pd


class LgbModel:
    def generate_data(self, df: pd.DataFrame, test_size=0.2):
        dff = df
        dff['future_side'] = dff['future_side'].map({'no': 0, 'buy': 1, 'sell': 2, 'both': 3}).astype(int)
        # dff = dff.drop(['dt','open','high','low','close','size'],axis = 1)
        dff = dff.drop(['dt', 'size'], axis=1)
        size = int(round(dff['future_side'].count() * (1 - test_size)))
        train_x = dff.drop('future_side', axis=1).iloc[0:size]
        train_y = dff['future_side'].iloc[0:size]
        test_x = dff.drop('future_side', axis=1).iloc[size:]
        test_y = dff['future_side'].iloc[size:]
        #        print('train length='+str(len(train_y)))
        #        print('test length='+str(len(test_y)))
        return train_x, test_x, train_y, test_y

    def generate_bsp_data(self, df: pd.DataFrame, side, train_size=0.6, valid_size=0.2):
        dfx = None
        dfy = None
        col_name = 'bp' if side == 'buy' else 'sp'
        dfx = df.drop(['dt', 'size', col_name], axis=1)
        dfy = df[col_name]
        dfy.columns = [col_name]
        train_x, test_x, train_y, test_y = train_test_split(dfx, dfy, train_size=train_size, shuffle=False)
        count_buy_in_train = train_y.values.sum()
        non_buy_list = []
        buy_list = []

        for i in range(len(train_y)):  # train_y = 0のdataをリスト化
            if train_y.iloc[i] == 0:
                non_buy_list.append(train_x.iloc[i])
            elif train_y.iloc[i] == 1:
                buy_list.append(train_x.iloc[i])
        if len(buy_list) != count_buy_in_train:
            print('len(buy_list) is not matched with count_buy_in_train !!')
        # リストからランダムにデータを選択して、buy pointsのtrain dataと合わせてdfを作る
        selected = random.sample(non_buy_list, count_buy_in_train)
        new_buy_points = [1] * len(buy_list)
        new_buy_points.extend([0] * count_buy_in_train)
        new_train_df = pd.DataFrame()
        new_train_df = new_train_df.append(buy_list)
        new_train_df = new_train_df.append(selected)
        if side == 'buy':
            new_train_df = new_train_df.assign(bp=new_buy_points)
        else:
            new_train_df = new_train_df.assign(sp=new_buy_points)
        train_y = new_train_df[col_name]
        new_train_df = new_train_df.drop([col_name], axis=1)
        train_xx, valid_x, train_yy, valid_y = train_test_split(new_train_df, train_y, train_size=1.0 - valid_size,
                                                                random_state=42)
        print('buy sell point data description:')
        print('side=', side)
        print('train_x', train_xx.shape)
        print('train_y', train_yy.shape)
        print('test_x', test_x.shape)
        print('test_y', test_y.shape)
        print('valid_x', valid_x.shape)
        print('valid_y', valid_y.shape)
        return train_xx, test_x, train_yy, test_y, valid_x, valid_y

    def train(self, train_x, train_y):
        # print('training data description')
        # print('train_x:',train_x.shape)
        # print('train_y:',train_y.shape)
        train_start_ind = OneMinMarketData.check_matched_index(train_x)
        print('train period:', OneMinMarketData.ohlc.dt[train_start_ind],
              OneMinMarketData.ohlc.dt[train_start_ind + len(train_y)])
        train = lgb.Dataset(train_x.values.astype(np.float32), train_y.values.astype(np.float32))
        lgbm_params = {
            'objective': 'multiclass',
            'num_class': 4,
            'boosting': 'dart',
            'tree_learner': 'data',
            'learning_rate': 0.05,
            'num_iterations': 200,
            #            'device':'gpu',
        }
        model = lgb.train(lgbm_params, train)
        return model

    def train_params(self, train_x, train_y, params):
        # print('training data description')
        # print('train_x:',train_x.shape)
        # print('train_y:',train_y.shape)
        train = lgb.Dataset(train_x.values.astype(np.float32), train_y.values.astype(np.float32))
        model = lgb.train(params, train)
        return model

    def train_params_with_validations(self, train_x, train_y, valid_x, valid_y, params):
        # print('training data description')
        # print('train_x:',train_x.shape)
        # print('train_y:',train_y.shape)
        train_start_ind = OneMinMarketData.check_matched_index(train_x)
        print('train period:', OneMinMarketData.ohlc.dt[train_start_ind],
              OneMinMarketData.ohlc.dt[train_start_ind + len(train_y)])
        train = lgb.Dataset(train_x.values.astype(np.float32), train_y.values.astype(np.float32))
        lgb_eval = lgb.Dataset(valid_x.values.astype(np.float32), valid_y.values.astype(np.float32), reference=train)
        model = lgb.train(params, train, valid_sets=lgb_eval)
        return model

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

    def bp_prediciton(self, model, test_x, kijun):
        pred = model.predict(test_x, num_iteration=model.best_iteration)
        res = []
        for i in pred:
            if i >= kijun:
                res.append(1)
            else:
                res.append(0)
        return res

    def bp_buysell_prediction(self, prediction_buy, prediction_sell, upper_kijun, lower_kijun):
        if len(prediction_buy) == len(prediction_sell):
            res = []
            for i in range(len(prediction_buy)):
                if prediction_buy[i] >= upper_kijun and prediction_sell[i] <= lower_kijun:
                    res.append(1)
                elif prediction_sell[i] >= upper_kijun and prediction_buy[i] <= lower_kijun:
                    res.append(-1)
                else:
                    res.append(0)
            return res
        else:
            print('bp_buysell_prediction - buy prediction and sell predition num is not matched!!')
            return []

    def calc_buysell_accuracy(self, predictions, test_y):
        num = predictions.count(1) + predictions.count(2)
        matched = 0
        y = np.array(test_y)
        for i in range(len(predictions)):
            if predictions[i] == 1 and y[i] == 1 or predictions[i] == 2 and y[i] == 2:
                matched += 1
        if num > 0:
            return float(matched) / float(num)
        else:
            return 0

    def calc_total_accuracy(self, predictions, test_y):
        matched = 0
        y = np.array(test_y)
        for i in range(len(predictions)):
            if predictions[i] == y[i]:
                matched += 1
        return float(matched) / float(len(predictions))

    def calc_bp_accuracy(self, predictions, test_y):
        matched = 0
        y = np.array(test_y)
        for i in range(len(predictions)):
            if predictions[i] == 1 and y[i] == 1:
                matched += 1
        if sum(predictions) > 0:
            return float(matched) / float(sum(predictions))
        else:
            return 0


