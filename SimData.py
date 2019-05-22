import pandas as pd
import numpy as np
from datetime import datetime
#from datetime import timedelta

class SimData:
    @classmethod
    def initialize(cls, num_term, window_term, initial_data_vol):
        cls.num_term = num_term
        cls.window_term = window_term
        cls.read_data_from_csv('./Data/ticks.csv')
        cls.ticks.del_data(initial_data_vol)
        cls.calc_all_index()
        print('Completed initialization of SimData')
        print('num_data={}, : from_dt={}, - to_dt={}'.format(len(cls.ticks.dt), cls.ticks.dt[0], cls.ticks.dt[-1]))

    @classmethod
    def read_data_from_csv(cls, path):
        print('Reading Data..')
        cls.ticks = SimTickData()
        #df = pd.read_csv(path, header=num_skip, names=['ut','price','size'])
        df = pd.read_csv(path)
        cls.ticks.ut = list(df['ut'])
        cls.ticks.dt = list(df['dt'])
        cls.ticks.price = list(df['price'])
        cls.ticks.size = list(df['size'])
        return df

    @classmethod
    def convert_to_sec(cls):
        pass

    @classmethod
    def calc_all_index(cls):
        print('Calculating Index Data..')
        num = round(cls.num_term / cls.window_term)
        if num > 1:
            for i in range(num):
                term = cls.window_term * (i + 1)
                if term > 1:
                    cls.ticks.sma[term] = cls.__calc_sma(term)
                    cls.ticks.sma_kairi[term] = cls.__calc_sma_kairi(term)
                    cls.ticks.sma_incli[term] = cls.__calc_sma_incli(term)


    @classmethod
    def __calc_sma(cls, term):
        return list(pd.Series(cls.ticks.price).rolling(window=term).mean())

    @classmethod
    def __calc_sma_kairi(cls, term):
        return list([x / y for (x,y) in zip(cls.ticks.price, cls.ticks.sma[term])])

    @classmethod
    def __calc_sma_incli(cls, term):
        return np.gradient(cls.ticks.sma[term])


class SimTickData:
    def __init__(self):
        self.ut = []
        self.dt = []
        self.price = []
        self.size = []
        self.sma = {}
        self.sma_kairi = {}
        self.sma_incli = {}

    def del_data(self, num_remain_data):
        if len(self.ut) > num_remain_data:
            print('deleted tick data for initialization. (use '+str(num_remain_data)+' data for simualtion.')
            del self.ut[:-num_remain_data]
            del self.dt[:-num_remain_data]
            del self.price[:-num_remain_data]
            del self.size[:-num_remain_data]
            for k in self.sma:  # assume term is same in all index
                del self.sma[k][:-num_remain_data]
                del self.sma_kairi[k][:-num_remain_data]
                del self.sma_incli[k][:-num_remain_data]


if __name__ == '__main__':
    SimData.initialize(1000,2)