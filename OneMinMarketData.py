from OneMinData import OneMinData
import numpy as np
import pandas as pd
import talib as ta
from datetime import datetime, timedelta, timezone



'''
using ta-lib for index calc.
'''


def calc_all_index_wrap(term, open, high, low, close, avep):
    omd = OneMinData()
    omd.initialize()
    omd.ema[term] = OneMinMarketData.calc_ema(term, close)
    omd.ema_ave[term] = OneMinMarketData.calc_ema(term, avep)
    omd.ema_kairi[term] = OneMinMarketData.calc_ema_kairi(term, close)
    omd.ema_gra[term] = OneMinMarketData.calc_ema_gra(term, omd.ema[term])
    omd.dema[term] = OneMinMarketData.calc_dema(term, close)
    omd.dema_ave[term] = OneMinMarketData.calc_dema(term, avep)
    omd.dema_kairi[term] = OneMinMarketData.calc_dema_kairi(term, close)
    omd.dema_gra[term] = OneMinMarketData.calc_dema_gra(term, omd.dema[term])
    omd.midprice[term] = OneMinMarketData.calc_midprice(term, high, low)
    omd.momentum[term] = OneMinMarketData.calc_momentum(term, close)
    omd.momentum_ave[term] = OneMinMarketData.calc_momentum(term, avep)
    omd.rate_of_change[term] = OneMinMarketData.calc_rate_of_change(term, close)
    omd.rsi[term] = OneMinMarketData.calc_rsi(term, close)
    omd.williams_R[term] = OneMinMarketData.calc_williams_R(term, high, low, close)
    omd.beta[term] = OneMinMarketData.calc_beta(term, high, low)
    omd.tsf[term] = OneMinMarketData.calc_time_series_forecast(term, close)
    omd.correl[term] = OneMinMarketData.calc_correl(term, high, low)
    omd.linear_reg[term] = OneMinMarketData.calc_linear_reg(term, close)
    omd.linear_reg_angle[term] = OneMinMarketData.calc_linear_reg_angle(term, close)
    omd.linear_reg_intercept[term] = OneMinMarketData.calc_linear_reg_intercept(term, close)
    omd.linear_reg_slope[term] = OneMinMarketData.calc_linear_reg_slope(term, close)
    omd.stdv[term] = OneMinMarketData.calc_stdv(term, close)
    omd.var[term] = OneMinMarketData.calc_var(term, close)
    omd.linear_reg_ave[term] = OneMinMarketData.calc_linear_reg(term, avep)
    omd.linear_reg_angle_ave[term] = OneMinMarketData.calc_linear_reg_angle(term, avep)
    omd.linear_reg_intercept_ave[term] = OneMinMarketData.calc_linear_reg_intercept(term, avep)
    omd.linear_reg_slope_ave[term] = OneMinMarketData.calc_linear_reg_slope(term, avep)
    omd.stdv_ave[term] = OneMinMarketData.calc_stdv(term, avep)
    omd.var_ave[term] = OneMinMarketData.calc_var(term, avep)
    omd.adx[term] = OneMinMarketData.calc_adx(term, high, low, close)
    omd.aroon_os[term] = OneMinMarketData.calc_aroon_os(term, high, low)
    omd.cci[term] = OneMinMarketData.calc_cci(term, high, low, close)
    omd.dx[term] = OneMinMarketData.calc_dx(term, high, low, close)
    if term >= 10:
        omd.macd[term], omd.macdsignal[term], omd.macdhist[term] = OneMinMarketData.calc_macd(close,
                                                                                              int(float(term) / 2.0),
                                                                                              term,
                                                                                              int(float(term) / 3.0))
        omd.macd[term] = list(omd.macd[term])
        omd.macdsignal[term] = list(omd.macdsignal[term])
        omd.macdhist[term] = list(omd.macdhist[term])
        omd.macd_ave[term], omd.macdsignal_ave[term], omd.macdhist_ave[term] = OneMinMarketData.calc_macd(avep, int(
            float(term) / 2.0), term, int(float(term) / 3.0))
        omd.macd_ave[term] = list(omd.macd_ave[term])
        omd.macdsignal_ave[term] = list(omd.macdsignal_ave[term])
        omd.macdhist_ave[term] = list(omd.macdhist_ave[term])
    return {term: omd}


class OneMinMarketData:
    @classmethod
    def initialize_for_bot(cls, num_term, window_term, future_side_period, future_side_kijun, initial_data_vol):
        cls.num_term = num_term
        cls.window_term = window_term
        cls.future_side_period = future_side_period
        cls.future_side_kijun = future_side_kijun
        cls.ohlc = cls.read_from_csv('./Data/one_min_data.csv')
        cls.ohlc.del_data(initial_data_vol)
        # cls.__calc_all_index(False)
        cls.__calc_all_index2_main(False)

    @classmethod
    def update_for_bot(cls):
        cls.__calc_all_index2_main(True)

    @classmethod
    def read_from_csv(cls, file_name):
        ohlc = OneMinData()
        ohlc.initialize()
        df = pd.read_csv(file_name)
        ohlc.dt = list(map(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'), list(df['dt'])))
        ohlc.unix_time = list(df['unix_time'])
        ohlc.open = list(df['open'])
        ohlc.high = list(df['high'])
        ohlc.low = list(df['low'])
        ohlc.close = list(df['close'])
        ohlc.size = list(df['size'])
        return ohlc

    @classmethod
    def __calc_all_index(cls, flg_for_bot):
        cls.ohlc.ave_price = cls.calc_ave_price(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
        num = round(cls.num_term / cls.window_term)
        if num > 1:
            for i in range(num):
                term = cls.window_term * (i + 1)
                if term > 1:
                    cls.ohlc.ema[term] = cls.calc_ema(term, cls.ohlc.close)
                    cls.ohlc.ema_ave[term] = cls.calc_ema(term, cls.ohlc.ave_price)
                    cls.ohlc.ema_kairi[term] = cls.calc_ema_kairi(term, cls.ohlc.close)
                    cls.ohlc.ema_gra[term] = cls.calc_ema_gra(term, cls.ohlc.ema[term])
                    cls.ohlc.dema[term] = cls.calc_dema(term, cls.ohlc.close)
                    cls.ohlc.dema_ave[term] = cls.calc_dema(term, cls.ohlc.ave_price)
                    cls.ohlc.dema_kairi[term] = cls.calc_dema_kairi(term, cls.ohlc.close)
                    cls.ohlc.dema_gra[term] = cls.calc_dema_gra(term, cls.ohlc.dema[term])
                    cls.ohlc.midprice[term] = cls.calc_midprice(term, cls.ohlc.high, cls.ohlc.low)
                    cls.ohlc.momentum[term] = cls.calc_momentum(term, cls.ohlc.close)
                    cls.ohlc.momentum_ave[term] = cls.calc_momentum(term, cls.ohlc.ave_price)
                    cls.ohlc.rate_of_change[term] = cls.calc_rate_of_change(term, cls.ohlc.close)
                    cls.ohlc.rsi[term] = cls.calc_rsi(term, cls.ohlc.close)
                    cls.ohlc.williams_R[term] = cls.calc_williams_R(term, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
                    cls.ohlc.beta[term] = cls.calc_beta(term, cls.ohlc.high, cls.ohlc.low)
                    cls.ohlc.tsf[term] = cls.calc_time_series_forecast(term, cls.ohlc.close)
                    cls.ohlc.correl[term] = cls.calc_correl(term, cls.ohlc.high, cls.ohlc.low)
                    cls.ohlc.linear_reg[term] = cls.calc_linear_reg(term, cls.ohlc.close)
                    cls.ohlc.linear_reg_angle[term] = cls.calc_linear_reg_angle(term, cls.ohlc.close)
                    cls.ohlc.linear_reg_intercept[term] = cls.calc_linear_reg_intercept(term, cls.ohlc.close)
                    cls.ohlc.linear_reg_slope[term] = cls.calc_linear_reg_slope(term, cls.ohlc.close)
                    cls.ohlc.stdv[term] = cls.calc_stdv(term, cls.ohlc.close)
                    cls.ohlc.var[term] = cls.calc_var(term, cls.ohlc.close)
                    cls.ohlc.linear_reg_ave[term] = cls.calc_linear_reg(term, cls.ohlc.ave_price)
                    cls.ohlc.linear_reg_angle_ave[term] = cls.calc_linear_reg_angle(term, cls.ohlc.ave_price)
                    cls.ohlc.linear_reg_intercept_ave[term] = cls.calc_linear_reg_intercept(term, cls.ohlc.ave_price)
                    cls.ohlc.linear_reg_slope_ave[term] = cls.calc_linear_reg_slope(term, cls.ohlc.ave_price)
                    cls.ohlc.stdv_ave[term] = cls.calc_stdv(term, cls.ohlc.ave_price)
                    cls.ohlc.var_ave[term] = cls.calc_var(term, cls.ohlc.ave_price)
                    cls.ohlc.adx[term] = cls.calc_adx(term, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
                    cls.ohlc.aroon_os[term] = cls.calc_aroon_os(term, cls.ohlc.high, cls.ohlc.low)
                    cls.ohlc.cci[term] = cls.calc_cci(term, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
                    cls.ohlc.dx[term] = cls.calc_dx(term, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
                    if term >= 10:
                        cls.ohlc.macd[term], cls.ohlc.macdsignal[term], cls.ohlc.macdhist[term] = cls.calc_macd(
                            cls.ohlc.close, int(float(term) / 2.0), term, int(float(term) / 3.0))
                        cls.ohlc.macd[term] = list(cls.ohlc.macd[term])
                        cls.ohlc.macdsignal[term] = list(cls.ohlc.macdsignal[term])
                        cls.ohlc.macdhist[term] = list(cls.ohlc.macdhist[term])
                        cls.ohlc.macd_ave[term], cls.ohlc.macdsignal_ave[term], cls.ohlc.macdhist_ave[
                            term] = cls.calc_macd(cls.ohlc.ave_price, int(float(term) / 2.0), term,
                                                  int(float(term) / 3.0))
                        cls.ohlc.macd_ave[term] = list(cls.ohlc.macd_ave[term])
                        cls.ohlc.macdsignal_ave[term] = list(cls.ohlc.macdsignal_ave[term])
                        cls.ohlc.macdhist_ave[term] = list(cls.ohlc.macdhist_ave[term])
        cls.ohlc.normalized_ave_true_range = cls.calc_normalized_ave_true_range(cls.ohlc.high, cls.ohlc.low,
                                                                                cls.ohlc.close)
        cls.ohlc.three_outside_updown = cls.calc_three_outside_updown(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low,
                                                                      cls.ohlc.close)
        cls.ohlc.breakway = cls.calc_breakway(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
        cls.ohlc.dark_cloud_cover = cls.calc_dark_cloud_cover(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low,
                                                              cls.ohlc.close)
        cls.ohlc.dragonfly_doji = cls.calc_dragonfly_doji(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
        cls.ohlc.updown_sidebyside_white_lines = cls.calc_updown_sidebyside_white_lines(cls.ohlc.open, cls.ohlc.high,
                                                                                        cls.ohlc.low, cls.ohlc.close)
        cls.ohlc.haramisen = cls.calc_haramisen(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
        cls.ohlc.hikkake_pattern = cls.calc_hikkake_pattern(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
        cls.ohlc.neck_pattern = cls.calc_neck_pattern(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
        cls.ohlc.upsidedownside_gap_three_method = cls.calc_upsidedownside_gap_three_method(cls.ohlc.open,
                                                                                            cls.ohlc.high, cls.ohlc.low,
                                                                                            cls.ohlc.close)
        cls.ohlc.sar = cls.calc_sar(cls.ohlc.high, cls.ohlc.low, 0.02, 0.2)
        cls.ohlc.bop = cls.calc_bop(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
        if flg_for_bot == False:
            cls.ohlc.future_side = cls.calc_future_side(cls.future_side_period, cls.future_side_kijun, cls.ohlc)

    '''
    joblib multiprocessing
    '''

    @classmethod
    def __calc_all_index2_main(cls, flg_for_bot):
        cls.ohlc.ave_price = cls.calc_ave_price(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
        '''
        num = round(cls.num_term / cls.window_term)
        args = range(num)
        n_cores = multi.cpu_count()
        p = Pool(processes=multiprocessing.cpu_count(),initializer=func1, initargs=())
        res = p.map(cls.calc_all_index2, args)
        '''

        terms = []
        n_cores = multi.cpu_count()
        num = int(round(cls.num_term / cls.window_term))
        for i in range(num):
            terms.append(cls.window_term * (i + 1))
        omd_list = joblib.Parallel(n_jobs=n_cores)([delayed(calc_all_index_wrap)(term, cls.ohlc.open, cls.ohlc.high,
                                                                                 cls.ohlc.low, cls.ohlc.close,
                                                                                 cls.ohlc.ave_price) for term in terms])
        for omd in omd_list:
            key = int(list(omd.keys())[0])
            cls.ohlc.ema[key] = omd[key].ema[key]
            cls.ohlc.ema_ave[key] = omd[key].ema_ave[key]
            cls.ohlc.ema_kairi[key] = omd[key].ema_kairi[key]
            cls.ohlc.ema_gra[key] = omd[key].ema_gra[key]
            cls.ohlc.dema[key] = omd[key].dema[key]
            cls.ohlc.dema_ave[key] = omd[key].dema_ave[key]
            cls.ohlc.dema_kairi[key] = omd[key].dema_kairi[key]
            cls.ohlc.dema_gra[key] = omd[key].dema_gra[key]
            cls.ohlc.midprice[key] = omd[key].midprice[key]
            cls.ohlc.momentum[key] = omd[key].momentum[key]
            cls.ohlc.momentum_ave[key] = omd[key].momentum_ave[key]
            cls.ohlc.rate_of_change[key] = omd[key].rate_of_change[key]
            cls.ohlc.rsi[key] = omd[key].rsi[key]
            cls.ohlc.williams_R[key] = omd[key].williams_R[key]
            cls.ohlc.beta[key] = omd[key].beta[key]
            cls.ohlc.tsf[key] = omd[key].tsf[key]
            cls.ohlc.correl[key] = omd[key].correl[key]
            cls.ohlc.linear_reg[key] = omd[key].linear_reg[key]
            cls.ohlc.linear_reg_angle[key] = omd[key].linear_reg_angle[key]
            cls.ohlc.linear_reg_intercept[key] = omd[key].linear_reg_intercept[key]
            cls.ohlc.linear_reg_slope[key] = omd[key].linear_reg_slope[key]
            cls.ohlc.stdv[key] = omd[key].stdv[key]
            cls.ohlc.var[key] = omd[key].var[key]
            cls.ohlc.linear_reg_ave[key] = omd[key].linear_reg_ave[key]
            cls.ohlc.linear_reg_angle_ave[key] = omd[key].linear_reg_angle_ave[key]
            cls.ohlc.linear_reg_intercept_ave[key] = omd[key].linear_reg_intercept_ave[key]
            cls.ohlc.linear_reg_slope_ave[key] = omd[key].linear_reg_slope_ave[key]
            cls.ohlc.stdv_ave[key] = omd[key].stdv_ave[key]
            cls.ohlc.var_ave[key] = omd[key].var_ave[key]
            cls.ohlc.adx[key] = omd[key].adx[key]
            cls.ohlc.aroon_os[key] = omd[key].aroon_os[key]
            cls.ohlc.cci[key] = omd[key].cci[key]
            cls.ohlc.dx[key] = omd[key].dx[key]
            if key >= 10:
                cls.ohlc.macd[key] = omd[key].macd[key]
                cls.ohlc.macdsignal[key] = omd[key].macdsignal[key]
                cls.ohlc.macdhist[key] = omd[key].macdhist[key]
                cls.ohlc.macd_ave[key] = omd[key].macd_ave[key]
                cls.ohlc.macdsignal_ave[key] = omd[key].macdsignal_ave[key]
                cls.ohlc.macdhist_ave[key] = omd[key].macdhist_ave[key]
        cls.ohlc.normalized_ave_true_range = cls.calc_normalized_ave_true_range(cls.ohlc.high, cls.ohlc.low,
                                                                                cls.ohlc.close)
        cls.ohlc.three_outside_updown = cls.calc_three_outside_updown(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low,
                                                                      cls.ohlc.close)
        cls.ohlc.breakway = cls.calc_breakway(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
        cls.ohlc.dark_cloud_cover = cls.calc_dark_cloud_cover(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low,
                                                              cls.ohlc.close)
        cls.ohlc.dragonfly_doji = cls.calc_dragonfly_doji(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
        cls.ohlc.updown_sidebyside_white_lines = cls.calc_updown_sidebyside_white_lines(cls.ohlc.open, cls.ohlc.high,
                                                                                        cls.ohlc.low, cls.ohlc.close)
        cls.ohlc.haramisen = cls.calc_haramisen(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
        cls.ohlc.hikkake_pattern = cls.calc_hikkake_pattern(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
        cls.ohlc.neck_pattern = cls.calc_neck_pattern(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
        cls.ohlc.upsidedownside_gap_three_method = cls.calc_upsidedownside_gap_three_method(cls.ohlc.open,
                                                                                            cls.ohlc.high, cls.ohlc.low,
                                                                                            cls.ohlc.close)
        cls.ohlc.sar = cls.calc_sar(cls.ohlc.high, cls.ohlc.low, 0.02, 0.2)
        cls.ohlc.bop = cls.calc_bop(cls.ohlc.open, cls.ohlc.high, cls.ohlc.low, cls.ohlc.close)
        if flg_for_bot == False:
            cls.ohlc.future_side = cls.calc_future_side(cls.future_side_period, cls.future_side_kijun, cls.ohlc)

    '''
    dema, adx, macdはnum_term * 2くらいnanが発生する
    print(df.isnull().sum())
    '''

    @classmethod
    def generate_df(cls):
        cut_size = cls.num_term * 2
        end = len(cls.ohlc.close) - cls.future_side_period
        df = pd.DataFrame()
        df = df.assign(dt=cls.ohlc.dt[cut_size:end])
        df = df.assign(open=np.array(cls.ohlc.open[cut_size:end], dtype=np.float32))
        df = df.assign(high=np.array(cls.ohlc.high[cut_size:end], dtype=np.float32))
        df = df.assign(low=np.array(cls.ohlc.low[cut_size:end], dtype=np.float32))
        df = df.assign(close=np.array(cls.ohlc.close[cut_size:end], dtype=np.float32))
        df = df.assign(ave_price=np.array(cls.ohlc.ave_price[cut_size:end], dtype=np.float32))
        df = df.assign(size=np.array(cls.ohlc.size[cut_size:end], dtype=np.float32))

        def __make_col_df(df, data, col_name):
            for k in data:
                col = col_name + str(k)
                df = df.assign(col=np.array(data[k][cut_size:end], dtype=np.float32))
                df.rename(columns={'col': col}, inplace=True)
            return df

        df = __make_col_df(df, cls.ohlc.ema, 'ema')
        df = __make_col_df(df, cls.ohlc.ema_ave, 'ema_ave')
        df = __make_col_df(df, cls.ohlc.ema_kairi, 'ema_kairi')
        df = __make_col_df(df, cls.ohlc.dema_kairi, 'dema_kairi')
        df = __make_col_df(df, cls.ohlc.ema_gra, 'ema_gra')
        df = __make_col_df(df, cls.ohlc.dema, 'dema')
        df = __make_col_df(df, cls.ohlc.dema_ave, 'dema_ave')
        df = __make_col_df(df, cls.ohlc.dema_gra, 'dema_gra')
        df = __make_col_df(df, cls.ohlc.midprice, 'midprice')
        df = __make_col_df(df, cls.ohlc.momentum, 'momentum')
        df = __make_col_df(df, cls.ohlc.momentum_ave, 'momentum_ave')
        df = __make_col_df(df, cls.ohlc.rate_of_change, 'rate_of_change')
        df = __make_col_df(df, cls.ohlc.rsi, 'rsi')
        df = __make_col_df(df, cls.ohlc.williams_R, 'williams_R')
        df = __make_col_df(df, cls.ohlc.beta, 'beta')
        df = __make_col_df(df, cls.ohlc.tsf, 'tsf')
        df = __make_col_df(df, cls.ohlc.correl, 'correl')
        df = __make_col_df(df, cls.ohlc.linear_reg, 'linear_reg')
        df = __make_col_df(df, cls.ohlc.linear_reg_angle, 'linear_reg_angle')
        df = __make_col_df(df, cls.ohlc.linear_reg_intercept, 'linear_reg_intercept')
        df = __make_col_df(df, cls.ohlc.linear_reg_slope, 'linear_reg_slope')
        df = __make_col_df(df, cls.ohlc.stdv, 'stdv')
        df = __make_col_df(df, cls.ohlc.var, 'var')
        df = __make_col_df(df, cls.ohlc.linear_reg_ave, 'linear_reg_ave')
        df = __make_col_df(df, cls.ohlc.linear_reg_angle_ave, 'linear_reg_angle_ave')
        df = __make_col_df(df, cls.ohlc.linear_reg_intercept_ave, 'linear_reg_intercept_ave')
        df = __make_col_df(df, cls.ohlc.linear_reg_slope_ave, 'linear_reg_slope_ave')
        df = __make_col_df(df, cls.ohlc.stdv_ave, 'stdv_ave')
        df = __make_col_df(df, cls.ohlc.var_ave, 'var_ave')
        df = __make_col_df(df, cls.ohlc.adx, 'adx')
        df = __make_col_df(df, cls.ohlc.aroon_os, 'aroon_os')
        df = __make_col_df(df, cls.ohlc.cci, 'cci')
        df = __make_col_df(df, cls.ohlc.dx, 'dx')
        df = __make_col_df(df, cls.ohlc.macd, 'macd')
        df = __make_col_df(df, cls.ohlc.macdsignal, 'macdsignal')
        df = __make_col_df(df, cls.ohlc.macdhist, 'macdhist')
        df = __make_col_df(df, cls.ohlc.macd_ave, 'macd_ave')
        df = __make_col_df(df, cls.ohlc.macdsignal_ave, 'macdsignal_ave')
        df = __make_col_df(df, cls.ohlc.macdhist_ave, 'macdhist_ave')
        df = df.assign(
            normalized_ave_true_range=np.array(cls.ohlc.normalized_ave_true_range[cut_size:end], dtype=np.float32))
        df = df.assign(three_outside_updown=cls.ohlc.three_outside_updown[cut_size:end])
        df = df.assign(breakway=cls.ohlc.breakway[cut_size:end])
        df = df.assign(dark_cloud_cover=cls.ohlc.dark_cloud_cover[cut_size:end])
        df = df.assign(dragonfly_doji=cls.ohlc.dragonfly_doji[cut_size:end])
        df = df.assign(updown_sidebyside_white_lines=cls.ohlc.updown_sidebyside_white_lines[cut_size:end])
        df = df.assign(haramisen=cls.ohlc.haramisen[cut_size:end])
        df = df.assign(hikkake_pattern=cls.ohlc.hikkake_pattern[cut_size:end])
        df = df.assign(neck_pattern=cls.ohlc.neck_pattern[cut_size:end])
        df = df.assign(upsidedownside_gap_three_method=cls.ohlc.upsidedownside_gap_three_method[cut_size:end])
        df = df.assign(sar=np.array(cls.ohlc.sar[cut_size:end], dtype=np.float32))
        df = df.assign(bop=np.array(cls.ohlc.bop[cut_size:end], dtype=np.float32))
        df = df.assign(future_side=cls.ohlc.future_side[cut_size:])
        print('future side unique val')
        print(df['future_side'].value_counts(dropna=False, normalize=True))
        return df

    @classmethod
    def generate_df_for_bot(cls):
        df = pd.DataFrame()
        df = df.assign(dt=cls.ohlc.dt[-1:])
        df = df.assign(open=cls.ohlc.open[-1:])
        df = df.assign(high=cls.ohlc.high[-1:])
        df = df.assign(low=cls.ohlc.low[-1:])
        df = df.assign(close=cls.ohlc.close[-1:])
        df = df.assign(ave_price=cls.ohlc.ave_price[-1:])
        df = df.assign(size=cls.ohlc.size[-1:])

        def __make_col_df(df, data, col_name):
            for k in data:
                col = col_name + str(k)
                df = df.assign(col=data[k][-1:])
                df.rename(columns={'col': col}, inplace=True)
            return df

        df = __make_col_df(df, cls.ohlc.ema, 'ema')
        df = __make_col_df(df, cls.ohlc.ema_ave, 'ema_ave')
        df = __make_col_df(df, cls.ohlc.ema_kairi, 'ema_kairi')
        df = __make_col_df(df, cls.ohlc.dema_kairi, 'dema_kairi')
        df = __make_col_df(df, cls.ohlc.ema_gra, 'ema_gra')
        df = __make_col_df(df, cls.ohlc.dema, 'dema')
        df = __make_col_df(df, cls.ohlc.dema_ave, 'dema_ave')
        df = __make_col_df(df, cls.ohlc.dema_gra, 'dema_gra')
        df = __make_col_df(df, cls.ohlc.midprice, 'midprice')
        df = __make_col_df(df, cls.ohlc.momentum, 'momentum')
        df = __make_col_df(df, cls.ohlc.momentum_ave, 'momentum_ave')
        df = __make_col_df(df, cls.ohlc.rate_of_change, 'rate_of_change')
        df = __make_col_df(df, cls.ohlc.rsi, 'rsi')
        df = __make_col_df(df, cls.ohlc.williams_R, 'williams_R')
        df = __make_col_df(df, cls.ohlc.beta, 'beta')
        df = __make_col_df(df, cls.ohlc.tsf, 'tsf')
        df = __make_col_df(df, cls.ohlc.correl, 'correl')
        df = __make_col_df(df, cls.ohlc.linear_reg, 'linear_reg')
        df = __make_col_df(df, cls.ohlc.linear_reg_angle, 'linear_reg_angle')
        df = __make_col_df(df, cls.ohlc.linear_reg_intercept, 'linear_reg_intercept')
        df = __make_col_df(df, cls.ohlc.linear_reg_slope, 'linear_reg_slope')
        df = __make_col_df(df, cls.ohlc.stdv, 'stdv')
        df = __make_col_df(df, cls.ohlc.var, 'var')
        df = __make_col_df(df, cls.ohlc.linear_reg_ave, 'linear_reg_ave')
        df = __make_col_df(df, cls.ohlc.linear_reg_angle_ave, 'linear_reg_angle_ave')
        df = __make_col_df(df, cls.ohlc.linear_reg_intercept_ave, 'linear_reg_intercept_ave')
        df = __make_col_df(df, cls.ohlc.linear_reg_slope_ave, 'linear_reg_slope_ave')
        df = __make_col_df(df, cls.ohlc.stdv_ave, 'stdv_ave')
        df = __make_col_df(df, cls.ohlc.var_ave, 'var_ave')
        df = __make_col_df(df, cls.ohlc.adx, 'adx')
        df = __make_col_df(df, cls.ohlc.aroon_os, 'aroon_os')
        df = __make_col_df(df, cls.ohlc.cci, 'cci')
        df = __make_col_df(df, cls.ohlc.dx, 'dx')
        df = __make_col_df(df, cls.ohlc.macd, 'macd')
        df = __make_col_df(df, cls.ohlc.macdsignal, 'macdsignal')
        df = __make_col_df(df, cls.ohlc.macdhist, 'macdhist')
        df = __make_col_df(df, cls.ohlc.macd_ave, 'macd_ave')
        df = __make_col_df(df, cls.ohlc.macdsignal_ave, 'macdsignal_ave')
        df = __make_col_df(df, cls.ohlc.macdhist_ave, 'macdhist_ave')
        df = df.assign(normalized_ave_true_range=cls.ohlc.normalized_ave_true_range[-1:])
        df = df.assign(three_outside_updown=cls.ohlc.three_outside_updown[-1:])
        df = df.assign(breakway=cls.ohlc.breakway[-1:])
        df = df.assign(dark_cloud_cover=cls.ohlc.dark_cloud_cover[-1:])
        df = df.assign(dragonfly_doji=cls.ohlc.dragonfly_doji[-1:])
        df = df.assign(updown_sidebyside_white_lines=cls.ohlc.updown_sidebyside_white_lines[-1:])
        df = df.assign(haramisen=cls.ohlc.haramisen[-1:])
        df = df.assign(hikkake_pattern=cls.ohlc.hikkake_pattern[-1:])
        df = df.assign(neck_pattern=cls.ohlc.neck_pattern[-1:])
        df = df.assign(sar=cls.ohlc.sar[-1:])
        df = df.assign(bop=cls.ohlc.bop[-1:])
        df = df.assign(upsidedownside_gap_three_method=cls.ohlc.upsidedownside_gap_three_method[-1:])
        return df

    @classmethod
    def calc_future_side(cls, future_side_period, future_side_kijun, ohlc):
        for i in range(len(ohlc.close) - future_side_period):
            buy_max = 0
            sell_max = 0
            for j in range(i, i + future_side_period):
                buy_max = max(buy_max, ohlc.high[j + 1] - ohlc.close[i])
                sell_max = max(sell_max, ohlc.close[i] - ohlc.low[j + 1])
            if buy_max >= future_side_kijun and sell_max >= future_side_kijun:
                ohlc.future_side.append('both')
            elif buy_max >= future_side_kijun and sell_max < future_side_kijun:
                ohlc.future_side.append('buy')
            elif buy_max < future_side_kijun and sell_max >= future_side_kijun:
                ohlc.future_side.append('sell')
            elif buy_max < future_side_kijun and sell_max < future_side_kijun:
                ohlc.future_side.append('no')
        return ohlc.future_side

    @classmethod
    def calc_ave_price(cls, open, high, low, close):
        return list(ta.AVGPRICE(np.array(open, dtype='f8'), np.array(high, dtype='f8'), np.array(low, dtype='f8'),
                                np.array(close, dtype='f8')))

    @classmethod
    def calc_ema(cls, term, close):
        return list(ta.EMA(np.array(close, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_ema_kairi(cls, term, close):
        kairi = []
        ema = cls.calc_ema(term, close)
        for i, em in enumerate(ema):
            if np.isnan(em):
                kairi.append(-1)
            else:
                kairi.append(100.0 * (close[i] - em) / em)
        return kairi

    @classmethod
    def calc_dema_kairi(cls, term, close):
        kairi = []
        dema = cls.calc_dema(term, close)
        for i, em in enumerate(dema):
            if np.isnan(em):
                kairi.append(-1)
            else:
                kairi.append(100.0 * (close[i] - em) / em)
        return kairi

    @classmethod
    def calc_ema_gra(cls, term, ema):
        diff = []
        for i in range(len(ema)):
            if ema[i] == np.nan:
                diff.append(np.nan)
            else:
                diff.append(ema[i] - ema[i - 1])
        return diff

    @classmethod
    def calc_dema_gra(cls, term, dema):
        diff = []
        for i in range(len(dema)):
            if dema[i] == np.nan:
                diff.append(np.nan)
            else:
                diff.append(dema[i] - dema[i - 1])
        return diff

    @classmethod
    def calc_dema(cls, term, close):
        return list(ta.DEMA(np.array(close, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_adx(cls, term, high, low, close):
        return list(
            ta.ADX(np.array(high, dtype='f8'), np.array(low, dtype='f8'), np.array(close, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_aroon_os(cls, term, high, low):
        return list(ta.AROONOSC(np.array(high, dtype='f8'), np.array(low, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_cci(cls, term, high, low, close):
        return list(
            ta.CCI(np.array(high, dtype='f8'), np.array(low, dtype='f8'), np.array(close, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_dx(cls, term, high, low, close):
        return list(
            ta.DX(np.array(high, dtype='f8'), np.array(low, dtype='f8'), np.array(close, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_midprice(cls, term, high, low):
        return list(ta.MIDPRICE(np.array(high, dtype='f8'), np.array(low, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_macd(cls, close, fastperiod=12, slowperiod=26, signalperiod=9):
        return ta.MACD(np.array(close, dtype='f8'), np.array(fastperiod, dtype='i8'), np.array(slowperiod, dtype='i8'),
                       np.array(signalperiod, dtype='i8'))

    @classmethod
    def calc_momentum(cls, term, close):
        return list(ta.MOM(np.array(close, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_rate_of_change(cls, term, close):
        return list(ta.ROC(np.array(close, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_rsi(cls, term, close):
        return list(ta.RSI(np.array(close, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_williams_R(cls, term, high, low, close):
        return list(ta.WILLR(np.array(high, dtype='f8'), np.array(low, dtype='f8'), np.array(close, dtype='f8'),
                             timeperiod=term))

    @classmethod
    def calc_beta(cls, term, high, low):
        return list(ta.BETA(np.array(high, dtype='f8'), np.array(low, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_time_series_forecast(cls, term, close):
        return list(ta.TSF(np.array(close, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_correl(cls, term, high, low):
        return list(ta.CORREL(np.array(high, dtype='f8'), np.array(low, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_linear_reg(cls, term, close):
        return list(ta.LINEARREG(np.array(close, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_linear_reg_angle(cls, term, close):
        return list(ta.LINEARREG_ANGLE(np.array(close, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_linear_reg_intercept(cls, term, close):
        return list(ta.LINEARREG_SLOPE(np.array(close, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_linear_reg_slope(cls, term, close):
        return list(ta.LINEARREG_INTERCEPT(np.array(close, dtype='f8'), timeperiod=term))

    @classmethod
    def calc_stdv(cls, term, close):
        return list(ta.STDDEV(np.array(close, dtype='f8'), timeperiod=term, nbdev=1))

    @classmethod
    def calc_var(cls, term, close):
        return list(ta.VAR(np.array(close, dtype='f8'), timeperiod=term, nbdev=1))

    @classmethod
    def calc_normalized_ave_true_range(cls, high, low, close):
        return list(ta.NATR(np.array(high, dtype='f8'), np.array(low, dtype='f8'), np.array(close, dtype='f8')))

    @classmethod
    def calc_three_outside_updown(cls, open, high, low, close):
        return list(ta.CDL3OUTSIDE(np.array(open, dtype='f8'), np.array(high, dtype='f8'), np.array(low, dtype='f8'),
                                   np.array(close, dtype='f8')))

    @classmethod
    def calc_breakway(cls, open, high, low, close):
        return list(ta.CDLBREAKAWAY(np.array(open, dtype='f8'), np.array(high, dtype='f8'), np.array(low, dtype='f8'),
                                    np.array(close, dtype='f8')))

    @classmethod
    def calc_dark_cloud_cover(cls, open, high, low, close):
        return list(
            ta.CDLDARKCLOUDCOVER(np.array(open, dtype='f8'), np.array(high, dtype='f8'), np.array(low, dtype='f8'),
                                 np.array(close, dtype='f8'), penetration=0))

    @classmethod
    def calc_dragonfly_doji(cls, open, high, low, close):
        return list(
            ta.CDLDRAGONFLYDOJI(np.array(open, dtype='f8'), np.array(high, dtype='f8'), np.array(low, dtype='f8'),
                                np.array(close, dtype='f8')))

    @classmethod
    def calc_updown_sidebyside_white_lines(cls, open, high, low, close):
        return list(
            ta.CDLGAPSIDESIDEWHITE(np.array(open, dtype='f8'), np.array(high, dtype='f8'), np.array(low, dtype='f8'),
                                   np.array(close, dtype='f8')))

    @classmethod
    def calc_haramisen(cls, open, high, low, close):
        return list(ta.CDLHARAMI(np.array(open, dtype='f8'), np.array(high, dtype='f8'), np.array(low, dtype='f8'),
                                 np.array(close, dtype='f8')))

    @classmethod
    def calc_hikkake_pattern(cls, open, high, low, close):
        return list(ta.CDLHIKKAKEMOD(np.array(open, dtype='f8'), np.array(high, dtype='f8'), np.array(low, dtype='f8'),
                                     np.array(close, dtype='f8')))

    @classmethod
    def calc_neck_pattern(cls, open, high, low, close):
        return list(ta.CDLINNECK(np.array(open, dtype='f8'), np.array(high, dtype='f8'), np.array(low, dtype='f8'),
                                 np.array(close, dtype='f8')))

    @classmethod
    def calc_sar(cls, high, low, accelation, maximum):
        return list(ta.SAR(np.array(high, dtype='f8'), np.array(low, dtype='f8'), np.array(accelation, dtype='f8'),
                           np.array(maximum, dtype='f8')))

    @classmethod
    def calc_bop(cls, open, high, low, close):
        return list(ta.BOP(np.array(open, dtype='f8'), np.array(high, dtype='f8'), np.array(low, dtype='f8'),
                           np.array(close, dtype='f8')))

    @classmethod
    def calc_upsidedownside_gap_three_method(cls, open, high, low, close):
        return list(
            ta.CDLXSIDEGAP3METHODS(np.array(open, dtype='f8'), np.array(high, dtype='f8'), np.array(low, dtype='f8'),
                                   np.array(close, dtype='f8')))

    @classmethod
    def check_matched_index(cls, test_x):
        test = list(test_x['open'])
        op = cls.ohlc.open
        for i in range(len(op)):
            flg = True
            for j in range(30):
                if test[j] != op[i + j]:
                    flg = False
                    break
            if flg:
                return i
        print('no matche index found!')
        return -1

    @classmethod
    def generate_tick_pred_data(cls, prediction, start_ind):  # assume index 0 of ohlc and prediction is matched
        tick = []
        dt = []
        ut = []
        stdata = SimTickData()
        start_dt = cls.ohlc.dt[0]
        start_ut = cls.ohlc.unix_time[0]
        pred = []
        split_num = 60
        cls.ohlc.del_data(len(cls.ohlc.close) - start_ind)
        for i in range(len(prediction)):
            om_tick = []
            if cls.ohlc.open[i] < cls.ohlc.close[i]:  # open, low, high, close
                ol = cls.ohlc.open[i] - cls.ohlc.low[i]
                lh = cls.ohlc.high[i] - cls.ohlc.low[i]
                hc = cls.ohlc.high[i] - cls.ohlc.close[i]
                sec_width = (ol + lh + hc) / split_num
                if sec_width > 0:
                    om_tick.extend(
                        list(np.round(np.linspace(cls.ohlc.open[i], cls.ohlc.low[i], round(ol / sec_width)))))
                    om_tick.extend(
                        list(np.round(np.linspace(cls.ohlc.low[i], cls.ohlc.high[i], round(lh / sec_width)))))
                    om_tick.extend(
                        list(np.round(np.linspace(cls.ohlc.high[i], cls.ohlc.close[i], round(hc / sec_width)))))
                else:
                    om_tick.extend([tick[-1]] * split_num)
            else:  # open, high, low, close
                oh = cls.ohlc.high[i] - cls.ohlc.open[i]
                hl = cls.ohlc.high[i] - cls.ohlc.low[i]
                lc = cls.ohlc.close[i] - cls.ohlc.low[i]
                sec_width = (oh + hl + lc) / split_num
                if sec_width > 0:
                    om_tick.extend(
                        list(np.round(np.linspace(cls.ohlc.open[i], cls.ohlc.high[i], round(oh / sec_width)))))
                    om_tick.extend(
                        list(np.round(np.linspace(cls.ohlc.high[i], cls.ohlc.low[i], round(hl / sec_width)))))
                    om_tick.extend(
                        list(np.round(np.linspace(cls.ohlc.low[i], cls.ohlc.close[i], round(lc / sec_width)))))
                else:
                    om_tick.extend([tick[-1]] * split_num)
            if split_num - len(om_tick) > 0:
                om_tick.extend([om_tick[-1] * (split_num - len(om_tick))])
            elif split_num - len(om_tick) < 0:
                del om_tick[-(len(om_tick) - split_num):]
            tick.extend(om_tick)
            ut.extend([start_ut + (j + 1) for j in range(split_num)])
            dt.extend([start_dt + timedelta(seconds=k + 1) for k in range(split_num)])
            start_ut = ut[-1]
            start_dt = dt[-1]
            if i == 0:
                pred.extend([0] * split_num)
            else:
                pred.extend([prediction[i - 1]] * split_num)
        stdata.dt.extend(dt)
        stdata.ut.extend(ut)
        stdata.price.extend(tick)
        stdata.prediction.extend(pred)
        return stdata