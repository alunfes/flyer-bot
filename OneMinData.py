from numba import jit


class OneMinData:
    def initialize(self):
        self.num_crypto_data = 0
        self.unix_time = []
        self.dt = []
        self.open = []
        self.high = []
        self.low = []
        self.close = []
        self.size = []
        self.ema = {}
        self.ema_kairi = {}
        self.ema_gra = {}
        self.dema = {}
        self.dema_kairi = {}
        self.dema_gra = {}
        self.midprice = {}
        self.rsi = {}
        self.momentum = {}
        self.rate_of_change = {}
        self.williams_R = {}
        self.beta = {}
        self.tsf = {}
        self.correl = {}
        self.adx = {}
        self.aroon_os = {}
        self.cci = {}
        self.dx = {}
        self.macd = {}
        self.macdsignal = {}
        self.macdhist = {}

        self.normalized_ave_true_range = []
        self.three_outside_updown = []
        self.breakway = []
        self.dark_cloud_cover = []
        self.dragonfly_doji = []
        self.updown_sidebyside_white_lines = []
        self.haramisen = []
        self.hikkake_pattern = []
        self.neck_pattern = []
        self.upsidedownside_gap_three_method = []
        self.sar = []
        self.bop = []
        self.future_side = []

    def cut_data(self, num_data):
        self.unix_time = self.unix_time[-num_data:]
        self.dt = self.dt[-num_data:]
        self.open = self.open[-num_data:]
        self.high = self.high[-num_data:]
        self.low = self.low[-num_data:]
        self.close = self.close[-num_data:]
        self.size = self.size[-num_data:]
        for k in self.ema: #assume term is same in all index except macd
            self.ema_kairi[k] = self.ema_kairi[k][-num_data:]
            self.ema_gra[k] = self.ema_gra[k][-num_data:]
            self.dema[k] = self.dema[k][-num_data:]
            self.dema_kairi[k] = self.dema_kairi[k][-num_data:]
            self.dema_gra[k] = self.dema_gra[k][-num_data:]
            self.rsi[k] = self.rsi[k][-num_data:]
            self.midprice = self.midprice[k][-num_data:]
            self.momentum[k] = self.momentum[k][-num_data:]
            self.rate_of_change[k] = self.rate_of_change[k][-num_data:]
            self.williams_R[k] = self.williams_R[k][-num_data:]
            self.beta[k] = self.beta[k][-num_data:]
            self.tsf[k] = self.tsf[k][-num_data:]
            self.correl[k] = self.correl[k][-num_data:]
            self.ema[k] = self.ema[k][-num_data:]
            self.adx[k] = self.adx[k][-num_data:]
            self.aroon_os[k] = self.aroon_os[k][-num_data:]
            self.cci[k] = self.cci[k][-num_data:]
            self.dx[k] = self.dx[k][-num_data:]
            if k in self.macd:
                self.macd[k] = self.macd[k][-num_data:]
                self.macdsignal[k] = self.macdsignal[k][-num_data:]
                self.macdhist[k] = self.macdhist[k][-num_data:]
        self.normalized_ave_true_range = self.normalized_ave_true_range[-num_data:]
        self.three_outside_updown = self.three_outside_updown[-num_data:]
        self.breakway = self.breakway[-num_data:]
        self.dark_cloud_cover = self.dark_cloud_cover[-num_data:]
        self.dragonfly_doji = self.dragonfly_doji[-num_data:]
        self.updown_sidebyside_white_lines = self.updown_sidebyside_white_lines[-num_data:]
        self.haramisen = self.haramisen[-num_data:]
        self.hikkake_pattern = self.hikkake_pattern[-num_data:]
        self.neck_pattern = self.neck_pattern[-num_data:]
        self.upsidedownside_gap_three_method = self.upsidedownside_gap_three_method[-num_data:]
        self.sar = self.sar[-num_data:]
        self.bop = self.bop[-num_data:]
        self.future_side = self.future_side[-num_data:]

    @jit
    def del_data(self, num_remain_data):
        if len(self.unix_time) > num_remain_data:
            del self.unix_time[:-num_remain_data]
            del self.dt[:-num_remain_data]
            del self.open[:-num_remain_data]
            del self.high[:-num_remain_data]
            del self.low[:-num_remain_data]
            del self.close[:-num_remain_data]
            del self.size[:-num_remain_data]
            for k in self.ema: #assume term is same in all index
                del self.ema_kairi[k][:-num_remain_data]
                del self.ema_gra[k][:-num_remain_data]
                del self.dema[k][:-num_remain_data]
                del self.dema_kairi[k][:-num_remain_data]
                del self.dema_gra[k][:-num_remain_data]
                del self.midprice[k][:-num_remain_data]
                del self.rsi[k][:-num_remain_data]
                del self.momentum[k][:-num_remain_data]
                del self.rate_of_change[k][:-num_remain_data]
                del self.williams_R[k][:-num_remain_data]
                del self.beta[k][:-num_remain_data]
                del self.tsf[k][:-num_remain_data]
                del self.correl[k][:-num_remain_data]
                del self.ema[k][:-num_remain_data]
                del self.adx[k][:-num_remain_data]
                del self.aroon_os[k][:-num_remain_data]
                del self.cci[k][:-num_remain_data]
                del self.dx[k][:-num_remain_data]
                if k in self.macd:
                    del self.macd[k][:-num_remain_data]
                    del self.macdsignal[k][:-num_remain_data]
                    del self.macdhist[k][:-num_remain_data]
            del self.normalized_ave_true_range[:-num_remain_data]
            del self.three_outside_updown[:-num_remain_data]
            del self.breakway[:-num_remain_data]
            del self.dark_cloud_cover[:-num_remain_data]
            del self.dragonfly_doji[:-num_remain_data]
            del self.updown_sidebyside_white_lines[:-num_remain_data]
            del self.haramisen[:-num_remain_data]
            del self.hikkake_pattern[:-num_remain_data]
            del self.neck_pattern[:-num_remain_data]
            del self.upsidedownside_gap_three_method[:-num_remain_data]
            del self.sar[:-num_remain_data]
            del self.bop[:-num_remain_data]
            del self.future_side[:-num_remain_data]

    def add_and_pop(self, unix_time, dt, open, high, low, close, size):
        self.unix_time.append(unix_time)
        self.unix_time.pop(0)
        self.dt.append(dt)
        self.dt.pop(0)
        self.open.append(open)
        self.open.pop(0)
        self.high.append(high)
        self.high.pop(0)
        self.low.append(low)
        self.low.pop(0)
        self.close.append(close)
        self.close.pop(0)
        self.size.append(size)
        self.size.pop(0)