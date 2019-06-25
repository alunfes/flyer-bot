from SimData import SimTickData
from Strategy import Strategy
from SimAccount import SimAccount

'''
sim levelの最適化：
loss cut, pt, 

'''


class Sim:
    @classmethod
    def sim_lgbmodel(cls, stdata, pl_kijun, ac):
        print('sim length:' + str(stdata.dt[0]) + str(stdata.dt[-1]))
        for i in range(len(stdata.prediction) - 1):
            dd = Strategy.model_prediction(pl_kijun, stdata, i, ac)
            if dd.cancel:
                ac.cancel_order(i, stdata.dt[i], stdata.ut[i])
            elif dd.side != '':
                ac.entry_order(dd.side, dd.price, dd.size, dd.type, dd.expire, i, stdata.dt[i], stdata.ut[i],
                               stdata.price[i])
            ac.move_to_next(i, stdata.dt[i], stdata.ut[i], stdata.price[i])
        ac.last_day_operation(len(stdata.prediction) - 1, stdata.dt[len(stdata.prediction) - 1],
                              stdata.ut[len(stdata.prediction) - 1], stdata.price[len(stdata.prediction) - 1])
        return ac

    @classmethod
    def sim_bp(cls, stdata, pl, ls, ac):
        print('sim length:' + str(stdata.dt[0]) + str(stdata.dt[-1]))
        for i in range(len(stdata.prediction) - 1):
            dd = Strategy.model_bp_prediction(pl, ls, stdata, i, ac)
            if dd.side != '':
                ac.entry_order(dd.side, dd.price, dd.size, dd.type, dd.expire, pl, ls, i, stdata.dt[i], stdata.ut[i],
                               stdata.price[
                                   i])  # ntry_order(self, side, price, size, type, expire, pl, ls, i, dt, ut, tick_price):
            ac.move_to_next(i, stdata.dt[i], stdata.ut[i], stdata.price[i])
        ac.last_day_operation(len(stdata.prediction) - 1, stdata.dt[len(stdata.prediction) - 1],
                              stdata.ut[len(stdata.prediction) - 1], stdata.price[len(stdata.prediction) - 1])
        return ac

    @classmethod
    def sim_sp(cls, stdata, pl, ls, ac):
        print('sim length:' + str(stdata.dt[0]) + str(stdata.dt[-1]))
        for i in range(len(stdata.prediction) - 1):
            dd = Strategy.model_sp_prediction(pl, ls, stdata, i, ac)
            if dd.side != '':
                ac.entry_order(dd.side, dd.price, dd.size, dd.type, dd.expire, pl, ls, i, stdata.dt[i], stdata.ut[i],
                               stdata.price[
                                   i])  # ntry_order(self, side, price, size, type, expire, pl, ls, i, dt, ut, tick_price):
            ac.move_to_next(i, stdata.dt[i], stdata.ut[i], stdata.price[i])
        ac.last_day_operation(len(stdata.prediction) - 1, stdata.dt[len(stdata.prediction) - 1],
                              stdata.ut[len(stdata.prediction) - 1], stdata.price[len(stdata.prediction) - 1])
        return ac

    @classmethod
    def sim_buysell(cls, stdata, pl, ls, ac):
        print('sim length:' + str(stdata.dt[0]) + str(stdata.dt[-1]))
        for i in range(len(stdata.prediction) - 1):
            dd = Strategy.model_buysell_prediction(pl, ls, stdata, i, ac)
            if dd.side != '':
                ac.entry_order(dd.side, dd.price, dd.size, dd.type, dd.expire, pl, ls, i, stdata.dt[i], stdata.ut[i],
                               stdata.price[
                                   i])  # ntry_order(self, side, price, size, type, expire, pl, ls, i, dt, ut, tick_price):
            ac.move_to_next(i, stdata.dt[i], stdata.ut[i], stdata.price[i])
        ac.last_day_operation(len(stdata.prediction) - 1, stdata.dt[len(stdata.prediction) - 1],
                              stdata.ut[len(stdata.prediction) - 1], stdata.price[len(stdata.prediction) - 1])
        return ac

    @classmethod
    def sim_lgbmodel_opt(cls, stdata, pl, losscut, time_exit, zero_three_exit_loss, zero_three_exit_profit,
                         ac: SimAccount):
        print('sim length:' + str(stdata.dt[0]) + str(stdata.dt[-1]))
        one_min_checker = 1
        for i in range(len(stdata.prediction) - 1):
            if ac.suspension_flg == False:
                dd = Strategy.model_prediction_opt(time_exit, zero_three_exit_loss, zero_three_exit_profit, stdata, i,
                                                   ac)
                if dd.cancel:
                    ac.cancel_order(i, stdata.dt[i], stdata.ut[i])
                elif dd.side != '':
                    ac.entry_order(dd.side, dd.price, dd.size, dd.type, dd.expire, pl, losscut, i, stdata.dt[i],
                                   stdata.ut[i], stdata.price[i])
            if i > one_min_checker * 60:
                one_min_checker += 1
                ac.suspension_flg = False
            ac.move_to_next(i, stdata.dt[i], stdata.ut[i], stdata.price[i])
        ac.last_day_operation(len(stdata.prediction) - 1, stdata.dt[len(stdata.prediction) - 1],
                              stdata.ut[len(stdata.prediction) - 1], stdata.price[len(stdata.prediction) - 1])
        return ac

    @classmethod
    def sim_ema_trend_follow(cls, stdata, ac):
        print('sim length:' + str(stdata.dt[0]) + str(stdata.dt[-1]))
        for i in range(len(stdata.prediction) - 1):
            dd = Strategy.ema_trend_follow(stdata, i, ac)
            if dd.side != '':
                ac.entry_order(dd.side, dd.price, dd.size, dd.type, dd.expire, i, stdata.dt[i], stdata.ut[i],
                               stdata.price[i])
            ac.move_to_next(i, stdata.dt[i], stdata.ut[i], stdata.price[i])
        ac.last_day_operation(len(stdata.prediction) - 1, stdata.dt[len(stdata.prediction) - 1],
                              stdata.ut[len(stdata.prediction) - 1], stdata.price[len(stdata.prediction) - 1])
        return ac

    @classmethod
    def sim_ema_trend_contrarian(cls, stdata, ac):
        print('sim length:' + str(stdata.dt[0]) + str(stdata.dt[-1]))
        for i in range(len(stdata.prediction) - 1):
            dd = Strategy.ema_trend_contrarian(stdata, i, ac)
            if dd.side != '':
                ac.entry_order(dd.side, dd.price, dd.size, dd.type, dd.expire, i, stdata.dt[i], stdata.ut[i],
                               stdata.price[i])
            ac.move_to_next(i, stdata.dt[i], stdata.ut[i], stdata.price[i])
        ac.last_day_operation(len(stdata.prediction) - 1, stdata.dt[len(stdata.prediction) - 1],
                              stdata.ut[len(stdata.prediction) - 1], stdata.price[len(stdata.prediction) - 1])
        return ac

    @classmethod
    def sim_ema_tftc_switch(cls, stdata, ac, pl_sma_term):
        print('sim length:' + str(stdata.dt[0]) + str(stdata.dt[-1]))
        tf_ac = SimAccount()
        tc_ac = SimAccount()
        tf_pl_sma = []
        tc_pl_sma = []
        tf_pl_sum = 0
        tc_pl_sum = 0
        sim_min_count = 60
        switch_flg = 0  # 0:tf, 1:tc
        for i in range(len(stdata.prediction) - 1):
            # sim for tf
            dd = Strategy.ema_trend_follow(stdata, i, tf_ac)
            if dd.side != '':
                tf_ac.entry_order(dd.side, dd.price, dd.size, dd.type, dd.expire, i, stdata.dt[i], stdata.ut[i],
                                  stdata.price[i])
            tf_ac.move_to_next(i, stdata.dt[i], stdata.ut[i], stdata.price[i])
            # sim for tc
            dd = Strategy.ema_trend_contrarian(stdata, i, tc_ac)
            if dd.side != '':
                tc_ac.entry_order(dd.side, dd.price, dd.size, dd.type, dd.expire, i, stdata.dt[i], stdata.ut[i],
                                  stdata.price[i])
            tc_ac.move_to_next(i, stdata.dt[i], stdata.ut[i], stdata.price[i])
        return ac

