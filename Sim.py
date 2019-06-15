from SimData import SimTickData
from Strategy import Strategy
from SimAccount import SimAccount

class Sim:
    @classmethod
    def sim_lgbmodel_opt(cls, stdata, pl, losscut, time_exit, zero_three_exit_pl, zero_three_exit_loss, zero_three_exit_profit, ac:SimAccount):
        print('sim length:' + str(stdata.dt[0]) + str(stdata.dt[-1]))
        for i in range(len(stdata.prediction) -1):
            dd =Strategy.model_prediction_opt(time_exit, zero_three_exit_pl, zero_three_exit_loss, zero_three_exit_profit, stdata, i, ac)
            if dd.cancel:
                ac.cancel_order(i, stdata.dt[i], stdata.ut[i])
            elif dd.side != '':
                cls.ac.entry_order(dd.side,dd.price,dd.size,dd.type,dd.expire, pl, losscut, i, stdata.dt[i], stdata.ut[i], stdata.price[i])
            cls.ac.move_to_next(i, stdata.dt[i], stdata.ut[i], stdata.price[i])
        ac.last_day_operation(len(stdata.prediction) - 1, stdata.dt[len(stdata.prediction) - 1],stdata.ut[len(stdata.prediction) - 1], stdata.price[len(stdata.prediction) - 1])
        return cls.ac



    @classmethod
    def sim_sma_trend_follow(cls, ticks:SimTickData, sma_term, from_i, to_i):
        cls.ac = SimAccount(ticks)
        for i in range(from_i,to_i):
            dd = Strategy.sma_trend_follow(sma_term,ticks,i,cls.ac)
            if dd.cancel == False and dd.price >0:
                cls.ac.entry_order(dd.side,dd.price,dd.size,dd.type,dd.expire)
            elif dd.cancel:
                cls.ac.cancel_order(i)
            cls.ac.move_to_next(i)
        cls.ac.last_day_operation(to_i)
        return cls.ac

    @classmethod
    def sim_lgbmodel(cls, stdata, pl_kijun):
        cls.ac = SimAccount(stdata)
        for i in range(len(stdata.prediction)-1):
            dd = Strategy.model_prediction(pl_kijun, stdata,i,cls.ac)
            if dd.cancel:
                cls.ac.cancel_order(i)
            elif dd.side != '':
                cls.ac.entry_order(dd.side,dd.price,dd.size,dd.type,dd.expire)
            cls.ac.move_to_next(i)
        cls.ac.last_day_operation(len(stdata.prediction)-1)
        return cls.ac



        for i in range(from_i,to_i):
            dd = Strategy.sma_trend_follow(sma_term,ticks,i,cls.ac)
            if dd.cancel == False and dd.price >0:
                cls.ac.entry_order(dd.side,dd.price,dd.size,dd.type,dd.expire)
            elif dd.cancel:
                cls.ac.cancel_order(i)
            cls.ac.move_to_next(i)
        cls.ac.last_day_operation(to_i)
        return cls.ac

