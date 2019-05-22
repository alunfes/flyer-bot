from SimData import SimTickData
from Strategy import Strategy
from SimAccount import SimAccount

class Sim:
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
