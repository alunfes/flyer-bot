from SimData import SimTickData
from Strategy import Strategy
from SimAccount import SimAccount

class Sim:
    @classmethod
    def sim_sma_trend_follow(cls, ticks:SimTickData, sma_term, from_i, to_i):
        cls.ac = SimAccount(ticks)
        for i in range(from_i,to_i):
            Strategy.sma_trend_follow(sma_term,ticks,i,cls.ac)


    @classmethod
