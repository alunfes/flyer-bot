from SimData import SimTickData
from SimAccount import SimAccount

class Strategy:
    @classmethod
    def sma_trend_follow(cls, sma_term, ticks:SimTickData, i, ac:SimAccount):
        dd = DecisionData()
        if ac.holding_side == '' and  ac.order_side =='': #no position / no order
            if ticks.sma_incli[sma_term][i] >= 0:
                dd.set_decision('buy',ticks.price[i],cls.__calc_opt_size(ticks.price[i],ac),'market',False,10)
            else:
                dd.set_decision('sell', ticks.price[i], cls.__calc_opt_size(ticks.price[i], ac), 'market', False, 10)
        elif ac.holding_side == '' and  ac.order_side != '':
            if ac.order_side == 'buy' and ticks.sma_incli[sma_term][i] >= 0:
                pass
            elif ac.order_side == 'buy' and ticks.sma_incli[sma_term][i] < 0:
                dd.set_decision('sell', ticks.price[i], cls.__calc_opt_size(ticks.price[i], ac), 'market', True, 10)
        return dd

    @classmethod
    def __calc_opt_size(cls, price, ac):
        return round((ac.asset * ac.leverage) / (price * 1.0 * ac.base_margin_rate), 2)

class DecisionData:
    def __init__(self):
        self.side = ''
        self.size = 0
        self.price = 0
        self.type = 0
        self.cancel = False
        self.expire = 0

    def set_decision(self, side, price, size, type, cancel, expire):
        self.side = side
        self.price = price
        self.size = size
        self.type = type
        self.cancel = cancel
        self.expire = expire








