from SimTickData import TickData
from SimAccount import SimAccount


class Strategy:
    '''
    assume all orders are market order
    '''
    @classmethod
    def sma_trend_follow(cls, sma_term, ticks:SimTickData, i, ac:SimAccount):
        dd = DecisionData()
        d_side = 'buy' if ticks.sma_incli[sma_term][i] >= 0 else 'sell'
        if ac.holding_side =='': #no position
            dd.set_decision(d_side, ticks.price[i], cls.__calc_opt_size(ticks.price[i], ac), 'market', False, 10)
        else: #position
            if ac.holding_side == d_side: #do nothing
                pass
            else: #exit and re-entry
                dd.set_decision(d_side, ticks.price[i], ac.holding_size + cls.__calc_opt_size(ticks.price[i], ac), 'market', True, 10)
        return dd


    @classmethod
    def model_prediction(cls, pl_kijun, pred, ticks, i, ac: SimAccount):
        dd = DecisionData()
        pred_side = pred[i].map({0: 'no', 1: 'buy', 2: 'sell', 3: 'both'}).astype(str)
        if ac.holding_side == '' and ac.order_side == '' and pred_side != 'other': #no position no order
            dd.set_decision(pred_side, cls.__calc_opt_size(ticks.price[i], ac), 'market', False, 10)
        elif (ac.holding_side == 'buy' or ac.holding_side == 'sell') and (pred_side =='buy' or pred_side == 'sell') and \
                ac.holding_side != pred_side and ac.order_side != '' and ac.order_side != ac.holding_side and ac.order_type =='limit': #holding side != pred side and pl ordering -> cancel pl order
            dd.set_decision(pred_side,0, 0,'', True, 10) #cancel order
        elif (ac.holding_side == 'buy' or ac.holding_side == 'sell') and (pred_side =='buy' or pred_side == 'sell') and \
                ac.holding_side != pred_side and ac.order_side =='':
            dd.set_decision(pred_side, ac.holding_size + cls.__calc_opt_size(ticks.price[i], ac), 'market', False, 10)
        elif ac.holding_side != '' and ac.order_side =='': #place pl order
            dd.set_decision('buy' if ac.holding_side =='sell' else 'sell', ac.holding_price + pl_kijun if ac.holding_side == 'buy' else ac.holding_price - pl_kijun,
                            ac.holding_size,'limit',False,360000)
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
        self.expire = 0  #sec

    def set_decision(self, side, price, size, type, cancel, expire):
        self.side = side
        self.price = price
        self.size = size
        self.type = type
        self.cancel = cancel
        self.expire = expire








