from datetime import datetime
from catboost import Pool
from BotAccount import BotAccount
from LogMaster import LogMaster
from Trade import Trade
from LineNotification import LineNotification
from WebsocketMaster import TickData
from OneMinMarketData import OneMinMarketData
from CatModel import CatModel
from SystemFlg import SystemFlg
from CryptowatchDataGetter import CryptowatchDataGetter
import time
import pytz
import math


'''
all order should be used wait till boarded or execution
'''
class FlyerBot:
    def __init__(self):
        SystemFlg.initialize()
        self.margin_rate = 120.0
        self.leverage = 15.0
        self.ac = BotAccount()
        self.JST = pytz.timezone('Asia/Tokyo')
        self.last_ohlc_min = -1
        self.last_sync_ut = datetime.now(self.JST).timestamp()
        self.model = None
        self.cbm = None
        self.prediction = [0]
        self.pred_side = ''

    def cancel_order(self):
        print('cancel order')
        status = Trade.cancel_and_wait_completion(self.ac.order_id)
        if len(status) > 0:
            print('cancel failed, partially executed')
            oid = Trade.order(status['side'].lower(),0,status['executed_size'],'market',1)
            LogMaster.add_log('action_message - cancel order - cancel failed and partially executed. closed position.',self.prediction[0],self.ac)
            self.ac.initialize_order()
        else:
            LogMaster.add_log('action_message - cancelled order',self.prediction[0],self.ac)
            self.ac.initialize_order()

    def entry_market_order(self, side, size):
        if self.ac.order_side == '':
            print('entry market order')
            status = Trade.market_order_wait_till_execution(side, size)
            if status is not None:
                LogMaster.add_log('Market order entry has been executed. '+' side='+status['side']+' size='+str(status['executed_size'])+' price='+str(status['price']), self.prediction,self.ac)
                print('Market order entry has been executed. '+' side='+status['side']+' size='+str(status['executed_size'])+' price='+str(status['price']))
                self.ac.update_holding(side,status['average_price'],status['executed_size'],status['child_order_acceptance_id'])
            else:
                LogMaster.add_log('Market order has been failed.'+' side='+side +' '+str(size),self.prediction,self.ac)
                print('market order failed!')
        else:
            print('Entry market order - order is already exitst!')
            LogMaster.add_log('Entry market order - order is already exitst!',self.prediction, self.ac)


    def entry_limit_order(self,side, price, size, expire):
        if self.ac.order_side == '':
            print('entry limit order')
            oid = Trade.order_wait_till_boarding(side,price,size,expire)
            if len(oid) > 10:
                self.ac.update_order(side, price, 0, size,oid,1,'new entry')
                LogMaster.add_log('Entry limit order. ' + ' side=' + side + ' size=' + str(size) + ' price=' + str(price), self.prediction[0],self.ac)
                print('Entry limit order. ' + ' side=' + side + ' size=' + str(size) + ' price=' + str(price))
            else:
                LogMaster.add_log('Limit order has been failed.' + ' side=' + side + ' ' + str(size),self.prediction[0], self.ac)
                print('limit order failed!')
        else:
            print('Entry limit order - order is already exitst!')
            LogMaster.add_log('Entry limit order - order is already exitst!', self.ac)

    def entry_pl_order(self):
        side = 'buy' if self.ac.holding_side == 'sell' else 'sell'
        pl_kijun = self.calc_opt_pl()
        price = self.ac.holding_price + pl_kijun if self.ac.holding_side == 'buy' else self.ac.holding_price - pl_kijun
        res = Trade.order(side, price, self.ac.holding_size, 'limit', 1440)
        if len(res) > 10:
            self.ac.update_order(side,price,0,self.ac.holding_size,res,1440,'pl order')
            print('pl order: side = {}, price = {}, outstanding size = {}'.format(self.ac.order_side, self.ac.order_price, self.ac.order_outstanding_size))
            LogMaster.add_log('action_message - pl entry for ' + side + ' @' + str(price) + ' x' + str(self.ac.order_outstanding_size), self.prediction[0],self.ac)
        else:
            LogMaster.add_log('action_message - failed pl entry!',self.prediction[0],self.ac)
            print('failed pl order!')

    def exit_order(self):
        if self.ac.holding_side != '':
            print('quick exit order')
            status = Trade.market_order_wait_till_execution('buy' if self.ac.holding_side == 'sell' else 'sell',self.ac.holding_size)
            if status is not None:
                if status['child_order_state'] == 'COMPLETED':
                    self.ac.initialize_holding()
                    print('exit order completed!')
                    LogMaster.add_log('exit order completed!',self.prediction[0],self.ac)
                    return 0
                else:
                    print('something wrong in exit order! '+str(status))
                    LogMaster.add_log('something wrong in exit order! '+str(status), self.prediction[0], self.ac)
                    return -1
            else:
                print('something wrong in exit order! '+str(status))
                LogMaster.add_log('something wrong in exit order! '+str(status), self.prediction[0], self.ac)
                return -1



    def start_flyer_bot(self, num_term, window_term, pl_kijun, future_period):
        self.__bot_initializer(num_term, window_term, pl_kijun, future_period)
        self.start_time = time.time()
        while SystemFlg.get_system_flg():
            self.__check_system_maintenance()
            self.__update_ohlc()
            if self.ac.holding_side == '' and self.ac.order_side == '': #no position no order
                if self.prediction[0] == 1 or self.prediction[0] == 2:
                    self.entry_market_order(self.pred_side, self.calc_opt_size())
            elif self.ac.holding_side != '' and self.ac.order_side == '': #holding position and no order
                self.entry_pl_order()
            elif (self.ac.holding_side == 'buy' and (self.prediction[0] == 2)) or (self.ac.holding_side == 'sell' and (self.prediction[0] == 1)):  # ポジションが判定と逆の時にexit,　もしplがあればキャンセル。。
                if self.ac.order_status != '':
                    self.cancel_order() #最初にキャンセルしないとexit order出せない。
                self.exit_order()
            elif self.ac.holding_side != '' and self.ac.order_side != '':#sleep until next ohlc update when prediction is same as
                time.sleep(1)
            if self.ac.order_side != '' and abs(self.ac.order_price - TickData.get_ltp()) <= 5000:
                res = self.ac.check_execution()
                LogMaster.add_log(res,self.prediction[0],self.ac)
            if Trade.flg_api_limit:
                time.sleep(60)
                print('Bot sleeping for 60sec due to API access limitation')
            else:
                time.sleep(0.5)


    def __bot_initializer(self, num_term, window_term, pl_kijun, future_period):
        Trade.cancel_all_orders()
        self.pl_kijun = pl_kijun
        print('bot - updating crypto data..')
        LogMaster.add_log('action_message - bot - updating crypto data..', 0,self.ac)
        CryptowatchDataGetter.get_and_add_to_csv()
        self.last_ohlc_min =  datetime.now(self.JST).minute-1
        print('bot - initializing MarketData3..')
        OneMinMarketData.initialize_for_bot(num_term, window_term, future_period, pl_kijun, num_term + 1)
        self.model = CatModel()
        print('bot - generating training data')
        LogMaster.add_log('bot - training xgb model..', self.prediction[0], self.ac)
        print('bot - training model..')
        self.cbm = self.model.read_dump_model('./Model/cat_model.dat')
        print('bot - load cat model completed..')
        print('bot - started bot loop.')
        LogMaster.add_log('action_message - bot - started bot loop.', self.prediction[0], self.ac)

    def __check_system_maintenance(self):
        if (datetime.now(tz=self.JST).hour == 3 and datetime.now(tz=self.JST).minute >= 48):
            print('sleep waiting for system maintenance')
            if self.ac.order_side != '':
                self.cancel_order()
            time.sleep(780)  # wait for daily system maintenace
            print('resumed from maintenance time sleep')

    def __update_ohlc(self): #should download ohlc soon after when it finalized
        if self.last_ohlc_min < datetime.now(self.JST).minute or (self.last_ohlc_min == 59 and datetime.now(self.JST).minute == 0):
            for i in range(10):
                res, omd = CryptowatchDataGetter.get_data_after_specific_ut(OneMinMarketData.ohlc.unix_time[-1])
                if res == 0 and len(omd.unix_time) > 0:
                    print('updated ohlc at ' + str(datetime.now(self.JST)))
                    for i in range(len(omd.dt)):
                        OneMinMarketData.ohlc.add_and_pop(omd.unix_time[i], omd.dt[i], omd.open[i], omd.high[i], omd.low[i],omd.close[i], omd.size[i])
                    self.last_ohlc_min = datetime.now(self.JST).minute
                    OneMinMarketData.update_for_bot()
                    df = OneMinMarketData.generate_df_for_bot()
                    pred_x = self.model.generate_bot_pred_data(df)
                    self.prediction = self.cbm.predict(Pool(pred_x))
                    #self.pred_side = self.prediction[0].map({0: 'no', 1: 'buy', 2: 'sell', 3: 'both'}).astype(str)
                    self.pred_side = str(int(self.prediction[0][0])).translate(str.maketrans({'0':'no', '1':'buy', '2':'sell', '3':'both'}))
                    self.ac.calc_collateral_change()
                    self.ac.sync_position_order()
                    LogMaster.add_log('updated ohlc at '+str(datetime.now(self.JST)), self.prediction[0], self.ac)
                    print('dt={}, open={},high={},low={},close={}'.format(OneMinMarketData.ohlc.dt[-1], OneMinMarketData.ohlc.open[-1],OneMinMarketData.ohlc.high[-1], OneMinMarketData.ohlc.low[-1], OneMinMarketData.ohlc.close[-1]))
                    print('prediction={},holding_side={},holding_price={},holding_size={}'.format(self.prediction[0],self.ac.holding_side,self.ac.holding_price,self.ac.holding_size))
                    print('private access per 300sec={}'.format(Trade.total_access_per_300s))
                    LineNotification.send_notification(LogMaster.get_latest_performance())
                    self.elapsed_time = time.time() - self.start_time
                    return 0
                else:
                    time.sleep(1)
                    print('cryptowatch download trial'+str(i+1))
            print('ohlc download error')
            LogMaster.add_log('ohlc download error',self.prediction[0],self.ac)
            return -1


    def __sync_order_poisition(self):
        self.ac.sync_position_order()


    def calc_opt_size(self):
        collateral = Trade.get_collateral()['collateral']
        if TickData.get_1m_std() > 10000:
            multiplier = 0.5
            print('changed opt size multiplier to 0.5')
            LogMaster.add_log('action_message - changed opt size multiplier to 0.5',self.prediction[0],self.ac)
            LineNotification.send_error('changed opt size multiplier to 0.5')
        else:
            multiplier = 1.5
        size = round((multiplier * collateral * self.margin_rate) / TickData.get_ltp() * 1.0 / self.leverage, 2)
        return size

    def calc_opt_pl(self):
        if TickData.get_1m_std() > 10000:
            newpl = self.pl_kijun * math.log((TickData.get_1m_std() / 100000)) + 5
            print('changed opt pl kijun to '+str(newpl))
            LogMaster.add_log('action_message - changed opt pl kijun to '+str(newpl),self.prediction[0],self.ac)
            LineNotification.send_error('changed opt pl kijun to '+str(newpl))
            return newpl
        else:
            return self.pl_kijun


if __name__ == '__main__':
    SystemFlg.initialize()
    TickData.initialize()
    Trade.initialize()
    LogMaster.initialize()
    LineNotification.initialize()
    fb = FlyerBot()
    fb.start_flyer_bot(500,2,100000,11) #num_term, window_term, pl_kijun, future_period
    #'JRF20190526-142616-930215'
    #JRF20190526-143431-187560