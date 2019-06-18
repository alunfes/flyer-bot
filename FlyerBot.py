from datetime import datetime
from catboost import Pool
from LgbModel import LgbModel
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
        self.bst = None
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
            self.ac.update_order(side,0,0,size,'',10,'')
            ltp = TickData.get_ltp()
            res, size, price, order_id = Trade.market_order_wait_till_execution3(side, size)
            if res == 0:
                print(side+' entry market order has been executed.'+'price='+str(price)+', size='+str(size))
                self.ac.update_holding(side,price,size,order_id)
                self.ac.sync_position_order()
                self.ac.calc_collateral_change()
                self.ac.calc_pl(TickData.get_ltp())
                self.ac.add_order_exec_price_gap(price,ltp,side)
                print('holding_side={},holding_price={},holding_size={},total_pl={},collateral={},collateral_change={},realized_pl={}'.
                      format(self.ac.holding_side,self.ac.holding_price,self.ac.holding_size,self.ac.total_pl,self.ac.collateral,self.ac.collateral_change,self.ac.realized_pl))
                LogMaster.add_log('Market order entry has been executed. ' + ' side=' + side + ' size=' + str(size)+ ' price=' + str(price), self.prediction[0],self.ac)

            else:
                LogMaster.add_log('Market order has been failed.'+' side='+side +' '+str(size),self.prediction[0], self.ac)
                print('market order failed!')
            self.ac.initialize_order()
        else:
            print('Entry market order - order is already exitst!')
            LogMaster.add_log('Entry market order - order is already exitst!', self.prediction[0], self.ac)


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



    def start_flyer_bot(self, num_term, window_term, pl_kijun, future_period, zero_three_exit_when_loss):
        self.__bot_initializer(num_term, window_term, pl_kijun, future_period)
        self.start_time = time.time()
        self.fixed_order_size = 0.05
        while SystemFlg.get_system_flg():
            self.__check_system_maintenance()
            self.__update_ohlc()
            if self.ac.holding_side == '' and self.ac.order_side == '': #no position no order
                if self.prediction[0] == 1 or self.prediction[0] == 2:
                    self.entry_market_order(self.pred_side, self.fixed_order_size)
            elif self.ac.holding_side != '' and self.ac.order_side == '': #holding position and no order
                self.entry_pl_order()
            elif (self.ac.holding_side == 'buy' and (self.prediction[0] == 2)) or (self.ac.holding_side == 'sell' and (self.prediction[0] == 1)):  # ポジションが判定と逆の時にexit,　もしplがあればキャンセル。。
                if self.ac.order_status != '':
                    self.cancel_order() #最初にキャンセルしないとexit order出せない。
                self.entry_market_order(self.pred_side, self.ac.holding_size)
                self.entry_market_order(self.pred_side, self.fixed_order_size)
            #elif self.ac.holding_side != '' and self.ac.order_side != '':#sleep until next ohlc update when prediction is same as
            #    time.sleep(1)
            if self.ac.holding_side != '' and zero_three_exit_when_loss and (self.prediction[0] == 0 or self.prediction[0] == 3):
                pl = self.ac.holding_price - TickData.get_ltp() if self.ac.holding_side == 'sell' else TickData.get_ltp() - self.ac.holding_price
                if pl < -500:
                    if self.ac.order_side != '':
                        self.cancel_order()
                    self.entry_market_order('sell' if self.ac.holding_side == 'buy' else 'buy', self.ac.holding_size)
                    print('exit zero_three_exit_loss')
                    LineNotification.send_error('exit zero_three_exit_loss')
            if self.ac.order_side != '' and abs(self.ac.order_price - TickData.get_ltp()) <= 5000:
                res = self.ac.check_execution()
                if res !='':
                    LogMaster.add_log(res,self.prediction[0],self.ac)
            if Trade.flg_api_limit:
                time.sleep(60)
                print('Bot sleeping for 60sec due to API access limitation')
            else:
                time.sleep(0.1)


    def __bot_initializer(self, num_term, window_term, pl_kijun, future_period):
        Trade.cancel_all_orders()
        self.pl_kijun = pl_kijun
        print('bot - updating crypto data..')
        LogMaster.add_log('action_message - bot - updating crypto data..', 0,self.ac)
        CryptowatchDataGetter.get_and_add_to_csv()
        self.last_ohlc_min = datetime.now(self.JST).minute-1
        print('bot - initializing MarketData..')
        OneMinMarketData.initialize_for_bot(num_term, window_term, future_period, pl_kijun, num_term + 1)
        #self.model = CatModel()
        self.model = LgbModel()
        print('bot - generating training data')
        LogMaster.add_log('bot - training xgb model..', self.prediction[0], self.ac)
        print('bot - training model..')
        #self.cbm = self.model.read_dump_model('./Model/cat_model.dat')
        self.bst = self.model.load_model('./Model/lgb_model.dat')
        print('bot - load cat model completed..')
        print('bot - started bot loop.')
        LogMaster.add_log('action_message - bot - started bot loop.', self.prediction[0], self.ac)

    def __check_system_maintenance(self):
        if (datetime.now(tz=self.JST).hour == 3 and datetime.now(tz=self.JST).minute >= 59):
            print('sleep waiting for system maintenance')
            LineNotification.send_error('sleep waiting for system maintenance')
            if self.ac.order_side != '':
                self.cancel_order()
            time.sleep(660)  # wait for daily system maintenace
            LineNotification.send_error('resumed from maintenance time sleep')
            print('resumed from maintenance time sleep')


    def __update_ohlc(self): #should download ohlc soon after when it finalized
        if self.last_ohlc_min < datetime.now(self.JST).minute or (self.last_ohlc_min == 59 and datetime.now(self.JST).minute == 0):
            flg = True
            num_conti = 0
            num_max_trial = 10
            time.sleep(1)
            omd = TickData.get_ohlc()
            if omd is not None:
                for i in range(len(omd.unix_time)):
                    if OneMinMarketData.ohlc.unix_time[-1] < omd.unix_time[i]:
                        OneMinMarketData.ohlc.add_and_pop(omd.unix_time[i], omd.dt[i], omd.open[i], omd.high[i],omd.low[i], omd.close[i], omd.size[i])
                        flg = False
                LogMaster.add_log('updated ws ohlc at ' + str(datetime.now(self.JST)), self.prediction[0], self.ac)
            else:
                while flg:
                    res, omd = CryptowatchDataGetter.get_data_after_specific_ut(OneMinMarketData.ohlc.unix_time[-1])
                    if res == 0 and omd is not None:
                        if len(omd.unix_time) > 0:
                            omd.cut_data(len(omd.unix_time)) #remove first data to exclude after ut
                            for i in range(len(omd.unix_time)):
                                if OneMinMarketData.ohlc.unix_time[-1] < omd.unix_time[i]:
                                    OneMinMarketData.ohlc.add_and_pop(omd.unix_time[i], omd.dt[i], omd.open[i], omd.high[i], omd.low[i],omd.close[i], omd.size[i])
                                    flg = False
                            if flg == False:
                                print('data download dt='+str(datetime.now(self.JST)))
                                LogMaster.add_log('updated ohlc at '+str(datetime.now(self.JST)), self.prediction[0], self.ac)
                    num_conti += 1
                    time.sleep(1)
                    if num_conti >= num_max_trial:
                        print('data can not be downloaded from cryptowatch.')
                        LogMaster.add_log('ohlc download error', self.prediction[0], self.ac)
                        break
            if flg == False:
                self.last_ohlc_min = datetime.now(self.JST).minute
                OneMinMarketData.update_for_bot()
                df = OneMinMarketData.generate_df_for_bot()
                pred_x = self.model.generate_bot_pred_data(df)
                #self.prediction = self.cbm.predict(Pool(pred_x))
                self.prediction = self.model.prediction2(self.bst, pred_x)
                if len(self.prediction) > 1:
                    print('prediction length error!')
                print(self.prediction)
                self.pred_side = str(int(self.prediction[0])).translate(str.maketrans({'0': 'no', '1': 'buy', '2': 'sell', '3': 'both'}))
            self.ac.sync_position_order()
            self.ac.calc_pl(TickData.get_ltp())
            print('dt={}, open={},high={},low={},close={}'.format(OneMinMarketData.ohlc.dt[-1],
                                                                  OneMinMarketData.ohlc.open[-1],
                                                                  OneMinMarketData.ohlc.high[-1],
                                                                  OneMinMarketData.ohlc.low[-1],
                                                                  OneMinMarketData.ohlc.close[-1]))
            print('total_pl={}, pl per min={}, collateral change={},num_trade={},win_rate={},prediction={},holding_side={},holding_price={},holding_size={},ltp={}'.
                  format(self.ac.total_pl,self.ac.total_pl_per_min,self.ac.collateral_change,self.ac.num_trade,self.ac.win_rate,self.prediction[0],self.ac.holding_side,self.ac.holding_price,self.ac.holding_size, TickData.get_ltp()))
            print('private access per 300sec={}'.format(Trade.total_access_per_300s))
            LogMaster.add_log('private access per 300sec = '+str(Trade.total_access_per_300s), self.prediction[0], self.ac)
            LineNotification.send_notification(LogMaster.get_latest_performance())


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
    TickData.initialize(30,30)
    Trade.initialize()
    LogMaster.initialize()
    LineNotification.initialize()
    fb = FlyerBot()
    fb.start_flyer_bot(500,10,100000,15, False) #num_term, window_term, pl_kijun, future_period,
    #'JRF20190526-142616-930215'
    #JRF20190526-143431-187560