import websocket
import threading
import time
import json
import asyncio
import ssl
from statistics import mean, median,variance,stdev
from datetime import datetime
import pytz

class WebsocketMaster:
    def __init__(self, channel, symbol=''):
        self.symbol = symbol
        self.ticker = None
        self.message = None
        self.exection = None
        self.channel = channel
        self.connect()
        self.time_start = time.time()

    def connect(self):
        self.ws = websocket.WebSocketApp(
            'wss://ws.lightstream.bitflyer.com/json-rpc', header=None,
            on_open = self.on_open, on_message = self.on_message,
            on_error = self.on_error, on_close = self.on_close)
        self.ws.keep_running = True
        websocket.enableTrace(False)
        #self.thread = threading.Thread(target=lambda: self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}))
        self.thread = threading.Thread(target=lambda: self.ws.run_forever())
        self.thread.daemon = True
        self.thread.start()

    def is_connected(self):
        return self.ws.sock and self.ws.sock.connected

    def disconnect(self):
        print('disconnected')
        self.ws.keep_running = False
        self.ws.close()


    def on_message(self, ws, message):
        message = json.loads(message)['params']
        self.message = message['message']
        if self.channel == 'lightning_executions_':
            if self.message is not None:
                self.exection = self.message
                TickData.add_exec_data(self.exection)
                pass
        elif self.channel == 'lightning_ticker_':
            if self.message is not None:
                self.ticker = self.message
                TickData.add_ticker_data(self.ticker)
                pass

    def on_error(self, ws, error):
        print('websocket error!')
        try:
            if self.is_connected():
                self.disconnect()
        except Exception as e:
            print('websocket - '+str(e))
        time.sleep(3)
        self.connect()

    def on_close(self, ws):
        print('Websocket disconnected')


    def on_open(self, ws):
        ws.send(json.dumps( {'method':'subscribe',
            'params':{'channel':self.channel + self.symbol}} ))
        time.sleep(1)
        print('Websocket connected for '+self.channel)


    async def loop(self):
        while True:
            await asyncio.sleep(1)



'''
TickData class
'''
class TickData:
    @classmethod
    def initialize(cls):
        cls.lock = threading.Lock()
        cls.ltp = 0
        cls.ws_execution = WebsocketMaster('lightning_executions_', 'FX_BTC_JPY')
        cls.ws_ticker = WebsocketMaster('lightning_ticker_', 'FX_BTC_JPY')
        cls.exec_data = []
        cls.ticker_data = []
        cls.std_1m = 0
        cls.ohlc = []
        cls.JST = pytz.timezone('Asia/Tokyo')
        th = threading.Thread(target=cls.start_thread)
        th.start()

    @classmethod
    def get_ltp(cls):
        with cls.lock:
            if len(cls.exec_data) > 0:
                return cls.exec_data[-1]['price']
            else:
                return None

    @classmethod
    def get_1m_std(cls):
        return cls.std_1m

    @classmethod
    def __calc_std(cls, ltps):
        if len(ltps) > 60:
            cls.std_1m = stdev(ltps[-60:])

    @classmethod
    def get_bid_price(cls):
        with cls.lock:
            if len(cls.ticker_data) > 0:
                return cls.ticker_data[-1]['best_bid']
            else:
                return None

    @classmethod
    def get_ask_price(cls):
        with cls.lock:
            if len(cls.ticker_data) > 0:
                return cls.ticker_data[-1]['best_ask']
            else:
                return None


    @classmethod
    def start_thread(cls):
        ltps = []
        while True:
            #ltps.expand(list([d.get('ltp') for d in cls.ticker_data]))
            cls.__check_thread_status()
            cls.__calc_std(list([d.get('ltp') for d in cls.ticker_data]))
            time.sleep(1)

    @classmethod
    def __calc_ohlc(cls):
        if datetime.now(cls.JST).second <2 and :
            if len(cls.ohlc) ==0 and len(cls.ticker_data) > 120:



    @classmethod
    def __check_thread_status(cls):
        if cls.ws_execution.is_connected == False:
            cls.ws_execution.connect()
        if cls.ws_ticker.is_connected == False:
            cls.ws_ticker.connect()

    @classmethod
    def add_exec_data(cls, exec):
        with cls.lock:
            if len(exec) > 0:
                cls.exec_data.extend(exec)
                if len(cls.exec_data) >= 30000:
                    del cls.exec_data[:-10000]


    @classmethod
    def add_ticker_data(cls, ticker):
        with cls.lock:
            if len(ticker) is not None:
                cls.ticker_data.append(ticker)
                if len(cls.ticker_data) >= 30000:
                    del cls.ticker_data[:-10000]
            else:
                print(ticker)

class OHLC:
    def __init__(self):
        self.open = 0
        self.high = 0
        self.low = 0
        self.close = 0
        self.dt = ''

if __name__ == '__main__':
    TickData.initialize()
    while True:
        time.sleep(1)
        print(len(TickData.ticker_data))
        #print(str(TickData.get_ltp()))
        #print(str(TickData.get_bid_price()))
        #print(str(TickData.get_ask_price()))
        #print('std 1m='+str(TickData.get_1m_std()))
        #time.sleep(1)