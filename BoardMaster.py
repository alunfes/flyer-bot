import threading
import time
import json
import websocket
from sortedcontainers import SortedDict

def update_board(board, d):
   for i in d:
       p, s = int(i['price']), i['size']
       if s != 0:
           board[p] = s
       elif p in board:
           del board[p]

class BitflyerBoard:
   def __init__(self):
       self.ws = None
       self.lock = threading.Lock()
       self.bids = SortedDict()
       self.asks = SortedDict()
       self.run()

   def run(self):
       def on_open(ws):
           def subscribe(ch):
               ws.send(json.dumps(
                   {"method": "subscribe", "params": {"channel": ch}}))
           subscribe('lightning_board_snapshot_FX_BTC_JPY')
           subscribe('lightning_board_FX_BTC_JPY')

       def on_message(ws, msg):
           data = json.loads(msg)
           if data['method'] != 'channelMessage':
               return
           params = data['params']
           channel = params['channel']
           message = params['message']

           if channel == 'lightning_board_snapshot_FX_BTC_JPY':
               bids, asks = SortedDict(), SortedDict()
               update_board(bids, message['bids'])
               update_board(asks, message['asks'])
               with self.lock:
                   self.bids, self.asks = bids, asks
           elif channel == 'lightning_board_FX_BTC_JPY':
               with self.lock:
                   update_board(self.bids, message['bids'])
                   update_board(self.asks, message['asks'])

       self.ws = websocket.WebSocketApp(
           'wss://ws.lightstream.bitflyer.com/json-rpc',
           on_message=on_message,
           on_open=on_open)
       self.wst = threading.Thread(target=lambda: self.ws.run_forever())
       self.wst.daemon = True
       self.wst.start()

## ここから動作確認用コード ##
bt = BitflyerBoard()

try:
   while True:
       with bt.lock:
           for p, s in reversed(bt.asks.items()[:10]):
               print(p, '{:>10.3f}'.format(s))
           print('======== BOARD ========')
           for p, s in reversed(bt.bids.items()[-10:]):
               print(p, '{:>10.3f}'.format(s))
           print('')
       time.sleep(1)
except KeyboardInterrupt:
   pass
