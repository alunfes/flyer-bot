from Trade import Trade
from WebsocketMaster import TickData

class BotAccount:
    def __init__(self):
        self.initialize_order()
        self.initialize_holding()

        self.initial_collateral = Trade.get_collateral()['collateral']
        self.collateral_change = 0
        self.collateral_change_per_min = 0
        self.collateral_change_log = []

        self.num_trade = 0
        self.num_win = 0
        self.win_rate = 0
        self.pl_per_min = 0
        self.elapsed_time = 0

        self.num_trade = 0
        self.num_win = 0
        self.win_rate = 0

        self.sync_position_order()


    def initialize_order(self):
        self.order_side = ''
        self.order_price = 0
        self.order_executed_size = 0
        self.order_outstanding_size = 0
        self.order_id = 0
        self.order_expire = 0
        self.order_status = '' #new entry, pl order

    def initialize_holding(self):
        self.holding_side = ''
        self.holding_price = 0
        self.holding_size = 0
        self.holding_id = 0
        self.holding_dt = ''

    def combine_status_data(self, status):
        side = ''
        size = 0
        price = 0
        for s in status:
            side = s['side'].lower()
            size += float(s['size'])
            price += float(s['price']) * float(s['size'])
        price = round(price / size)
        return side, round(size,8), round(price)

    def sync_position_order(self):
        position = Trade.get_positions()
        orders = Trade.get_orders()
        if len(position) > 0:
            holding_side, holding_size, holding_price = self.combine_status_data(position)
            if self.holding_side != holding_side or abs(self.holding_price - holding_price) >= 1 or abs(self.holding_size - holding_size) >= 0.01:
                self.holding_side, self.holding_size, self.holding_price = holding_side, holding_size, holding_price
                print('position unmatch was detected! Synchronize with account position data.')
                print('holding_side={},holding_price={},holding_size={}'.format(self.holding_side,self.holding_price,self.holding_size))
                print(position)
            print('synchronized position data, side='+str(self.holding_side)+', size='+str(self.holding_size)+', price='+str(self.holding_price))
        else:
            self.initialize_holding()
        if len(orders) > 0:#need to update order status
            if len(orders) > 1:
                print('multiple orders are found! Only the first one will be synchronized!')
            try:
                if orders[0]['info']['child_order_state'] == 'ACTIVE':
                    self.order_id = orders[0]['info']['child_order_acceptance_id']
                    self.order_side = orders[0]['info']['side'].lower()
                    self.order_outstanding_size = float(orders[0]['info']['outstanding_size'])
                    self.order_executed_size = float(orders[0]['info']['executed_size'])
                    self.order_price = round(float(orders[0]['info']['price']))
                    self.order_status = 'new entry' if self.holding_side=='' else 'pl order'
                    print('synchronized order data, side='+str(self.order_side)+', outstanding size='+str(self.order_outstanding_size)+', price='+str(self.order_price))
            except Exception as e:
                print('Bot-sync_position_order:sync order key error!' + str(e))
        else:
            self.initialize_order()


    def calc_collateral_change(self):
        col = Trade.get_collateral()
        self.collateral_change = round(float(col['collateral']) + float(col['open_position_pnl']) - self.initial_collateral)
        if self.elapsed_time > 0:
            self.collateral_change_per_min = round(self.collateral_change / (self.elapsed_time/60.0), 4)
        else:
            pass

    def update_holding(self, side, price, size, id):
        self.holding_side = side
        self.holding_price = price
        self.holding_size = size
        self.holding_id = id

    def update_order(self, side, price, exec_size, outstanding_size, id, expire, status):
        self.order_side = side
        self.order_price = price
        self.order_executed_size = exec_size
        self.order_id = id
        self.order_expire = expire
        self.order_outstanding_size = outstanding_size
        self.order_status = status

    def check_execution(self):
        if self.order_side !='':
            status = Trade.get_order_status(self.order_id) #{'id': 0, 'child_order_id': 'JFX20190218-133228-026751F', 'product_code': 'FX_BTC_JPY', 'side': 'BUY', 'child_order_type': 'LIMIT', 'price': 300000.0, 'average_price': 0.0, 'size': 0.01, 'child_order_state': 'ACTIVE', 'expire_date': '2019-03-20T13:32:16', 'child_order_date': '2019-02-18T13:32:16', 'child_order_acceptance_id': 'JRF20190218-133216-339861', 'outstanding_size': 0.01, 'cancel_size': 0.0, 'executed_size': 0.0, 'total_commission': 0.0}
            if status is not None:
                order_type = 'new entry' if self.order_status == 'new entrying' else 'pl'
                if len(status) > 0: #check if executed (fully / partially)
                    if status[0]['child_order_state'] =='COMPLETED':
                        print(order_type+' order has been executed.')
                        self.update_holding(status[0]['side'].lower(),status[0]['average_price'],status[0]['executed_size'],status[0]['child_order_acceptance_id'])
                        self.initialize_order()
                        self.calc_collateral_change()
                        return order_type+' order has been executed.'
                    elif status[0]['executed_size'] > self.order_executed_size:
                        print(order_type + ' order has been partially executed.')
                        self.order_executed_size = status[0]['executed_size']
                        self.order_outstanding_size = status[0]['outstanding_size']
                        return order_type+' order has been partially executed.'
                else:
                    print('order has been expired. '+status[0])
                    self.initialize_order()
                    return 'order has been expired'
            else:
                return 'get_order_status is temporary exhibited due to API access limitation!'
