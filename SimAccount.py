class SimAccount:
    def __init__(self, ticks):
        self.ticks = ticks
        self.__initialize_order()
        self.__initialize_holding()

        self.base_margin_rate = 1.2
        self.leverage = 15.0
        self.slip_page = 500
        self.force_loss_cut_rate = 0.5
        self.initial_asset = 5000
        self.order_cancel_delay = 3
        self.prediction_delay = 3

        self.total_pl = 0
        self.realized_pl = 0
        self.current_pl = 0
        self.num_trade = 0
        self.num_win = 0
        self.win_rate = 0
        self.asset = self.initial_asset

        self.dt_log = []
        self.i_log = {}
        self.order_log = {}
        self.holding_log = {}
        self.total_pl_log = {}
        self.action_log = {}


    def __initialize_order(self):
        self.order_side = ''
        self.order_price = 0
        self.order_size = 0
        self.order_i = 0
        self.order_dt = ''
        self.order_ut = 0
        self.order_type = ''
        self.order_cancel = False
        self.order_expire = 0

    def __initialize_holding(self):
        self.holding_side = ''
        self.holding_price = 0
        self.holding_size = 0
        self.holding_i = 0
        self.holding_dt = ''
        self.holding_ut = 0

    def move_to_next(self,  i):
        self.__check_loss_cut(i)
        self.__check_execution(i)
        self.__check_cancel(i)
        if self.order_side != '':
            self.current_pl = (self.ticks.price[i] - self.holding_price) * self.holding_size if self.holding_side == 'buy' else (self.holding_price - self.ticks.price[i]) * self.holding_size
        else:
            self.current_pl = 0
        self.total_pl = self.realized_pl + self.current_pl
        self.total_pl_log.append(self.total_pl)
        self.asset = self.initial_asset + self.total_pl
        self.__add_log('i:'+str(i), i)

    def last_day_operation(self,i):
        self.__check_loss_cut(i)
        self.__check_execution(i)
        self.__check_cancel(i)
        if self.holding_side != '':
            if self.order_side != '':
                self.realized_pl += (self.ticks.price[i] - self.holding_price) * self.holding_size if self.holding_side == 'buy' else (self.holding_price -self.ticks.price[i]) * self.holding_size
        self.total_pl = self.realized_pl + self.current_pl
        self.total_pl_log.append(self.total_pl)
        if self.num_trade > 0:
            self.win_rate = self.num_win / self.num_trade

    def entry_order(self, side, price, size, type, expire, i):
        if self.order_side == '':
            self.order_side = side
            self.order_price = price
            self.order_size = size
            self.order_i = i
            self.order_dt = self.ticks.dt[i]
            self.order_ut = self.ticks.ut[i]
            self.order_type = type  # limit, market
            self.order_cancel = False
            self.order_expire = expire
        else:
            print('order is already exist!')


    def __update_holding(self, side, price, size, i):
        self.holding_side = side
        self.holding_price = price
        self.holding_size = size
        self.holding_i = i
        self.holding_dt = self.ticks.dt[i]
        self.holding_ut = self.ticks.ut[i]

    def cancel_order(self,  i):
        if self.order_type != 'losscut':
            self.order_cancel = True
            self.order_i = i
            self.order_dt = self.ticks.dt[i]
            self.order_ut = self.ticks.ut[i]

    def __check_cancel(self,i):
        if self.order_cancel:
            if self.ticks.ut[i] - self.order_ut >= self.order_cancel_delay:
                self.__add_log('order cancelled.',i)
                self.__initialize_order()

    def __check_expiration(self,i):
        if self.ticks.ut[i] - self.order_ut >= self.order_expire and self.order_type != 'market' and self.order_type != 'losscut':
            self.__add_log('order expired.', i)
            self.__initialize_order()


    def __check_execution(self, i):
        if self.ticks.ut[i] - self.order_ut >= self.order_cancel_delay and self.order_side != '':
            if self.order_type == 'market' or self.order_type == 'losscut':
                self.__process_execution(self.ticks.price[i],i)
                self.__initialize_order()
            elif self.order_type == 'limit' and ((self.order_side == 'buy' and self.order_price >= self.ticks.price[i]) or (self.order_side == 'sell' and self.order_price <= self.ticks.price[i])):
                self.__process_execution(self.order_price,i)
                self.__initialize_order()
            elif self.order_type != 'market' and self.order_type != 'limit' and self.order_type != 'losscut':
                print('Invalid order type!' + self.order_type)

    def __process_execution(self, exec_price, i):
        if self.order_side != '':
            if self.holding_side == '':  # no position
                self.__update_holding(self.order_side, exec_price, self.order_size, i)
                self.__add_log('New Entry:'+self.order_type, i)
            else:
                if self.holding_side == self.order_side:  # order side and position side is matched
                    price = round(((self.holding_price * self.holding_size) + (exec_price * self.order_size)) / (self.order_size + self.holding_size))
                    self.__update_holding(self.holding_side, price, self.order_size + self.holding_size, i)
                    self.__add_log('Additional Entry:' + self.order_type, i)
                elif self.holding_size > self.order_size:  # side is not matched and holding size > order size
                    self.__calc_executed_pl(exec_price, i)
                    self.__update_holding(self.holding_side,self.holding_price,self.holding_size - self.order_size,i)
                    self.__add_log('Exit Order (h>o):' + self.order_type, i)
                elif self.holding_size == self.order_size:
                    self.__add_log('Exit Order (h=o):' + self.order_type, i)
                    self.__calc_executed_pl(exec_price, i)
                    self.__initialize_holding()
                else:  # in case order size is bigger than holding size
                    self.__calc_executed_pl(exec_price, i)
                    self.__add_log('Exit & Entry Order::' + self.order_type, i)
                    self.__update_holding(self.order_side, exec_price, self.order_size - self.holding_size, i)


    def __calc_executed_pl(self,exec_price,i): #assume all order size was executed
        pl = (exec_price - self.holding_price) * self.order_size if self.holding_side == 'buy' else (self.holding_price - exec_price) * self.order_size
        self.realized_pl += round(pl)
        self.num_trade += 1
        if pl >0:
            self.num_win +=1

    def __check_loss_cut(self,  i):
        if self.holding_side != '' and self.order_type !='losscut':
            req_collateral = self.holding_size * self.ticks.price[i] / self.leverage
            pl = self.ticks.price[i] - self.holding_price if self.holding_side == 'buy' else self.holding_price - self.ticks.price[i]
            pl = pl * self.holding_size
            margin_rate = (self.initial_asset + self.realized_pl + pl) / req_collateral
            if margin_rate <= self.force_loss_cut_rate:
                self.__add_log('Loss cut postion! margin_rate=' + str(margin_rate), i)
                self.__force_exit(i)

    def __force_exit(self,  i):
        self.order_side = 'buy' if self.holding_side == 'sell' else 'sell'
        self.order_size = self.holding_size
        self.order_type = 'losscut'
        self.order_i = i
        self.order_dt = self.ticks.dt[i]
        self.order_ut = self.ticks.ut[i]
        self.order_cancel = False
        self.order_expire = 86400

    def __add_log(self, log, i):
        dt =self.ticks.dt[i]
        self.total_pl_log[dt] = self.total_pl
        self.action_log[dt] = log
        self.holding_log[dt] = self.holding_side+' @'+str(self.holding_price)+' x'+str(self.holding_size)
        self.order_log[dt] = self.order_side+' type:'+self.order_type+' cancel:'+self.order_cancel+' @'+str(self.order_price)+' x'+str(self.order_size)
        self.i_log[dt] = i
        self.dt_log.append(dt)
