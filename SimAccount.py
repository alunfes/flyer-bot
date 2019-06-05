class SimAccount:
    def __init__(self):
        self.__initialize_order()
        self.__initialize_holding()

        self.base_margin_rate = 1.2
        self.leverage = 4.0
        self.slip_page = 50
        self.force_loss_cut_rate = 0.5
        self.initial_asset = 15000
        self.order_cancel_delay = 1

        self.total_pl = 0
        self.realized_pl = 0
        self.current_pl = 0
        self.num_trade = 0
        self.num_win = 0
        self.win_rate = 0
        self.asset = self.initial_asset

        self.dt_log = []
        self.i_log = []
        self.order_log = []
        self.holding_log = []
        self.total_pl_log = []
        self.action_log = []
        self.performance_total_pl_log = []

        self.start_dt = ''
        self.end_dt = ''


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

    def move_to_next(self,  i, dt, ut, tick_price):
        if self.start_dt == '':
            self.start_dt = dt
        self.__check_loss_cut(i)
        self.__check_execution(i)
        self.__check_cancel(i)
        if self.holding_side != '':
            self.current_pl = (tick_price - self.holding_price) * self.holding_size if self.holding_side == 'buy' else (self.holding_price - tick_price) * self.holding_size
        else:
            self.current_pl = 0
        self.total_pl = self.realized_pl + self.current_pl
        self.performance_total_pl_log.append(self.total_pl)
        self.asset = self.initial_asset + self.total_pl
        #self.__add_log('i:'+str(i), i)

    def last_day_operation(self,i, dt, ut, tick_price):
        self.__check_loss_cut(i)
        self.__check_execution(i)
        self.__check_cancel(i)
        if self.holding_side != '':
            self.realized_pl += (tick_price - self.holding_price) * self.holding_size if self.holding_side == 'buy' else (self.holding_price -tick_price) * self.holding_size
        self.total_pl = self.realized_pl + self.current_pl
        self.num_trade += 1
        self.total_pl_log.append(self.total_pl)
        self.performance_total_pl_log.append(self.total_pl)
        if self.num_trade > 0:
            self.win_rate = self.num_win / self.num_trade
        self.__add_log('Sim Finished.',i,dt,ut,tick_price)
        self.end_dt = dt
        print('from dt={}, : to_dt={}, total p={}, num trade={}, win rate={}'.format(self.start_dt,self.end_dt,self.total_pl,self.num_trade,self.win_rate))

    def entry_order(self, side, price, size, type, expire, i, dt, ut, tick_price):
        if self.order_side == '':
            self.order_side = side
            self.order_price = price
            self.order_size = size
            self.order_i = i
            self.order_dt = dt
            self.order_ut = ut
            self.order_type = type  # limit, market
            self.order_cancel = False
            self.order_expire = expire
            self.__add_log('entry order'+side+' type='+type, i,dt,ut,tick_price)
        else:
            print('order is already exist!')
            self.__add_log('order is already exist!', i,dt,ut,tick_price)


    def __update_holding(self, side, price, size, i, dt, ut):
        self.holding_side = side
        self.holding_price = price
        self.holding_size = size
        self.holding_i = i
        self.holding_dt = dt
        self.holding_ut = ut

    def cancel_order(self,  i, dt, ut):
        if self.order_type != 'losscut' and self.order_cancel == False:
            self.order_cancel = True
            self.order_i = i
            self.order_dt = dt
            self.order_ut = ut


    def __check_cancel(self,i, dt, ut, tick_price):
        if self.order_cancel:
            if ut - self.order_ut >= self.order_cancel_delay:
                self.__initialize_order()
                self.__add_log('order cancelled.', i, dt, ut, tick_price)

    def __check_expiration(self,i, dt, ut, tick_price):
        if self.ticks.ut[i] - self.order_ut >= self.order_expire and self.order_type != 'market' and self.order_type != 'losscut':
            self.__initialize_order()
            self.__add_log('order expired.', i, dt, ut, tick_price)


    def __check_execution(self, i, dt, ut, tick_price):
        if ut - self.order_ut >= self.order_cancel_delay and self.order_side != '':
            if self.order_type == 'market' or self.order_type == 'losscut':
                self.__process_execution(tick_price,i, dt, ut, tick_price)
                self.__initialize_order()
            elif self.order_type == 'limit' and ((self.order_side == 'buy' and self.order_price >= tick_price) or (self.order_side == 'sell' and self.order_price <= tick_price)):
                self.__process_execution(self.order_price,i, dt, ut, tick_price)
                self.__initialize_order()
            elif self.order_type != 'market' and self.order_type != 'limit' and self.order_type != 'losscut':
                print('Invalid order type!' + self.order_type)
                self.__add_log('invalid order type!'+self.order_type, i,dt, ut, tick_price)

    def __process_execution(self, exec_price, i, dt, ut, tick_price):
        if self.order_side != '':
            if self.holding_side == '':  # no position
                self.__update_holding(self.order_side, exec_price, self.order_size, i, dt, ut)
                self.__add_log('New Entry:'+self.order_type, i,dt, ut, tick_price)
            else:
                if self.holding_side == self.order_side:  # order side and position side is matched
                    price = round(((self.holding_price * self.holding_size) + (exec_price * self.order_size)) / (self.order_size + self.holding_size))
                    self.__update_holding(self.holding_side, price, self.order_size + self.holding_size, i, dt, ut)
                    self.__add_log('Additional Entry:' + self.order_type, i, dt, ut, tick_price)
                elif self.holding_size > self.order_size:  # side is not matched and holding size > order size
                    self.__calc_executed_pl(exec_price, i)
                    self.__update_holding(self.holding_side,self.holding_price,self.holding_size - self.order_size,i, dt, ut)
                    self.__add_log('Exit Order (h>o):' + self.order_type, i, dt, ut, tick_price)
                elif self.holding_size == self.order_size:
                    self.__add_log('Exit Order (h=o):' + self.order_type, i, dt, ut, tick_price)
                    self.__calc_executed_pl(exec_price, i)
                    self.__initialize_holding()
                else:  # in case order size is bigger than holding size
                    self.__calc_executed_pl(exec_price, i)
                    self.__add_log('Exit & Entry Order::' + self.order_type, i, dt, ut, tick_price)
                    self.__update_holding(self.order_side, exec_price, self.order_size - self.holding_size, i, dt, ut)


    def __calc_executed_pl(self,exec_price,i): #assume all order size was executed
        pl = (exec_price - self.holding_price) * self.order_size if self.holding_side == 'buy' else (self.holding_price - exec_price) * self.order_size
        self.realized_pl += round(pl)
        self.num_trade += 1
        if pl >0:
            self.num_win +=1

    def __check_loss_cut(self,  i, dt, ut, tick_price):
        if self.holding_side != '' and self.order_type !='losscut':
            req_collateral = self.holding_size * tick_price / self.leverage
            pl = tick_price - self.holding_price if self.holding_side == 'buy' else self.holding_price - tick_price
            pl = pl * self.holding_size
            margin_rate = (self.initial_asset + self.realized_pl + pl) / req_collateral
            if margin_rate <= self.force_loss_cut_rate:
                self.__force_exit(i, dt, ut)
                self.__add_log('Loss cut postion! margin_rate=' + str(margin_rate), i, dt, ut, tick_price)

    def __force_exit(self,  i, dt, ut):
        self.order_side = 'buy' if self.holding_side == 'sell' else 'sell'
        self.order_size = self.holding_size
        self.order_type = 'losscut'
        self.order_i = i
        self.order_dt = dt
        self.order_ut = ut
        self.order_cancel = False
        self.order_expire = 86400

    def __add_log(self, log, i, dt, ut, tick_price):
        self.total_pl_log.append(self.total_pl)
        self.action_log.append(log)
        self.holding_log.append(self.holding_side + ' @' + str(self.holding_price) + ' x' + str(self.holding_size))
        self.order_log.append(self.order_side + ' @' + str(self.order_price) + ' x' + str(self.order_size) + ' cancel=' + str(self.order_cancel) + ' type=' + self.order_type)
        self.i_log.append(i)
        self.dt_log.append(dt)
        print('i={},dt={},action={},holding side={}, holding price={},holding size={},order side={},order price={},order size={},pl={},num_trade={}'.
              format(i, tick_price,log,self.holding_side,self.holding_price,self.holding_size,self.order_side,self.order_price,self.order_size,self.total_pl,self.num_trade))