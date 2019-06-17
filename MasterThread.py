import multiprocessing as mp
from OneMinMarketData import OneMinMarketData
from FlyerBot import FlyerBot
from LgbModel import LgbModel

class MasterThread:
    def start_master_thread(self, num_term, window_term, future_period, future_kijun, pl_kijun, train_period, zero_three_exit_loss):
        manager = mp.Manager()
        manager_dict = manager.dict()
        omd = OneMinMarketData()
        omd.initialize_for_bot(num_term, window_term, future_period, future_kijun, num_term + 1)
        manager_dict['omd'] = omd
        manager_dict['params'] = {'num_term':num_term, 'window_term':window_term, 'future_period':future_period, 'future_kijun':future_period,
                                  'pl_kijun':pl_kijun, 'train_period':train_period, 'zero_three_exit_loss':zero_three_exit_loss}

        mp.Process(target=FlyerBot., args=[data], daemon=True).start()
        mp.Process(target=calc_proc, args=[data, cmdq], daemon=True).start()
        mp.Process(target=deal_proc, args=[data, cmdq], daemon=True).start()
        while True: time.sleep(60)




if __name__ == '__main__':