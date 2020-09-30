import os
import pandas as pd
import numpy as np
from environment_Demonstration import Mota

class Feature_Data(object):
    def __init__(self, mode, env_name, route_mode):
        self.mode = mode
        self.env_name = env_name
        self.route_mode = route_mode
        self.env = Mota()
        # self.show_name = ['mapData_1','mapData_3','24層魔塔']
        #self.Find_Env_List()
        # self.Output_Excel()

    def Find_Env_List(self):
        self.map_list = []
        if self.route_mode == '單一地圖':
            max_hp = 0
            save_file_name = None
            for file_name in os.listdir('player_route/'):
                map_name = file_name.split('_v')[0]
                if self.env_name == map_name:
                    map_name_hp = int(file_name.split('_h')[1].split('.')[0])
                    if map_name_hp > max_hp:
                        save_file_name = file_name
            if save_file_name != None:
                self.map_list.append(save_file_name)

        elif self.route_mode == '強化地圖':
            for file_name in os.listdir('player_route/'):
                map_name = file_name.split('_v')[0]
                if self.env_name == map_name:
                    self.map_list.append(file_name)

        elif self.route_mode == '強化泛化':
            for file_name in os.listdir('player_route/'):
                self.map_list.append(file_name)

        print(self.map_list)

        if self.map_list:
            return True
        else:
            return False

    def Output_Excel(self):

        self.all_choose_index_list = []
        for load_map_name in self.map_list:
            self.all_choose_index_list.append(np.load('player_route/' + load_map_name))

        self.env.build_env(self.env_name)
        self.env.create_nodes()

        # 提取特徵
        df_dict = {'p_hp':[],'p_atk':[],'p_def':[],'p_money':[],'p_exp':[],
                   'p_yellowKey':[],'p_blueKey':[],'p_redKey':[],
                   'class':[],'hp':[],'atk':[],'def':[],'money':[],'exp':[],
                   'yellowKey':[],'blueKey':[],'redKey':[],'special':[],'choose':[]}

        for rate_, choose_index_list in enumerate(self.all_choose_index_list):
            if  rate_ > 0 and self.map_list[rate_].split('_v')[0] != self.map_list[rate_ - 1].split('_v')[0] or self.map_list[rate_].split('_v')[0] != self.env_name and rate_ == 0:
                del self.env
                self.env = Mota()
                self.env.build_env(self.map_list[rate_].split('_v')[0])
                self.env.create_nodes()
                print('Rebuild env')

            for rate, index in enumerate(choose_index_list, start=1):
                feasible_actions = self.env.get_feasible_actions()
                true_action = feasible_actions[index]
                actions = self.env.get_actions()
                # 狀態字典
                states = {}
                b = self.env.get_player_state()
                for i in range(len(actions)):
                    # 紀錄正確行動的索引值
                    if true_action == actions[i]:
                        true_index = i
                    self.env.step(actions[i])
                    a = self.env.get_player_state()
                    states[i] = a - b
                    self.env.back_step(1)
                # 添加數據
                for n in range(len(actions)):
                    df_dict['p_hp'].append(self.env.player.hp)
                    df_dict['p_atk'].append(self.env.player.atk)
                    df_dict['p_def'].append(self.env.player.def_)
                    df_dict['p_money'].append(self.env.player.money)
                    df_dict['p_exp'].append(self.env.player.exp)
                    df_dict['p_yellowKey'].append(self.env.player.items['yellowKey'])
                    df_dict['p_blueKey'].append(self.env.player.items['blueKey'])
                    df_dict['p_redKey'].append(self.env.player.items['redKey'])
                    df_dict['class'].append(actions[n].class_)
                    df_dict['hp'].append(states[n][0])
                    df_dict['atk'].append(states[n][1])
                    df_dict['def'].append(states[n][2])
                    df_dict['money'].append(states[n][3])
                    df_dict['exp'].append(states[n][4])
                    df_dict['yellowKey'].append(states[n][5])
                    df_dict['blueKey'].append(states[n][6])
                    df_dict['redKey'].append(states[n][7])
                    if actions[n].class_ == 'enemys':
                        df_dict['special'].append(actions[n].special)
                    else:
                        df_dict['special'].append(0)

                    if n == true_index:
                        df_dict['choose'].append(1)
                    else:
                        df_dict['choose'].append(0)
                # 前進
                self.env.step(feasible_actions[index])
                # 回傳進度到show_learniong
                yield int(((rate + rate_ + 1)/(len(choose_index_list) + len(self.all_choose_index_list)))*100)
            self.env.reset()
            print('Env Reset')
        # 輸出檔案
        df = pd.DataFrame(df_dict)

        if self.mode =='train':
            self.file_name = 'model/train.xls'
        else:
            self.file_name = 'model/test.xls'

        df.to_excel(self.file_name, index=False)
        print(df)

if __name__ == '__main__':
    # train = Feature_Data('test','mapData_3')
    # train = Feature_Data('text', 'mapData_3', '強化地圖')
    # for rate in train.Output_Excel():
    #     print(rate)

    train = Feature_Data('test', 'mapData_1', '強化地圖')
    a = train.Find_Env_List()
    for rate in train.Output_Excel():
        pass

    # train = Feature_Data('test', 'mapData_1', '強化泛化')
    # a = train.Find_Env_List()
    # for rate in train.Output_Excel():
    #     print(rate)
