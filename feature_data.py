# -*- coding: utf-8 -*-
"""
【模仿學習】                        by Hung1  2020/4/29
以人類專家演示一條通關路線，經過特徵工程後，篩選出有用的資料做成訓練集。
"""
import random
import pandas as pd
from environment import Mota

# 產生訓練集、測試集
env = Mota()
env_index = 2
network = 'policy'
#network = 'value'
file_name = f'model/{network}_training.xls'
#file_name = f'model/{network}_test.xls'

# 獲取人類專家的動作選擇
if env_index == 0:
    env.build_env('mapData_1')
    choose_index_list = [
    0, 0, 1, 2, 1, 0, 1, 1, 1, 2, 0, 1, 0, 0]
elif env_index == 1:
    env.build_env('mapData_3')
    choose_index_list = [
    0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 1, 0, 1, 0, 3, 1, 0]
elif env_index == 2:
    env.build_env('24層魔塔')
    choose_index_list = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
    0, 2, 0, 1, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 2,
    2, 0, 4, 0, 0, 0, 1, 5, 4, 0, 0, 0, 0, 2, 0, 0, 4, 0, 0, 0, 0, 1, 3, 0, 0,
    1, 1, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 2, 5, 1, 0, 0, 0, 9, 2, 8, 1, 1, 0, 0, 0, 0,
    0, 1, 2, 1, 0,17, 0,13, 0, 4, 0, 0,22, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1,25,
    0,11, 2, 2, 1, 1, 1, 0, 0,13,10, 0, 7, 0, 0, 0, 0, 0,11, 0, 0, 0, 0, 1, 4,
    1, 0,16, 6, 1, 1, 0, 9,19,19, 0, 0, 0, 0, 0, 0,20, 1, 1, 9, 9, 0,16, 1,16,
    5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 4, 7,18, 0,21,24, 2, 0,33,17,
    0, 1, 0, 0, 0, 0, 7, 0, 0, 6, 2, 0, 1, 5,11, 0, 0, 0, 0, 0,15, 2,28,24, 2,
    0, 0,14, 0, 0, 0, 1, 0, 0, 0,28, 0, 0, 0, 0, 5,40, 5,42, 1, 5,40,24, 3, 3,
    0, 0, 0, 0, 0, 0, 0,43, 0, 0, 0, 0, 0, 0, 0,38, 5, 0, 2, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 2,24, 0, 0,48,18, 0, 0, 0,41, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0,30,27, 0, 0, 0, 1, 1, 0,44, 2, 0, 1,26, 0,13, 0,35, 0, 0,13,23,40,
    6, 0, 1, 0,21, 0, 1, 0, 0, 0,11, 0,27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,43,17, 0, 0, 1,43, 0, 0, 0,43, 1, 9, 0,48, 0, 0, 7, 0, 1, 1, 6,
    0, 0,42, 0, 0, 3, 0, 0,11, 0,25, 0,43,40, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 1,44, 0, 0,38,42, 0, 0, 0, 2, 2, 0, 2, 2, 1, 1, 0, 0, 1,42, 0, 0,
   47, 0,42, 0, 0,46, 0, 0,19,19, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0,45, 2, 0, 0, 0, 2, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,53, 0, 0,57,33,25, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0,50, 0, 0,53, 0, 0,15,24, 0, 1, 0,47, 0, 0, 0,40, 0, 0, 0, 0,
    0,40, 0, 0,51,52, 0, 0, 0,51, 0, 6, 0, 0, 0, 0, 0,59, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 0,60, 0, 0, 0, 0, 0, 0, 0, 0, 0,54, 7,45,47, 2, 0, 0, 0, 8, 0, 0, 0,
   40, 0, 0, 0, 0,42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,56,
    2, 0,32,21, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0,55, 0,52,57, 0,56, 8,55, 0,
    0, 0, 0, 0, 0, 0, 0, 0,50, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2,
    0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
    1, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 2, 6,49, 71, 29, 40, 0, 0,
   71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0,66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,36, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,20,22, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 1, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4, 8, 0, 0, 1,
    0, 1, 0, 0, 2, 2, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 7, 0, 0, 1, 0, 0,
    1, 5, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3,19, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    4, 0, 6, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 6,16,78, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   78, 0, 0, 0,74, 0, 0, 0,72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,45, 0, 0,
    0, 0, 0,46, 0, 0, 1, 0, 8,28,41,40, 0, 0,37, 1,12,23,12,32,12, 0, 0, 0, 3,
   24, 0, 0, 0, 0, 0, 0, 0, 0, 0,23, 0, 0, 5,21,20, 0,20, 0, 2, 0, 0, 0, 0,12,
    9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 1, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0]
env.create_nodes()

# 策略網路
if network == 'policy':
    # 提取特徵
    df_dict = {'p_hp':[],'p_atk':[],'p_def':[],'p_money':[],'p_exp':[],
               'p_yellowKey':[],'p_blueKey':[],'p_redKey':[],
               'class':[],'hp':[],'atk':[],'def':[],'money':[],'exp':[],
               'yellowKey':[],'blueKey':[],'redKey':[],'special':[],
               'class2':[],'hp2':[],'atk2':[],'def2':[],'money2':[],'exp2':[],
               'yellowKey2':[],'blueKey2':[],'redKey2':[],'special2':[],'choose':[]}
    for index in choose_index_list:
        actions = env.get_feasible_actions()
        # 狀態字典
        states = {}
        b = env.get_player_state()
        for i in range(len(actions)):
            env.step(actions[i])
            a = env.get_player_state()
            states[i] = a - b
            env.back_step(1)
        # 添加數據
        for m in range(len(actions)):
            n = index
            if not (states[n] == states[m]).all() or actions[n].class_ != actions[m].class_:
                df_dict['p_hp'].append(env.player.hp)
                df_dict['p_atk'].append(env.player.atk)
                df_dict['p_def'].append(env.player.def_)
                df_dict['p_money'].append(env.player.money)
                df_dict['p_exp'].append(env.player.exp)
                df_dict['p_yellowKey'].append(env.player.items['yellowKey'])
                df_dict['p_blueKey'].append(env.player.items['blueKey'])
                df_dict['p_redKey'].append(env.player.items['redKey'])
                if random.randint(0, 1) == 1:
                    n, m = m, n
                    df_dict['choose'].append(1)
                else:
                    df_dict['choose'].append(0)
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
                df_dict['class2'].append(actions[m].class_)
                df_dict['hp2'].append(states[m][0])
                df_dict['atk2'].append(states[m][1])
                df_dict['def2'].append(states[m][2])
                df_dict['money2'].append(states[m][3])
                df_dict['exp2'].append(states[m][4])
                df_dict['yellowKey2'].append(states[m][5])
                df_dict['blueKey2'].append(states[m][6])
                df_dict['redKey2'].append(states[m][7])
                if actions[m].class_ == 'enemys':
                    df_dict['special2'].append(actions[m].special)
                else:
                    df_dict['special2'].append(0)
        # 前進
        env.step(actions[index])
    # 輸出檔案
    df = pd.DataFrame(df_dict)
    df.to_excel(file_name, index=False)
    print(df)

# 估值網路
elif network == 'value':
    # 提取特徵
    df_dict = {'p_hp':[],'p_atk':[],'p_def':[],'p_money':[],'p_exp':[],
               'p_yellowKey':[],'p_blueKey':[],'p_redKey':[],
               'class':[],'hp':[],'atk':[],'def':[],'money':[],'exp':[],
               'yellowKey':[],'blueKey':[],'redKey':[],'special':[],'choose':[]}
    for index in choose_index_list:
        feasible_actions = env.get_feasible_actions()
        true_action = feasible_actions[index]
        actions = env.get_actions()
        # 狀態字典
        states = {}
        b = env.get_player_state()
        for i in range(len(actions)):
            # 紀錄正確行動的索引值
            if true_action == actions[i]:
                true_index = i
            env.step(actions[i])
            a = env.get_player_state()
            states[i] = a - b
            env.back_step(1)
        # 添加數據
        for n in range(len(actions)):
            df_dict['p_hp'].append(env.player.hp)
            df_dict['p_atk'].append(env.player.atk)
            df_dict['p_def'].append(env.player.def_)
            df_dict['p_money'].append(env.player.money)
            df_dict['p_exp'].append(env.player.exp)
            df_dict['p_yellowKey'].append(env.player.items['yellowKey'])
            df_dict['p_blueKey'].append(env.player.items['blueKey'])
            df_dict['p_redKey'].append(env.player.items['redKey'])
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
        env.step(feasible_actions[index])
    # 輸出檔案
    df = pd.DataFrame(df_dict)
    df.to_excel(file_name, index=False)
    print(df)
