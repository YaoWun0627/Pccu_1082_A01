# -*- coding: utf-8 -*-
"""
【使用已訓練模型來玩魔塔】                        by Hung1  2020/5/2
讓代理利用模型輸入來選擇動作，直接實際的玩一個魔塔遊戲。
"""
import os
import time
import random
import numpy as np
from environment import Mota
from keras.models import load_model
from sklearn.externals import joblib

def feature_engineering(player_state, states, action_a, action_b, assigned):
    """
    特徵工程，將兩個行動轉成可以輸入模型的特徵
    """
    class_a = assigned['class'][action_a.class_]
    class_b = assigned['class2'][action_b.class_]
    if action_a.class_ == 'enemys':
        special_a = action_a.special
    else:
        special_a = 0
    if action_b.class_ == 'enemys':
        special_b = action_b.special
    else:
        special_b = 0
    array = np.hstack((player_state,
                       class_a, states[action_a], special_a,
                       class_b, states[action_b], special_b))
    array = np.expand_dims(array, axis=0)
    return array

def feature_engineering2(player_state, states, action, assigned):
    """
    特徵工程，將一個行動轉成可以輸入模型的特徵
    """
    class_ = assigned['class'][action.class_]
    if action.class_ == 'enemys':
        special = action.special
    else:
        special = 0
    array = np.hstack((player_state,
                       class_, states[action], special))
    array = np.expand_dims(array, axis=0)
    return array
    
    

env = Mota()
env.build_env('mapData_3')
env.build_env('24層魔塔(7層)')
env.create_nodes()
policy_model_file_name = 'model/policy_rfc_model.pkl'
value_model_file_name = 'model/value_rfc_model.pkl'

# 載入模型
policy_file_name = os.path.splitext(policy_model_file_name)
if policy_file_name[1] == '.h5':
    policy_model = load_model(policy_model_file_name)
elif policy_file_name[1] == '.pkl':
    policy_model = joblib.load(policy_model_file_name)
policy_labels_assigned = np.load(policy_file_name[0] + '_labels.npy').item()
value_file_name = os.path.splitext(value_model_file_name)
if value_file_name[1] == '.h5':
    value_model = load_model(value_model_file_name)
elif value_file_name[1] == '.pkl':
    value_model = joblib.load(value_model_file_name)
value_labels_assigned = np.load(value_file_name[0] + '_labels.npy').item()

# 開始遊戲
startTime = time.perf_counter()
ending = 'continue'
while ending == 'continue':
    # 獲取所有行動
    actions = env.get_feasible_actions()
    if not actions:
        ending = 'no actions'
        break
    # 每個行動的狀態差
    states = {}
    b = env.get_player_state()
    for action in actions:
        env.step(action)
        a = env.get_player_state()
        states[action] = a - b
        env.back_step(1)
    # 行動多次驗證(交叉驗證)
    best_actions_index = []
    for _ in range(1):
        ''' #!!!model名稱
        # 二分尋路
        random_index = random.sample(range(len(actions)), len(actions))
        while len(random_index) > 1:
            index_a = random_index.pop()
            index_b = random_index.pop()
            feature_row = feature_engineering(env.get_player_state(), states,
                                              actions[index_a], actions[index_b], labels_assigned)
            if file_name[1] == '.h5':
                weights = model.predict(feature_row)[0][0]
            elif file_name[1] == '.pkl':
                weights = model.predict_proba(feature_row)[0][1]
            if round(weights):
                random_index.append(index_b)
            else:
                random_index.append(index_a)
        best_actions_index.append(random_index[0])
        '''
        best_index = 0
        best_weights = 0
        for index in range(len(actions)):
            feature_row = feature_engineering2(env.get_player_state(), states,
                                               actions[index], value_labels_assigned)
            if value_file_name[1] == '.h5':
                weights = value_model.predict(feature_row)[0][0]
            elif value_file_name[1] == '.pkl':
                weights = value_model.predict_proba(feature_row)[0][1]
            if weights > best_weights:
                best_weights = weights
                best_index = index
        best_actions_index.append(best_index)
    
    # 選擇出現次數最多的行動
    label, count = np.unique(best_actions_index, return_counts=True)
    indexs, = np.where(count==np.max(count))
    index = np.random.choice(indexs)
    # 執行行動
    ending = env.step(actions[label[index]])

# 遊戲成績
print('花費時間：', time.perf_counter() - startTime)
print('★★★★★ 遊戲結果 ★★★★★')
print('結局：', ending)
print('分數：', env.player.hp)
print('前進步數：', len(env.observation))
print('最高樓層：', np.max([env.n2p[n][0] for n in env.observation]))
