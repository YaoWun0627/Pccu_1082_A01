
from import_tool import *
'''
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
from lzw import LZW
from environment_maze import Mota_Maze
from typing import List
# 迷宮版QLearning
class Maze_QLearning:
    def __init__(self,
                 actions=list(range(5)),
                 learning_rate=0.9, # 學習率
                 reward_decay=0.9, # 折扣因子
                 e_greedy=0.9 ): # 隨機機率
        self.actions = actions
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.lr = learning_rate
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float32)# Q_Table內存

    def check_state(self, observation: List[int]):
        if observation not in self.q_table.index:  # 如果狀態不再Q_table中
            self.q_table = self.q_table.append(    # 加入 狀態 至 尾端
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = observation
                )
            )

    def choose_action(self, observation:List[int]):
        self.check_state(observation) #查看next_state存不存在
        if np.random.uniform() < self.epsilon:  #依照 隨機機率
            # 隨機選擇動作
            observation_action = self.q_table.loc[observation, :]
            # 依照狀態選擇最高值
            action = np.random.choice(observation_action[observation_action == np.max(observation_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, observation:List[int], action:int, reward:int, observation_:List[int], end:bool):
        self.check_state(observation_) # 檢查狀態
        old_value = self.q_table.loc[observation, action]
        #  Q(S,A) ← Q(S,A) + α[R+γ maxQ(S',α)]
        if end:
            target_value = reward + self.lr * (reward - old_value)
        else:
           target_value = reward + self.gamma * self.q_table.loc[observation_, :].max()
        self.q_table.loc[observation, action] += self.lr * (target_value - old_value)

# 節點版QLearning
class Node_QLearning:
    DATA_TYPE = np.float32                # Q表的資料型態

    def __init__(self,
                 learning_rate: float   = 0.01,
                 discount_factor: float = 0.9,
                 e_greedy: float        = 0.1):
        self.alpha = learning_rate        # 學習速率
        self.gamma = discount_factor      # 折扣因子
        self.epsilon = e_greedy           # 隨機機率
        self.cost_capacity = 0            # 儲存格計數
        self.q_table = {}

    def create_state_qtable(self, state: str, actions: List[object]):
        self.q_table[state] = pd.Series(
                [0] * len(actions),
                index = actions,
                dtype = Node_QLearning.DATA_TYPE
            )
        self.cost_capacity += len(actions)

    def choose_action(self, state: str) -> tuple:
        if np.random.rand() < self.epsilon:
            # 隨機選擇方向行動
            action = np.random.choice(self.q_table[state].index)
        else:
            # 依照最高分數行動，若有複數個最高值，從中隨機取一個
            state_action = self.q_table[state]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action

    def learn(self, state: str, action: object, reward: int, next_state: str, terminal: bool):
        old_value = self.q_table[state][action]
        if terminal:
            self.q_table[state][action] = old_value + self.alpha * (reward - old_value)
        else:
            #  Q(S,A) ← (1-α)Q(S,A)+α[R+γ maxQ(S',α)]
            learned_value = reward + self.gamma * self.q_table[next_state].max()
            self.q_table[state][action] = (1 - self.alpha) * old_value + self.alpha * learned_value

# 節點版QLearning v2
class Node_QLearning_v2(Node_QLearning):

    def create_state_qtable(self, state: str, actions: List[object], rewards: List[int]):
        self.q_table[state] = pd.Series(
                rewards,
                index = actions,
                dtype = Node_QLearning.DATA_TYPE
            )
        self.cost_capacity += len(actions)