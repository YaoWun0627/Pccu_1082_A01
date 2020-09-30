# -*- coding: utf-8 -*-
#import numpy as np
from import_tool import *
#============================================================================
#  Node
#----------------------------------------------------------------------------
#  節點(集)
#============================================================================
class Node:
    #------------------------------------------------------------------------
    #  初始化
    #------------------------------------------------------------------------
    def __init__(self):
        self.children = set()               # 子節點的列表
        self.child_scores = {}              # 子節點的期望分數
        self.child_visits = {}              # 子節點的訪問次數
    #------------------------------------------------------------------------
    #  ★獲取權重係數，這裡可以自己訂立規則！
    #------------------------------------------------------------------------
    def get_weights_coefficient(self, visits):
        #return 1.5
        n = np.min(visits)# / 10

        if n <= 1:
            return 1
        else:
            return np.log2(n)
    #------------------------------------------------------------------------
    #  獲取分數和走訪次數
    #------------------------------------------------------------------------
    def get_scores_and_visits(self, actions):
        scores = []
        visits = []
        for action in actions:
            if action not in self.children:
                self.add_action(action)
            scores.append(self.child_scores[action])
            visits.append(self.child_visits[action])
        return scores, visits
    #------------------------------------------------------------------------
    #  添加行動
    #------------------------------------------------------------------------
    def add_action(self, action):
        self.children.add(action)
        self.child_scores[action] = 0
        self.child_visits[action] = 0
    #------------------------------------------------------------------------
    #  選擇
    #------------------------------------------------------------------------
    def select(self, actions):
        scores, visits = self.get_scores_and_visits(actions)
        # 權重參數c
        c = self.get_weights_coefficient(visits)
        # 權重
        _, indices = np.unique(scores, return_inverse=True)
        weights = np.power(c, indices)
        # 將權重轉成機率
        probability = weights / np.sum(weights)
        # 依機率選擇行動
        index = np.random.choice(len(weights), p=probability)
        return actions[index]
    #------------------------------------------------------------------------
    #  最高分數選擇
    #------------------------------------------------------------------------
    def max_select(self, actions):
        scores, _ = self.get_scores_and_visits(actions)
        x = np.max(scores)
        index = np.random.choice(np.where(scores == x)[0])
        return actions[index]
    #------------------------------------------------------------------------
    #  更新，r=學習率
    #------------------------------------------------------------------------
    def update(self, action, score, r=0.1):
        self.child_scores[action] = self.child_scores[action] * (1 - r) + score * r
        self.child_visits[action] += 1

#============================================================================
#  Explore
#----------------------------------------------------------------------------
#  啟發式搜尋，特色為節點共享學習經驗，大幅減少資料儲存空間。適用於魔塔環境。
#============================================================================
class Explore:
    #------------------------------------------------------------------------
    #  初始化
    #------------------------------------------------------------------------
    def __init__(self):
        self.data_set = {}                  # 資料集
        self.visit_node = {}                # 走訪過的節點
        self.enter_num = {}                 # 觀測值進入次數
    #------------------------------------------------------------------------
    #  選擇行動
    #------------------------------------------------------------------------
    def choose_action(self, observation, actions, max_select=False):
        # 不重複觀測值
        if observation not in self.enter_num:
            self.enter_num[observation] = 0
        else:
            self.enter_num[observation] += 1
            observation = observation + (self.enter_num[observation],)
        # 建立資料集
        if observation not in self.data_set:
            self.data_set[observation] = Node()
        node = self.data_set[observation]
        # 動作選擇
        if max_select:
            action = node.max_select(actions)
        else:
            action = node.select(actions)
        # 添加進已行動序列
        self.visit_node[node] = action
        return action
    #------------------------------------------------------------------------
    #  反向傳播
    #------------------------------------------------------------------------
    def backpropagate(self, score):
        for node, action in self.visit_node.items():
            node.update(action, score)
        # 重置
        self.visit_node.clear()
        self.enter_num.clear()

    def run(self, env, rounds, agent):
        score = 0
        # 紀錄資料

        for episode in range(1, rounds+1):
            while True:
                actions = env.get_feasible_actions()
                # 選擇行動
                if actions:
                    action = agent.choose_action(env.n2p[env.observation[-1]], actions)
                    pos = env.n2p[action]
                    ending = env.step(action)
                    if ending == 'clear':
                        score = env.player.hp
                    if ending == 'continue':
                        done = False
                    else:
                        done = True
                    yield pos, done, episode + 1, score
                else:
                    ending = 'stop'

                if ending != 'continue':
                    # 顯示路徑
                    #print([env.n2p[node] for node in env.observation])
                    break
            # 結局成績
            if ending == 'clear':
                score = env.player.hp
            # 反向傳播
            f = np.max([env.n2p[n][0] for n in env.observation])
            l = len(env.observation)
            agent.backpropagate(score + f + 0.001 * l)
            env.reset()
