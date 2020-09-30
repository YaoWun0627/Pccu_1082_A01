"""
Created on Thu Jun  4 16:24:58 2020

@author: yoyo4
"""
from import_tool import *
from lzw import LZW
from Maze_DeepQNetwork import DeepQlearning
from Maze_DoubleDeepQNetwork import DoubleDeepQlearning
from prioritizedDQN import PrioritizedReplayDQN
from QLearning import Maze_QLearning, Node_QLearning, Node_QLearning_v2
from Maze_Actor_Critic import ActorCritic
from Sarsa import Maze_Sarsa, Node_Sarsa
from MCTS import MCTS
from MT_Star import Explore

#============================================================================
#  所有演算法學習
#============================================================================
class learning(object):
    def __init__(self, env, algorithm_name, rounds):
        self.env = env
        self.algorithm_name = algorithm_name
        self.rounds = rounds
        self.switch = True
#----------------------------------------------------------------------------
#   迷宮版學習
#----------------------------------------------------------------------------
    def maze_mota_learning(self):
        self.agent = None
        if self.algorithm_name =='Q-Learning':
            self.agent = Maze_QLearning()
            for ob, done, episode, score in self.Maze_QL_run(self.rounds):
                yield ob, done, episode, score

        if self.algorithm_name =='SarsaLambda':
            self.agent = Maze_Sarsa()
            for ob, done, episode, score in self.Maze_sarsa_run(self.rounds):
                yield ob, done, episode, score

        if self.algorithm_name == 'DeepQNetwork':
            self.agent = DeepQlearning()
            for ob, done, episode, score in self.DQN_run(self.rounds):
                yield ob, done, episode, score

        if self.algorithm_name == 'DoubleDeepQNetwork':
            self.agent = DoubleDeepQlearning()
            for ob, done, episode, score in self.DDQN_run(self.rounds):
                yield ob, done, episode, score

        if self.algorithm_name == 'PrioritizedDQN':
            self.agent = PrioritizedReplayDQN()
            for ob, done, episode, score in self.PRDQN_run(self.rounds):
                yield ob, done, episode, score

        if self.algorithm_name == 'Actor-Critic':
            self.agent = ActorCritic()
            for ob, done, episode, score in self.AC_run(self.rounds):
                yield ob, done, episode, score
#----------------------------------------------------------------------------
#   節點版學習
#----------------------------------------------------------------------------
    def Node_mota_learning(self):
        self.agent = None
        if self.algorithm_name =='Q-Learning':
            self.agent = Node_QLearning()
            for ob, done, episode, score, save_action in self.Node_QL_run(self.rounds):
                yield ob, done, episode, score, save_action

        if self.algorithm_name =='Q-Learning v2':
            self.agent = Node_QLearning_v2()
            for ob, done, episode, score, save_action in self.Node_QL_v2_run(self.rounds):
                yield ob, done, episode, score, save_action

        if self.algorithm_name =='Sarsa':
            self.agent = Node_Sarsa()
            for ob, done, episode, score, save_action in self.Node_Sarsa_run(self.rounds):
                yield ob, done, episode, score, save_action

        if self.algorithm_name =='MCTS':
            self.agent = MCTS(self.env.get_feasible_actions())
            for ob, done, episode, score, save_action in self.MCTS_run(self.rounds):
                yield ob, done, episode, score, save_action
        if self.algorithm_name =='MCTS v2':
            self.agent = Explore()
            for ob, done, episode, score, save_action in self.MT_Star_run(self.rounds):
                yield ob, done, episode, score, save_action
#----------------------------------------------------------------------------
#   迷宮版結局判斷
#----------------------------------------------------------------------------
    def ending_read(self,ending):
        score = 0
        ending_set = ['continue','invalid']
        if ending not in ending_set:
            done = True
            if ending == 'clear':
                score = self.env.player.hp
        else:
            done = False
        return score, done
#----------------------------------------------------------------------------
#   開啟節點版動畫
#----------------------------------------------------------------------------
    def on_animation(self):
        self.switch = True
#----------------------------------------------------------------------------
#   關閉節點版動畫
#----------------------------------------------------------------------------
    def off_animation(self):
        self.switch = False
#============================================================================
#  節點版演算法
#============================================================================
#----------------------------------------------------------------------------
#  Node_QL
#----------------------------------------------------------------------------
    def Node_QL_run(self, rounds):
        print('Running Node_QLearning')
        lzw = LZW()
        state = lzw.compress(self.env.observation)
        _, actions = self.env.get_actions2()
        self.agent.create_state_qtable(state, actions)
        # 訓練迴圈
        score = 0
        for _ in range(rounds):
            # 儲存動作list
            self.save_action = []
            # 新一次訓練開始
            self.env.reset(refresh_frame=True)
            # 初始化觀測值
            state = lzw.compress(self.env.observation)
            while True:
                actions = self.env.get_feasible_actions()
                # 選擇下一步位置
                action = self.agent.choose_action(state)
                pos = self.env.n2p[action]
                # 採取行動並獲得觀測和獎勵值
                index = actions.index(action)
                # 儲存動作
                self.save_action.append(index)
                ending, reward = self.env.step(actions[index],return_reward =True ,refresh_frame=self.switch)
                state_ = lzw.compress(self.env.observation)
                if ending == 'clear':
                    score = self.env.player.hp
                if ending == 'continue':
                    done = False
                    # 檢查下一狀態的Q表是否存在
                    if state_ not in self.agent.q_table:
                        __, actions = self.env.get_actions2()
                        if actions:
                            self.agent.create_state_qtable(state_, actions)
                        else:
                            # 刪除前次狀態的行動
                            self.agent.q_table[state].drop(action)
                            ending = 'stop'
                            done = True
                else:
                    done = True
                yield pos, done, _, score, self.save_action
                # 從過程中學習
                self.agent.learn(state, action, reward, state_, done)
                # 將狀態傳到下一次循環
                state = state_
                # 如果走到終點，結束本回合
                if done:
                    break
#----------------------------------------------------------------------------
#  Node_QL_v2
#----------------------------------------------------------------------------
    def Node_QL_v2_run(self, rounds):
        print('Running Node_QLearning v2')
        lzw = LZW()
        # 建立初始Q表
        state = lzw.compress(self.env.observation)
        _ , actions = self.env.get_actions2()
        self.agent.create_state_qtable(state, actions, [0] * len(actions))
        # 訓練迴圈
        score = 0
        for _ in range(rounds):
            # 儲存動作list
            self.save_action = []
            # 新一次訓練開始
            self.env.reset(refresh_frame=True)
            # 初始化觀測值
            state = lzw.compress(self.env.observation)
            while True:
                actions = self.env.get_feasible_actions()
                # 選擇下一步位置
                action = self.agent.choose_action(state)
                pos = self.env.n2p[action]
                # 採取行動並獲得觀測和獎勵值
                index = actions.index(action)
                # 儲存動作
                self.save_action.append(index)
                ending, reward = self.env.step(actions[index],return_reward =True,refresh_frame=self.switch)
                state_ = lzw.compress(self.env.observation)
                if ending == 'clear':
                    score = self.env.player.hp
                if ending == 'continue':
                    done = False
                    # 檢查下一狀態的Q表是否存在
                    if state_ not in self.agent.q_table:
                        rewards, actions = self.env.get_actions2()
                        if actions:
                            # 建立Q表
                            self.agent.create_state_qtable(state_, actions, rewards)
                        else:
                            # 刪除前次狀態的行動
                            self.agent.q_table[state].drop(action)
                            ending = 'stop'
                            done = True
                else:
                    done = True
                yield pos, done, _, score, self.save_action
                # 從過程中學習
                self.agent.learn(state, action, reward, state_, done)
                # 將狀態傳到下一次循環
                state = state_
                # 如果走到終點，結束本回合
                if done:
                    break
#----------------------------------------------------------------------------
#  Node_Sarsa
#----------------------------------------------------------------------------
    def Node_Sarsa_run(self, rounds):
        print('Running Node_Sarsa')
        pos = None
        self.save_action = []
        lzw = LZW()
        # 建立初始狀態
        state = lzw.compress(self.env.observation)
        self.agent.create_state_qtable(state, self.env.get_actions())
        # 訓練迴圈
        score = 0
        for _ in range(rounds):
            # 新一次訓練開始
            self.env.reset(refresh_frame=True)
            # 根據觀察選擇行動
            state = lzw.compress(self.env.observation)
            action = self.agent.choose_action(state)
            while True:
                # 採取行動並獲得觀測和獎勵值
                ending, reward = self.env.step(action, return_reward =True, refresh_frame=self.switch)
                state_ = lzw.compress(self.env.observation)
                if ending == 'clear':
                    score = self.env.player.hp
                    # 儲存通關路徑
                    save_observation = self.env.observation.copy()
                    self.save_action_ = []
                    self.env.reset()
                    for action in save_observation[len(self.env.observation):]:
                        self.save_action_.append(self.env.get_feasible_actions().index(action))
                        self.env.step(action)
                    self.save_action = self.save_action_.copy()
                if ending == 'continue':
                    done = False
                    # 檢查下一狀態的Q表是否存在
                    if state_ not in self.agent.q_table:
                        self.agent.create_state_qtable(state_, self.env.get_actions())
                    # 根據下次觀察選擇行動
                    action_ = self.agent.choose_action(state_)
                    pos = self.env.n2p[action_]
                else:
                    done = True
                    action_ = None
                yield pos, done, _, score, self.save_action
                # 從過程中學習
                self.agent.learn(state, action, reward, state_, action_, done)
                # 更新狀態和行動
                state = state_
                action = action_
                # 如果走到終點，結束本回合
                if done:
                    break
#----------------------------------------------------------------------------
#  MCTS
#----------------------------------------------------------------------------
    def MCTS_run(self, rounds):
        score = 0
        # 儲存動作list
        self.save_action = []
        for episode in range(1, rounds+1):
            #!!!root被刪掉
            # 選擇：從根節點R開始，遞迴選擇子節點，直到達到葉子節點L
            for action in self.agent.select():
                self.env.step(action, refresh_frame=self.switch)
            # 擴展：如果葉子節點L不是一個終止節點，創建並選擇一個子節點C
            ending = self.env.step(self.agent.choose_expansion_node(), refresh_frame=self.switch)
            actions = self.env.get_feasible_actions()
            if ending == 'continue' and actions:
                self.agent.expand(actions)
            else:
                self.agent.condense()
            # 模擬：從子節點C開始，用隨機策略進行遊戲(又稱為playout或者rollout)，直到遊戲結束
            while ending == 'continue' and actions:
                action = np.random.choice(actions)
                pos = self.env.n2p[action]
                ending = self.env.step(action, refresh_frame=self.switch)
                if ending == 'clear':
                    score = self.env.player.hp
                    # 儲存通關路徑
                    save_observation = self.env.observation.copy()
                    self.save_action_ = []
                    self.env.reset()
                    for action in save_observation[len(self.env.observation):]:
                        self.save_action_.append(self.env.get_feasible_actions().index(action))
                        self.env.step(action)
                    self.save_action = self.save_action_
                if ending == 'continue':
                    done = False
                else:
                    done = True
                if not self.env.get_feasible_actions():
                    done = True
                yield pos, done, episode - 1, score, self.save_action
                #actions = env.get_actions()      # 速度快上10倍
                actions = self.env.get_feasible_actions() # 速度較慢
            # 反向傳播：使用隨機遊戲的結果，更新從C到R的路徑上的節點資訊
            if ending == 'clear':
                score = self.env.player.hp
            else:
                score = 0
            self.agent.backpropagate(score)
            self.env.reset(refresh_frame=True)

#----------------------------------------------------------------------------
#  MT_Star
#----------------------------------------------------------------------------
    def MT_Star_run(self, rounds):
        score = 0
        # 紀錄資料
        for episode in range(1, rounds+1):
            # 儲存動作list
            self.save_action = []
            while True:
                actions = self.env.get_feasible_actions()
                # 選擇行動
                if actions:
                    action = self.agent.choose_action(self.env.n2p[self.env.observation[-1]], actions)
                    # 儲存動作
                    index = actions.index(action)
                    self.save_action.append(index)
                    pos = self.env.n2p[action]
                    ending = self.env.step(action, refresh_frame=self.switch)
                    if ending == 'clear':
                        score = self.env.player.hp
                    if ending == 'continue':
                        done = False
                    else:
                        done = True
                    if not self.env.get_feasible_actions():
                        done = True
                    yield pos, done, episode - 1, score, self.save_action
                else:
                    ending = 'stop'

                if ending != 'continue':
                    # 顯示路徑
                    #print([env.n2p[node] for node in env.observation])
                    break
            # 結局成績
            if ending == 'clear':
                score = self.env.player.hp
            # 反向傳播
            f = np.max([self.env.n2p[n][0] for n in self.env.observation])
            l = len(self.env.observation)
            self.agent.backpropagate(score + f + 0.001 * l)
            self.env.reset(refresh_frame=True)
#============================================================================
#  迷宮版演算法
#============================================================================
#----------------------------------------------------------------------------
#  Maze_QL
#----------------------------------------------------------------------------
    def Maze_QL_run(self, rounds):
        print('Running Maze_QLearning')
        score = 0
        for episode in range(rounds):
            observation = self.env.reset()
            while True:
                action = self.agent.choose_action(str(observation))
                observation_, reward, ending = self.env.step(action)
                score, done = self.ending_read(ending)
                if (action==4) and (reward == -1 or reward == 0):
                    reward = -100
                self.agent.learn(str(observation), action, reward,str(observation_),done)
                yield observation_, done, episode, score
                observation = observation_
                if done:
                    break
#----------------------------------------------------------------------------
#  Sarsa
#----------------------------------------------------------------------------
    def Maze_sarsa_run(self, rounds):
        print('Running SarasLambda')
        score = 0
        for episode in range(rounds):
            observation = self.env.reset()
            action = self.agent.choose_action(str(observation))
            self.agent.traceing *= 0
            while True:
                observation_, reward, ending = self.env.step(action)
                action_ = self.agent.choose_action(str(observation_))
                score, done = self.ending_read(ending)
                if tuple(observation[:3]) != tuple(observation_[:3]):
                    reward = 1
                if (action==4) and (reward == -1 or reward == 0):
                    reward = -100
                yield observation_, done, episode, score
                self.agent.learn(str(observation), action, reward,str(observation_),action_,done)
                observation = observation_
                action = action_
                if done:
                    break
#----------------------------------------------------------------------------
#  DeepQNetwork
#----------------------------------------------------------------------------
    def DQN_run(self,rounds):
        print('Runnung DQN')
        step = 0
        for episode in range(rounds):
            observation = self.env.reset()
            while True:
                action = self.agent.choose_action(observation)
                observation_, reward, ending = self.env.step(action)
                score, done = self.ending_read(ending)
                if (action == 4 ) and (reward == -1 or reward == 0):
                    reward = -200
                if tuple(observation[:3]) != tuple(observation_[:3]):
                    reward = 1
                self.agent.store_transition(observation, action, reward, observation_)
                if (step > 1000) :
                    self.agent.learn()
                yield observation_, done, episode, score  ## 這裡
                observation = observation_
                step+=1
                if done:
                    break
#----------------------------------------------------------------------------
#  DoubleDeepQNetwork
#----------------------------------------------------------------------------
    def DDQN_run(self,rounds):
        print('Runnung DDQN')
        step = 0
        for episode in range(rounds):
            observation = self.env.reset()
            while True:
                action = self.agent.choose_action(observation)
                observation_, reward, ending = self.env.step(action)
                score, done = self.ending_read(ending)
                if (action == 4 ) and (reward == -1 or reward == 0):
                    reward = -200
                if tuple(observation[:3]) != tuple(observation_[:3]):
                    reward = 1
                self.agent.store_transition(observation, action, reward, observation_)
                if (step > 1000) :
                    self.agent.learn()
                yield observation_, done, episode, score
                observation = observation_
                step+=1
                if done:
                    break
#----------------------------------------------------------------------------
#  PrioritizedDQN
#----------------------------------------------------------------------------
    def PRDQN_run(self,rounds):
        print('Runnung DDQN')
        step = 0
        for episode in range(rounds):
            observation = self.env.reset()
            while True:
                action = self.agent.choose_action(observation)
                observation_, reward, ending = self.env.step(action)
                score, done = self.ending_read(ending)
                if (action == 4 ) and (reward == -1 or reward == 0):
                    reward = -200
                if tuple(observation[:3]) != tuple(observation_[:3]):
                    reward = 1
                self.agent.store_transition(observation, action, reward, observation_)
                if (step > 1000) :
                    self.agent.learn()
                yield observation_, done, episode, score  ## 這裡
                observation = observation_
                step+=1
                if done:
                    break
#----------------------------------------------------------------------------
#  AC
#----------------------------------------------------------------------------
    def AC_run(self,rounds):
        print('Running AC')
        for episode in range(rounds):
            observation = self.env.reset()
            while True:
                action = self.agent.choose_action(observation)
                observation_, reward, ending = self.env.step(action)
                score, done = self.ending_read(ending)
                self.agent.learn(observation,action, reward/20, observation_, done)
                yield observation_, done, episode, score
                observation = observation_
                if done:
                    break
#============================================================================
#  main
#============================================================================
if __name__ == '__main__':
    pass