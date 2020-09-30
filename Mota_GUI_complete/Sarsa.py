from QLearning import *

# 迷宮版Sarsa
class Maze_Sarsa(Maze_QLearning):
    def __init__(self,
                 actions = list(range(5)),
                 learning_rate = 0.01,
                 reward_decay = 0.9,
                 e_greedy = 0.1,
                 lambda_ = 0.9
    ):
        super(Maze_Sarsa, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        self.lambda_ = lambda_
        self.traceing = self.q_table.copy()

    def check_state(self, observation: List[int]):
        if observation not in self.q_table.index:  # 如果狀態不再Q_table中
           observation_table = pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = observation)
           self.q_table = self.q_table.append(observation_table)
           self.traceing =self.traceing.append(observation_table)


    def learn(self, observation:List[int], action:int, reward:int, observation_:List[int], action_:int, end:bool):
        self.check_state(observation_) # 檢查狀態
        old_value = self.q_table.loc[observation, action]
        #  Q(S,A) ← Q(S,A) + α[R+γ Q(s',a')-(s,a)]
        if end:
            target_value = reward + self.lr * (reward - old_value)
        else:
           target_value = reward + self.gamma * self.q_table.loc[observation_, action_]

        self.traceing.loc[observation, :] *=0
        self.traceing.loc[observation, action]  =1
        self.q_table = self.q_table + self.lr * (target_value - old_value) * self.traceing
        self.traceing = self.traceing * self.gamma * self.lambda_

# 節點版Sarsa
class Node_Sarsa(Node_QLearning):

    def __init__(self,
                 learning_rate: float   = 0.01,
                 discount_factor: float = 0.9,
                 e_greedy: float        = 0.1):
        super(Node_Sarsa, self).__init__(learning_rate, discount_factor, e_greedy)

    def learn(self, state: str, action: object, reward: int, next_state: str,
              next_action: object, terminal: bool):
        old_value = self.q_table[state][action]
        if terminal:
            self.q_table[state][action] = old_value + self.alpha * (reward - old_value)
        else:
            # 與Q-Learning差別在於 q[s'].max 改成 q[s'][a']
            learned_value = reward + self.gamma * self.q_table[next_state][next_action]
            self.q_table[state][action] = (1 - self.alpha) * old_value + self.alpha * learned_value