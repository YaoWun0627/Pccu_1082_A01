from import_tool import *
from lzw import LZW
from typing import List

class imitation_form_QLearning(object):

    def __init__(self,
                 env,
                 learning_rate: float   = 0.01,
                 discount_factor: float = 0.9,
                 e_greedy: float        = 0.1):
        self.alpha = learning_rate        # 學習速率
        self.gamma = discount_factor      # 折扣因子
        self.epsilon = e_greedy           # 隨機機率
        self.cost_capacity = 0            # 儲存格計數
        self.q_table = {}
        self.env = env
        self.load_model()


    def create_state_qtable(self, state: str, actions: List[object], rewards: List[int]):
        self.actions = actions
        self.q_table[state] = pd.Series(
                rewards,
                index = actions,
                dtype = np.int16
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

    def learn(self, state: str, action: object, best_action:object, next_state: str, terminal: bool):
        old_value = self.q_table[state][action]
        if action == best_action:
            demonstration_reward = 1
        else:
            demonstration_reward = 0

        if terminal:
            self.q_table[state][action] = old_value + self.alpha * (demonstration_reward - old_value)
        else:
            #  Q(S,A) ← (1-α)Q(S,A)+α[R+γ maxQ(S',α)]
            learned_value = demonstration_reward + self.gamma * self.q_table[next_state].max()
            self.q_table[state][action] = (1 - self.alpha) * old_value + self.alpha * learned_value
#------------------------------------------------------------------------
#  載入模型
#------------------------------------------------------------------------
    def load_model(self):
        try:
            self.model = joblib.load('model/MCTSfD_model.pkl')
            self.labels_assigned = np.load('model/MCTSfD_labels.npy', allow_pickle=True).item()
        except FileNotFoundError:
            print('請先訓練模型')
        feature_id = {'p_hp':0,'p_atk':1,'p_def':2,'p_money':3,'p_exp':4,
                      'p_yellowKey':5,'p_blueKey':6,'p_redKey':7,
                      'class':8,'hp':9,'atk':10,'def':11,'money':12,'exp':13,
                      'yellowKey':14,'blueKey':15,'redKey':16,'special':17,
                      'z':18,'y':19,'x':20,'e':21}
        delete_feature = ['z','y','x','e']
        self.delete_feature = [feature_id[i] for i in delete_feature]
#------------------------------------------------------------------------
#  特徵工程
#------------------------------------------------------------------------
    def feature_engineering(self,before_state, actions, assigned):
        array_list = []
        for action in actions:
            self.env.step(action)
            after_state = self.env.get_player_state()
            state = after_state - before_state
            class_ = assigned['class'][action.class_]
            if action.class_ == 'enemys':
                special = action.special
            else:
                special = 0
            pos = self.env.n2p[action]
            if len(pos) == 3:
                pos = pos + (0,)
            array = np.hstack((before_state, class_, state, special, pos))
            array_list.append(array)
            self.env.back_step(1)
        data = np.vstack(array_list)
        data = np.delete(data, self.delete_feature, axis=1)
        return data
#------------------------------------------------------------------------
#   權重搜尋
#------------------------------------------------------------------------
    def maximum_weight(self,array):
        max_num = np.max(array)
        max_index = np.where(array == max_num)[0]
        return max_num, max_index
#------------------------------------------------------------------------
# DFS
#------------------------------------------------------------------------
    # 以DFS深入搜索最高權重的行動
    # prediction_horizon: 預測視野距離
    def DFS(self, best_index, actions, model, prediction_horizon = 2):
        max_num = -1
        max_path = []
        now_nums = []
        now_path = []
        unvisited = list(best_index)[::-1] #反向添加(不影響結果，只改變搜尋順序)
        pop_times = [0] * len(unvisited)
        actions_ = actions
        have_index_array = True
        while unvisited:
            i = unvisited.pop()
            now_path.append(i)
            pop_times[-1] += 1
            self.env.step(actions_[i])
            actions_ = self.env.get_feasible_actions()
            player_state = self.env.get_player_state()
            if actions_:
                data = self.feature_engineering(player_state, actions_, self.labels_assigned)
                weights = model.predict_proba(data)
                best_num_, best_index_ = self.maximum_weight(weights[:,1])
                now_nums.append(best_num_)
                have_index_array = True
            else:
                if now_nums: #不為空時
                    now_nums.append(sum(now_nums) / len(now_nums)) #平均值
                else:
                    now_nums.append(0)
                have_index_array = False
            if have_index_array and len(now_path) != prediction_horizon:
                # 繼續前進
                unvisited += list(best_index_)[::-1] #反向添加(不影響結果，只改變搜尋順序)
                pop_times += [0] * (len(best_index_) - 1)
            else:
                # 退回
                if sum(now_nums) / len(now_nums) > max_num:
                    max_num = sum(now_nums) / len(now_nums) #平均值
                    max_path = now_path.copy()
                pop_time = pop_times.pop()
                self.env.back_step(pop_time)
                actions_ = self.env.get_feasible_actions()
                now_nums = now_nums[:-pop_time]
                now_path = now_path[:-pop_time]
        return max_path[0]
#------------------------------------------------------------------------
# 預測出最高價值的行動
#------------------------------------------------------------------------
    def predict(self, actions):
        player_state = self.env.get_player_state()
        data = self.feature_engineering(player_state, actions, self.labels_assigned)
        weights = self.model.predict_proba(data)
        _, best_index = self.maximum_weight(weights[:,1])
        if len(best_index) > 1:
            best_index = self.DFS(best_index, actions, self.model)
        else:
            best_index = best_index[0]
        return actions[best_index], weights
#------------------------------------------------------------------------
# 列印成績
#------------------------------------------------------------------------
    def print_score(self, score:List[int], episode:List[int], titel:str):
        plt.title(titel)
        plt.plot()
        plt.plot(episode, score)
        plt.ylabel('score')
        plt.xlabel('episode')
        plt.show()

if __name__ == '__main__':
    from environment import Mota
    env = Mota()
    env.build_env('standard_map')
    env.create_nodes()
    agent = imitation_form_QLearning(env)
    lzw = LZW()

    state = lzw.compress(env.observation)

    actions = env.get_feasible_actions()

    agent.create_state_qtable(state, actions, [0] * len(actions))

    best_score = 0
    score = 0
    rounds = 1000
    best_score_list = []
    time_list = []
    for episode in range(rounds):
        print('Running QLfD')
        env.reset()
        state = lzw.compress(env.observation)
        while True:
            actions = env.get_feasible_actions()
            action = agent.choose_action(state)
            best_action, weights = agent.predict(actions)
            #pos = env.n2p[action]
            index = actions.index(action)
            ending, _ = env.step(actions[index], return_reward = True)
            state_ = lzw.compress(env.observation)
            if ending == 'clear':
                score = env.player.hp
                done = True
            if ending == 'continue':
                done = False
                if state_ not in agent.q_table:
                    rewards, actions = env.get_actions2()

                    if actions:
                        # 建立Q表
                        agent.create_state_qtable(state_, actions, rewards)
                    else:
                        # 刪除前次狀態的行動
                        agent.q_table[state].drop(action)
                        ending = 'stop'
                        done = True
            else:
                done = True

            agent.learn(state, action, best_action, state_, done)

            state = state_
            if best_score < score :
                best_score = score

            best_score_list.append(score)
            time_list.append(episode)

            if done:
                break
        print('episode:',episode+1," ",score," ",best_score)
    agent.print_score(best_score_list, time_list, "QLfD")