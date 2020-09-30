from import_tool import *

#============================================================================
#   MCTS From Demonstration Modle
#============================================================================
class MCTSfDm_Algorithm(object):
#------------------------------------------------------------------------
#  初始化
#------------------------------------------------------------------------
    def __init__(self, actions: list, env: object, c: float = np.sqrt(2)):
        self.root = treeNode(actions)# 根節點
        self.visit_path = [] # 行動序列
        self.c = c
#------------------------------------------------------------------------
        self.env = env
        self.load_model() #載入模型
#------------------------------------------------------------------------
#  選擇
#------------------------------------------------------------------------
    def select(self) -> list:
        node = self.root
        # 清除行動序列
        self.visit_path.clear()
        steps = []
        self.visit_path.append(node)
        # 從根節點開始，遞迴選擇子節點，直到達到葉子節點
        while node.all_visit:
            action, node, index = node.select(self.c)
            steps.append(action)
            self.visit_path.append(node)
        return steps
#------------------------------------------------------------------------
#  選擇擴展節點
#------------------------------------------------------------------------
    def choose_expansion_node(self) -> object:
        node = self.visit_path[-1]
        # 探索沒走過的子節點
        unvisited = np.where(node.children==0)[0]
        expand_index = unvisited[0]
        # 檢查所有子節點是否都走訪過
        if len(unvisited) == 1:
            node.all_visit = True
        return node.actions[expand_index], expand_index
#------------------------------------------------------------------------
#  擴展
#------------------------------------------------------------------------
    def expand(self, expand_index: int, expand_actions: list):
        # 創建一個子節點
        expand_node = treeNode(expand_actions)
        # 與父節點連結
        node = self.visit_path[-1]
        node.children[expand_index] = expand_node
        # 添加到行動序列
        self.visit_path.append(expand_node)
#------------------------------------------------------------------------
#  反向傳播
#------------------------------------------------------------------------
    def backpropagate(self, score: int):
        # 用模擬的結果輸出，更新當前行動序列
        for node in self.visit_path[::-1]:
            node.visits += 1
            if score > node.score:
                node.score = score
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
        return actions[best_index]
#============================================================================
#  treeNode
#----------------------------------------------------------------------------
#  蒙特卡洛樹的節點。
#============================================================================
class treeNode:
#------------------------------------------------------------------------
#  初始化
#------------------------------------------------------------------------
    def __init__(self, actions: list):
        self.score = 0                      # 獲勝次數
        self.visits = 0                     # 訪問次數
        self.actions = np.array(actions)    # 所有行動選項
        self.children = np.zeros(           # 子節點的列表
                len(actions), dtype=object)
        self.all_visit = False              # 是否每個子節點都走訪過
#------------------------------------------------------------------------
#  選擇
#------------------------------------------------------------------------
    def select(self, c: float) -> object:
        # 根據UCB公式選擇子節點
        ucb_base = self.get_ucb_base()
        #ucb_base = np.array(
        #        [node.win / node.visits for node in self.children],
        #        dtype=np.float16)
        ucb_bonus = np.array(
                [c * np.sqrt(np.log(self.visits) / node.visits) for node in self.children],
                dtype=np.float16)
        index = np.argmax(ucb_base + ucb_bonus)
        # 返回選擇的行動
        return self.actions[index], self.children[index], index
#------------------------------------------------------------------------
#  獲取UCB基礎值
#------------------------------------------------------------------------
    def get_ucb_base(self):
        scores = np.array([node.score for node in self.children])
        # 將重複分數取出
        scores, indices = np.unique(scores, return_inverse=True)
        length = len(scores)
        # 依照分數進行排序，並對數值做正規化
        bonus = np.array([1 - i / length for i in range(length, 0, -1)],
                          dtype=np.float16)
        return bonus[indices]

if __name__ == '__main__':
    pass
