# -*- coding: utf-8 -*-
from import_tool import *

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
        self.score = 0                      # 期望分數
        self.visits = 0                     # 訪問次數
        self.actions = np.array(actions)    # 所有行動名稱(子節點)
        self.children = np.zeros(           # 子節點的列表
                len(actions), dtype=object)
        self.ucb_base = np.zeros(           # 子節點的UCB基礎值(依分數排名決定)
                len(actions), dtype=np.float16)
        self.all_visit = False              # 是否每個子節點都走訪過
    #------------------------------------------------------------------------
    #  選擇
    #------------------------------------------------------------------------
    def select(self, c: float) -> object:
        # 根據UCB公式選擇子節點
        ucb_bonus = np.array(
                [c * np.sqrt(np.log(self.visits) / node.visits) for node in self.children],
                dtype=np.float16)
        index = np.argmax(self.ucb_base + ucb_bonus)
        # 返回選擇的行動
        return self.actions[index], self.children[index], index
    #------------------------------------------------------------------------
    #  更新UCB基礎值
    #------------------------------------------------------------------------
    def update_ucb_base(self):
        scores = np.array([node.score for node in self.children])
        # 將重複分數取出
        scores, indices = np.unique(scores, return_inverse=True)
        length = len(scores)
        # 依照分數進行排序，並對數值做正規化
        for i, index in enumerate(np.argsort(scores)):
            self.ucb_base[index] = 1 - i / length
        # 還原數組
        self.ucb_base = self.ucb_base[indices]

#============================================================================
#  MCTS
#----------------------------------------------------------------------------
#  使用蒙特卡洛樹搜索來做學習，適用於魔塔環境。
#  此演算法增加兩種特色：
#  1. 刪除已經走過的選擇，避免再次走重複路線
#  2. 增加第二分數，讓UCT更能發揮作用
#============================================================================
class MCTS:
    #------------------------------------------------------------------------
    #  初始化
    #------------------------------------------------------------------------
    def __init__(self, actions: list, c: float = np.sqrt(2)):
        self.root = treeNode(actions)       # 根節點
        self.visit_path = []                # 行動序列
        self.visit_index = []               # 每個子節點的索引
        self.c = c                          # UCB加權係數
    #------------------------------------------------------------------------
    #  選擇
    #------------------------------------------------------------------------
    def select(self) -> (list, list):
        node = self.root
        # 清除行動序列
        self.visit_path.clear()
        self.visit_index.clear()
        steps = []
        self.visit_path.append(node)
        # 從根節點開始，遞迴選擇子節點，直到達到葉子節點
        while node.all_visit:
            action, node, index = node.select(self.c)
            steps.append(action)
            self.visit_path.append(node)
            self.visit_index.append(index)
        return steps
    #------------------------------------------------------------------------
    #  選擇擴展節點
    #------------------------------------------------------------------------
    def choose_expansion_node(self) -> object:
        node = self.visit_path[-1]
        # 隨機探索沒走過的子節點
        unvisited = np.where(node.children==0)[0]
        expand_index = np.random.choice(unvisited)
        self.visit_index.append(expand_index)
        return node.actions[expand_index]
    #------------------------------------------------------------------------
    #  擴展
    #------------------------------------------------------------------------
    def expand(self, expand_actions: list):
        # 創建一個子節點
        expand_node = treeNode(expand_actions)
        # 與父節點連結
        node = self.visit_path[-1]
        node.children[self.visit_index[-1]] = expand_node
        # 添加到行動序列
        self.visit_path.append(expand_node)
    #------------------------------------------------------------------------
    #  斂縮
    #------------------------------------------------------------------------
    def condense(self):
        # 將子節點從父節點中去除
        node = self.visit_path[-1]
        index = self.visit_index.pop()
        node.actions = np.delete(node.actions, index)
        node.children = np.delete(node.children, index)
        node.ucb_base = np.delete(node.ucb_base, index)
        # 繼續向前刪除空行動節點
        while not len(node.actions):
            self.visit_path.pop()
            node = self.visit_path[-1]
            index = self.visit_index.pop()
            node.actions = np.delete(node.actions, index)
            node.children = np.delete(node.children, index)
            node.ucb_base = np.delete(node.ucb_base, index)
    #------------------------------------------------------------------------
    #  反向傳播
    #------------------------------------------------------------------------
    def backpropagate(self, score: int):
        # 用模擬的結果輸出，更新當前行動序列
        for node in self.visit_path[::-1]:
            node.visits += 1
            if node.all_visit:
                if score > node.score:
                    node.score = score
                    node.update_ucb_base()
            else:
                if score > node.score:
                    node.score = score
                #np.count_nonzero(node.children) #另一種寫法
                if len(node.actions) == len(node.children.nonzero()[0]):
                    node.all_visit = True
                    node.update_ucb_base()

    def run(self, env, rounds, mcts):
        score = 0
        for episode in range(1, rounds+1):
            #!!!root被刪掉
            env.reset()
            # 選擇：從根節點R開始，遞迴選擇子節點，直到達到葉子節點L
            for action in mcts.select():
                env.step(action)
            # 擴展：如果葉子節點L不是一個終止節點，創建並選擇一個子節點C
            ending = env.step(mcts.choose_expansion_node())
            actions = env.get_feasible_actions()
            if ending == 'continue' and actions:
                mcts.expand(actions)
            else:
                print(ending, actions)
                mcts.condense()
            # 模擬：從子節點C開始，用隨機策略進行遊戲(又稱為playout或者rollout)，直到遊戲結束
            while ending == 'continue' and actions:
                action = np.random.choice(actions)
                index = actions.index(action)
                ending = env.step(actions[index])
                pos = env.n2p[action]
                #ending = env.step(action)
                if ending == 'clear':
                    score = env.player.hp
                if ending == 'continue':
                    done = False
                else:
                    done = True
                yield pos, done, episode + 1, score
                #actions = env.get_actions()      # 速度快上10倍
                actions = env.get_feasible_actions() # 速度較慢
            # 反向傳播：使用隨機遊戲的結果，更新從C到R的路徑上的節點資訊
            if ending == 'clear':
                score = env.player.hp
            else:
                score = 0
            mcts.backpropagate(score)
