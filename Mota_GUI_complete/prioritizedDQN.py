from import_tool import *
from environment_maze import Mota_Maze
from typing import List
#============================================================================
#  Prioritized Replay DeepQNetwork
#----------------------------------------------------------------------------
#  使用Mota_Maze環境學習  Ver.1  by Yaowang  2020.05.20
#============================================================================
 
#============================================================================
#  Q估計網路模型 
#============================================================================
class Eval_model(tf.keras.Model): 
    def __init__(self ,num_actions):
        super().__init__('Eval_network')       
        self.layer1 = tf.keras.layers.Dense(512, 
                        kernel_initializer = tf.random_normal_initializer(0, 0.1),
                        activation='relu') 
        self.layer2 = tf.keras.layers.Dense(256, 
                        kernel_initializer = tf.random_normal_initializer(0, 0.1),
                                            activation='relu') 
        self.layer3 = tf.keras.layers.Dense(64, 
                        kernel_initializer = tf.random_normal_initializer(0, 0.1),
                                            activation='relu')                                     
        self.logits = tf.keras.layers.Dense(num_actions, activation=None) 
    def call(self, inputs, training = False): 
        x = tf.convert_to_tensor(inputs) 
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)    
        layer3 = self.layer3(layer2)
        logits = self.logits(layer3)      
        return logits
#============================================================================
#  Q現實網路模型
#============================================================================    
class Target_model(tf.keras.Model): #Q現實  
    def __init__(self, num_actions):
        super().__init__('Target_network')
        self.layer1 = tf.keras.layers.Dense(512, 
                        kernel_initializer = tf.random_normal_initializer(0, 0.1),
                        activation='relu') 
        self.layer2 = tf.keras.layers.Dense(256, 
                        kernel_initializer = tf.random_normal_initializer(0, 0.1),
                                            activation='relu') 
        self.layer3 = tf.keras.layers.Dense(64, 
                        kernel_initializer = tf.random_normal_initializer(0, 0.1),
                                            activation='relu')         
        self.logits = tf.keras.layers.Dense(num_actions, activation=None)      
    def call(self, inputs):       
        x = tf.convert_to_tensor(inputs)     
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)    
        layer3 = self.layer3(layer2)
        logits = self.logits(layer3)         
        return logits
#============================================================================    
# SumTree & Memory 參考 莫凡Python https://morvanzhou.github.io/tutorials/   
#============================================================================
#  SumTree
#============================================================================       
class SumTree:
    data_pointer = 0

    def __init__(self, capacity:int):
        # 記憶庫容量
        self.capacity = capacity  
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object) 
#---------------------------------------------------------------------------- 
#  新增
#----------------------------------------------------------------------------
    def add(self, priority:float, data:List[int]):
        # 索引位置 由葉點插入
        tree_index = self.data_pointer + self.capacity - 1        
        self.data[self.data_pointer] = data                         
        #更新樹 
        self.update(tree_index, priority)                           
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:                  
            self.data_pointer = 0
#---------------------------------------------------------------------------- 
#  更新
#----------------------------------------------------------------------------
    def update(self, tree_index:int, priority:float):
        # 將優先級插入樹中 新的優先級進來就更新
        
        change = priority - self.tree[tree_index]          
        self.tree[tree_index] = priority
        
        while tree_index != 0:   
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
#---------------------------------------------------------------------------- 
#  到葉節點
#----------------------------------------------------------------------------            
    def get(self, v:float):
        # 父節點索引
        parent_index = 0
        while True:    
            leftchild_index = 2 * parent_index + 1        
            rightchild_index = (2 * parent_index + 1) + 1
            # 如果到葉點停止
            if leftchild_index >= len(self.tree):        
                leaf_index = parent_index
                break
            # 搜尋優先度高的節點
            else:       
                if v <= self.tree[leftchild_index]:
                    parent_index = leftchild_index
                else:
                    v -= self.tree[leftchild_index]
                    parent_index = rightchild_index
        # 取得優先度高的資料索引位置
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]
#---------------------------------------------------------------------------- 
#  所有優先級
#----------------------------------------------------------------------------
    # 裝飾器 私有
    @property
    def total_priority(self):
        # 把所有優先級加起來 回傳root
        return self.tree[0]  
    
#============================================================================
#  記憶庫
#============================================================================ 
class Memory:    
    def __init__(self, capacity:int):
        self.tree = SumTree(capacity)
        self.alpha = 0.4 # TD_error 轉成 優先級 
        self.beta = 0.6  # 重要性抽樣幅度
        self.beta_increment_per_sampling = 0.001
        self.TD_error_upper = 200. # 限制範圍
#---------------------------------------------------------------------------- 
#  存取記憶
#----------------------------------------------------------------------------
    def store(self, transition:List[float]):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.TD_error_upper
        
        self.tree.add(max_priority, transition)   
#---------------------------------------------------------------------------- 
#  抽樣
#----------------------------------------------------------------------------
    def sample(self, n:int):
        minibatch_index, minibatch_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        # 分割優先級區間
        priority_segment = self.tree.total_priority / n  
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  
        min_probability = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority  
        # ISWeights = (N * P(j))^-β / max(Weights) 化簡成 ISWeights = (pi / min[pi]^ -β)
        for i in range(n):
            # 再區間中隨機選取一個數 
            v = np.random.uniform((priority_segment * i), (priority_segment * (i + 1)))
            # 尋找到葉點的優先級
            index, priority, data = self.tree.get(v)
            # 樣本 pi
            probability = priority / self.tree.total_priority
            # ISWeights
            ISWeights[i, 0] = np.power(probability/min_probability, -self.beta)  
            minibatch_index[i], minibatch_memory[i, :] = index, data

        return minibatch_index, minibatch_memory, ISWeights
#---------------------------------------------------------------------------- 
#  更新樹
#----------------------------------------------------------------------------
    def batch_update(self, tree_index, TD_error):
        TD_error += 1  
        clipped_errors = np.minimum(TD_error, self.TD_error_upper)
        
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_index, ps):
            self.tree.update(ti, p)
        
#============================================================================
#  PrioritizedReplayDQN
#============================================================================    
   
class PrioritizedReplayDQN:
    def __init__(self,
                 actions= 5,  # 動作
                 features= 5,  # 觀測值數量
                 learning_rate = 0.01, # 學習率
                 reward_decay = 0.9, # 折扣因子
                 e_greedy = 0.8,# 貪婪率 
                 replace_updata_target =200,# target更新頻率 
                 memory_size = 2000, #記憶庫大小
                 batch_size = 32, #批次
                 e_greedy_increment = 1 #0.0000005
                 ):
        self.name ='PrioritizedReplayDQN'            
        self.actions = actions
        self.features = features

        self.lr = learning_rate
        self.gamma = reward_decay
        self.e_greedy = e_greedy
        self.updata_target = replace_updata_target
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_greedy_increment = e_greedy_increment
        #由 np.zeros() 更換成宣告 Memory 類別
        self.memory =  Memory(self.memory_size)
        self.learn_step_counter = 0
        self.epsilon = 0 if self.e_greedy_increment is not None else self.e_greedy    
        self.eval_model = Eval_model(self.actions)   
        self.target_model = Target_model(self.actions)      
        self.ISWeights = np.power(self.memory_size , -0.4) 
        self.eval_model.compile(optimizer=RMSprop(lr=self.lr),loss=self.loss)
#---------------------------------------------------------------------------- 
#  定義 Loss
#----------------------------------------------------------------------------       
    def loss(self, y_target, y_pred):
        return tf.reduce_mean(self.ISWeights * tf.math.squared_difference(y_target, y_pred))

#---------------------------------------------------------------------------- 
#  選擇動作
#----------------------------------------------------------------------------  
    def choose_action(self, observation:List[int]):
        # 轉成二維
        observation = observation[np.newaxis, :]  
        # 型態轉換int-> float
        observation = np.array(observation,dtype=np.float32)  
        if np.random.uniform() < self.epsilon: 
            actions_value = self.eval_model.predict(observation)
            print(actions_value)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.actions)
        return action
#---------------------------------------------------------------------------- 
#   儲存到記憶庫
#----------------------------------------------------------------------------       
    def store_transition(self , observation:List[int], action:int, reward:int, observation_:List[int]):
        # 建立索引位置
        if not hasattr(self, 'memory_counter'): 
            self.memory_counter = 0
        transition = np.hstack((observation, [action, reward], observation_))
        self.memory.store(transition)  
        self.memory_counter += 1
#---------------------------------------------------------------------------- 
#   學習
#----------------------------------------------------------------------------        
    def learn(self):
        #提取 樹 的 優先級 做 抽樣         
        tree_index, minibatch, self.ISWeights = self.memory.sample(self.batch_size)
        #樣本預測       
        Q_eval = self.eval_model.predict(minibatch[:,:self.features])
        Q_next = self.target_model.predict(minibatch[:,-self.features:])

        # 在相對應的動作進行反傳遞
        Q_target = Q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        eval_action_index = minibatch[:, self.features].astype(int)
        reward = minibatch[:, self.features+ 1]
        # Q_target = R+γ maxQ(Q_next)
        Q_target[batch_index, eval_action_index] = reward + self.gamma * np.max(Q_next, axis=1)
        # 取得誤差
        TD_error = np.sum(np.abs(Q_target - Q_eval) ,axis=1)

        # target_更新
        if self.learn_step_counter % self.updata_target == 0:
            for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
                target_layer.set_weights(eval_layer.get_weights())
        # 更新SumTree        
        self.memory.batch_update(tree_index, TD_error)

        # 減少探索 提高最優選擇
        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.e_greedy else self.e_greedy
        self.learn_step_counter += 1
#---------------------------------------------------------------------------- 
#   印出結果
#----------------------------------------------------------------------------         
    def print_score(self, score:List[int], episode:List[int], titel:str): 
        plt.title(titel)
        plt.plot()
        plt.plot(episode, score)
        plt.ylabel('score')
        plt.xlabel('episode')
        plt.show() 
    