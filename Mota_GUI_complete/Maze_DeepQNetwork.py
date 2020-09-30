
from import_tool import *
from typing import List

from environment_maze import Mota_Maze

#============================================================================
#  DeepQNetwork
#----------------------------------------------------------------------------
#  使用Mota_Maze環境學習  Ver.1.3  by Yaowang  2020.05.09
#============================================================================
#============================================================================
# 參考
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5_Deep_Q_Network/DQN_modified.py
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
#  DeepQlearning
#============================================================================

class DeepQlearning:
    def __init__(self,
                 actions = 5,  # 動作
                 features = 5,  # 觀測值數量
                 learning_rate = 0.01, # 學習率
                 reward_decay = 0.9, # 折扣因子
                 e_greedy = 0.9,# 貪婪率
                 replace_updata_target =200,# target更新頻率
                 memory_size = 10000, #記憶庫大小
                 batch_size = 32, #批次
                 e_greedy_increment = 1.0
                 ):
        self.name ='DeepQlearning'
        self.actions = actions
        self.features = features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e_greedy = e_greedy
        self.updata_target = replace_updata_target
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_greedy_increment = e_greedy_increment
        self.memory =  np.zeros((self.memory_size, self.features * 2 + 2))# 記憶庫
        self.learn_step_counter = 0
        self.epsilon = 0 if self.e_greedy_increment is not None else self.e_greedy
        self.eval_model = Eval_model(self.actions)
        self.target_model = Target_model(self.actions)
        self.eval_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.lr), loss='mse')
        self.cost_his = []
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
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
#----------------------------------------------------------------------------
#   學習
#----------------------------------------------------------------------------
    def learn(self):
        #隨機抽樣
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, self.batch_size)
        else :
            sample_index = np.random.choice(self.memory_counter, self.batch_size)

        minibatch = self.memory[sample_index, :]
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
        # target_更新
        if self.learn_step_counter % self.updata_target == 0:
            for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
                target_layer.set_weights(eval_layer.get_weights())
        # 計算誤差
        self.cost = self.eval_model.train_on_batch(minibatch[:, :self.features], Q_target)
        #self.cost_his.append(self.cost)
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