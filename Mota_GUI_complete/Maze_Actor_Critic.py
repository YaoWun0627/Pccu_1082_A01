
from import_tool import *
from typing import List
from environment_maze import Mota_Maze

#============================================================================
#  ActorCritic 
#----------------------------------------------------------------------------
#  使用Mota_Maze環境學習  Ver.1  by Yaowang  2020.05.09
#============================================================================

class ActorCritic():
    def __init__(self, actions = 5, featrue = 5, lr_actor = 0.01, lr_critic = 0.1, gamma=0.99):
        self.name = 'ActorCritic'
        self.actions = actions
        self.featrue = featrue
        self.lr_actor = lr_actor # actor 學習率
        self.lr_critic = lr_critic # critic 學習率
        self.gamma = gamma 
        self.actor, self.critic ,self.policy = self.ActorCritic_network()
#----------------------------------------------------------------------------
#   ActorCritic-網路架構
#----------------------------------------------------------------------------                 
    def ActorCritic_network(self):
        inputs = tf.keras.layers.Input(shape=(self.featrue,))
        self.TD_error =tf.keras.layers.Input(shape=[1])
        layers1 = tf.keras.layers.Dense(512,activation='relu')(inputs)        
        layers2 = tf.keras.layers.Dense(256,activation='relu')(layers1)
        layers3 = tf.keras.layers.Dense(64,activation='relu')(layers2)        
        probs = tf.keras.layers.Dense(self.actions,activation='softmax')(layers3)  
        value = tf.keras.layers.Dense(1,activation='linear')(layers3)
        
        # 將 critic 值 傳給 actor 
        actor = tf.keras.Model([inputs, self.TD_error], [probs])
        actor.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.lr_actor), loss=self.actor_loss)      
        critic =  tf.keras.Model([inputs], [value])
        critic.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.lr_critic), loss='mean_squared_error')        
        policy =  tf.keras.Model([inputs], [probs])
        
        return actor, critic, policy
    
        # 定義 actor_loss
    def actor_loss(self,y_true, y_pred):
            # 0 ~ 1 限制在之間的範圍
            out = tf.keras.backend.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true * tf.keras.backend.log(out)
            # actor_loss = - TD_error * log π(s,a)
            return tf.keras.backend.sum(-self.TD_error * log_lik)
#---------------------------------------------------------------------------- 
#   Actor-選擇動作
#----------------------------------------------------------------------------     
    def choose_action(self, observation:List[int]):
        state = observation[np.newaxis, :] 
        probs = self.policy.predict(state)
        # 隨機選擇概率
        action = np.random.choice(np.arange(self.actions), p=probs.ravel())     
        return action 
#---------------------------------------------------------------------------- 
#   學習
#----------------------------------------------------------------------------     
    def learn (self, state:List[int], action:int, reward:int, next_state:List[int], done:bool):
        state = state[np.newaxis, :] 
        next_state= next_state[np.newaxis, :] 
        
        critic_value = self.critic.predict(state)
        critic_value_ = self.critic.predict(next_state)
        # 算法 TD_error = reward + gamma * S+1 - S
        target = reward + self.gamma * critic_value_ * int(done)
        TD_error = target - critic_value
        # 把選擇的動作記錄下來給 actor 訓練
        actions = np.zeros([1,self.actions])
        actions[np.arange(1),action] = 1.0
        
        self.actor.fit([state,TD_error], actions, verbose=0)
        self.critic.fit(state, TD_error, verbose=0)
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