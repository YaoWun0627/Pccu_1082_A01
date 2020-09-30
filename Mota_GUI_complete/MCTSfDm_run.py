from import_tool import *
from MCTSfDm import MCTSfDm_Algorithm
from QLfD import imitation_form_QLearning
from lzw import LZW

class MCTSfDm_learning(object):
    def __init__(self, env, algorithm_name, rounds):
        self.env = env
        self.algorithm_name = algorithm_name
        self.rounds = rounds
        self.model_switch = True

    def Learning_Algorithm(self):
        self.agent = None
        if self.algorithm_name =='MCTSfD':
            self.agent = MCTSfDm_Algorithm(self.env.get_feasible_actions(re=True),self.env)
            for  pos, done, episode, score, save_action in self.MCTSfDm_run(self.rounds):
                yield pos, done, episode, score, save_action

        if self.algorithm_name =='QLfD':
            self.agent = imitation_form_QLearning(self.env)
            for pos, done, episode, score, save_action in self.QLfD_run(self.rounds):
                yield pos, done, episode, score, save_action

    def on_model_animation(self):
        self.model_switch = True
    def off_model_animation(self):
        self.model_switch = False

    def QLfD_run(self, rounds):
        print('Running QLfD')
        lzw = LZW()
        state = lzw.compress(self.env.observation)
        actions = self.env.get_feasible_actions()
        self.agent.create_state_qtable(state, actions, [0] * len(actions))
        score = 0
        for episode in range(rounds):
            self.env.reset(refresh_frame=True)
            save_action = []
            state = lzw.compress(self.env.observation)
            while True:
                actions = self.env.get_feasible_actions()
                action  = self.agent.choose_action(state)
                pos = self.env.n2p[action]
                best_action, weights = self.agent.predict(actions)
                index = actions.index(action)
                # 儲存遊玩路徑
                save_action.append(index)
                ending, _ = self.env.step(actions[index], return_reward = True, refresh_frame=self.model_switch)
                state_ = lzw.compress(self.env.observation)
                if ending == 'clear':
                    score = self.env.player.hp
                if ending == 'continue':
                    done = False
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
                yield pos, done, episode, score, save_action
                self.agent.learn(state, action, best_action, state_, done)
                state = state_

                if done:
                    break

    def MCTSfDm_run(self, rounds):
        highest_score = 0
        highest_round = 0
        score = 0
        save_action = []
        startTime = time.perf_counter()
        for episode in range(1, rounds+1):
            ending = 'continue'
            actions_path = self.agent.select()
            for action in actions_path:
                ending = self.env.step(action, return_reward=False, refresh_frame=self.model_switch)
            actions = self.env.get_feasible_actions(re=True)
            if ending == 'continue' and actions:
                expand_action, expand_index = self.agent.choose_expansion_node()
                ending = self.env.step(expand_action, return_reward=False,refresh_frame=self.model_switch)
                actions = self.env.get_feasible_actions(re=True)
                self.agent.expand(expand_index, actions)

            while ending == 'continue' and actions:
                action = self.agent.predict(actions)
                pos = self.env.n2p[action]
                ending = self.env.step(action, refresh_frame=self.model_switch)
                actions = self.env.get_feasible_actions(re=True)

                if ending == 'clear':
                    score = self.env.player.hp
                    # 儲存通關路徑
                    save_observation = self.env.observation.copy()
                    save_action_ = []
                    self.env.reset()
                    for action in save_observation[len(self.env.observation):]:
                        save_action_.append(self.env.get_feasible_actions().index(action))
                        self.env.step(action)
                    save_action = save_action_
                if ending == 'continue':
                    if actions:
                        done = False
                    else:
                        done = True
                else:
                    done = True
                yield pos, done, episode-1, score, save_action
            print('##################################')
            print('##################################')
            print(done)
            print('##################################')
            print('##################################')
            if ending == 'clear':
                score = self.env.player.hp
                yield pos, done, episode-1, score, save_action
            else:
                score = 0
                yield pos, done, episode-1, score, save_action

            self.agent.backpropagate(score)
            self.env.reset(refresh_frame=True)
            # 反饋本次成績
            if highest_score < score:
                highest_score = score
                highest_round = episode

            print('round: %d    score: %d    max_score: %d    max_round: %d    time: %0.4f' %
                  (episode, score, highest_score, highest_round, time.perf_counter() - startTime))
            if episode % 100 == 0:
                print([ [self.env.n2p[n] for n in self.env.observation]])



if __name__ == '__main__':
    pass



