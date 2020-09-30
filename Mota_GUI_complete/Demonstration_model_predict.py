from import_tool import *


class Demonstration_predict(object):
    def __init__(self, env, model_name):
        self.env = env
        self.model_name = model_name
        self.search_model_all()

    def search_model_all(self):
        networks = self.model_name
        try:
            if self.model_name == 'Multilayer Perceptron':
                model_file_name = f'model/{networks}_model.h5'
            else :
                model_file_name = f'model/{networks}_model.pkl'

        except NameError :
            return print("找不到模型，請先進行模型訓練!!!")

        # 載入模型
        self.values_model_file_name = os.path.splitext(model_file_name)
        if self.values_model_file_name[1] == '.h5':
            self.value_model = load_model(model_file_name)
        elif self.values_model_file_name[1] == '.pkl':
            self.value_model = joblib.load(model_file_name)

        self.value_labels_assigned = np.load('model/'+self.model_name +'_model_labels.npy', allow_pickle=True).item()

    def feature_engineering2(self,player_state, states, action, assigned):
        class_ = assigned['class'][action.class_]
        if action.class_ == 'enemys':
            special = action.special
        else:
            special = 0
        array = np.hstack((player_state,
                           class_, states[action], special))
        array = np.expand_dims(array, axis=0)
        return array

    def predict_environment(self):
        self.save_action = []
        index = 0
        start_time = time.perf_counter()
        self.env.reset(refresh_frame=True)
        ending = 'continue'
        while 'continue' in ending :
            actions = self.env.get_feasible_actions(re=True)
            if not actions:
                ending = 'no actions'
                endtime_time = time.perf_counter()
                return_time = endtime_time - start_time
                yield return_time, ending
                break
            # 每個行動的狀態差
            states = {}
            b = self.env.get_player_state()
            for action in actions:
                ending = self.env.step(action, return_reward=False)
                a = self.env.get_player_state()
                states[action] = a - b
                self.env.back_step(1)
            # 行動多次驗證(交叉驗證)
            best_actions_index = []
            for _ in range(1):
                best_index = 0
                best_weights = 0
                for index in range(len(actions)):
                    feature_row = self.feature_engineering2(self.env.get_player_state(), states,
                                                       actions[index], self.value_labels_assigned)
                    if self.values_model_file_name[1] == '.h5':
                        weights = self.value_model.predict(feature_row)[0][0]
                    elif self.values_model_file_name[1] == '.pkl':
                        weights = self.value_model.predict_proba(feature_row)[0][1]
                    if weights > best_weights:
                        best_weights = weights
                        best_index = index
                best_actions_index.append(best_index)

            label, count = np.unique(best_actions_index, return_counts=True)
            indexs, = np.where(count==np.max(count))
            index = np.random.choice(indexs)
            # 執行行動
            ending = self.env.step(actions[label[index]], refresh_frame=True)
            self.save_action.append(label[index])
            endtime_time = time.perf_counter()
            return_time = endtime_time - start_time
            yield return_time, ending
        # 儲存通關路線
        if ending == 'clear':
            for file_name in os.listdir('player_route/'):
                map_name = file_name.split('_v')[0]
                if map_name == self.env.env_name:
                    index += 1

            np.save(f'player_route/{self.env.env_name}_v{index}_h{self.env.player.hp}', self.save_action)


if __name__ == '__main__':
    pass