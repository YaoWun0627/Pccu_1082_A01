from import_tool import *
from environment_node import Mota

#============================================================================
#   建立模型
#============================================================================
class MCTSfDm_Model(object):
    def __init__(self, env_name, route_mode):
        self.env_name = env_name
        self.route_mode = route_mode
        self.env = Mota()
        # self.Find_Env_List()
#============================================================================
#   資料預處理
#============================================================================
    def preprocess(self,df, labels_assigned=None, label=[]):
        # 型態轉為整數
        if labels_assigned is None:
            assigned = {}
            for col in label:
                c = df[col].astype('category')
                df[col] = c.cat.codes
                assigned[col] = {v: k for k, v in enumerate(c.cat.categories)}
            return df, assigned
        else:
            for col in label:
                assigned = labels_assigned[col]
                df[col] = df[col].map(lambda x: assigned[x]) #映射表若沒有對應將會出錯
            return df
#============================================================================
#   建立演示資料
#============================================================================
    def Find_Env_List(self):
        self.map_list = []
        if self.route_mode == '單一地圖':
            max_hp = 0
            save_file_name = None
            for file_name in os.listdir('player_route/'):
                map_name = file_name.split('_v')[0]
                if self.env_name == map_name:
                    map_name_hp = int(file_name.split('_h')[1].split('.')[0])
                    if map_name_hp > max_hp:
                        save_file_name = file_name
            if save_file_name != None:
                self.map_list.append(save_file_name)

        elif self.route_mode == '強化地圖':
            for file_name in os.listdir('player_route/'):
                map_name = file_name.split('_v')[0]
                if self.env_name == map_name:
                    self.map_list.append(file_name)

        elif self.route_mode == '強化泛化':
            exist_map = False
            for file_name in os.listdir('player_route/'):
                map_name = file_name.split('_v')[0]
                if self.env_name == map_name:
                    exist_map = True
                self.map_list.append(file_name)
            if not exist_map:
                return False
        print(self.map_list)
        if self.map_list:
            return True
        else:
            return False

#============================================================================
#   建立價值網路資料集
#============================================================================
    def Output_featureData(self):
        self.all_choose_index_list = []
        for load_map_name in self.map_list:
            self.all_choose_index_list.append(np.load('player_route/' + load_map_name))

        self.env.build_env(self.env_name)
        self.env.create_nodes()

        df_dict = {'p_hp':[],'p_atk':[],'p_def':[],'p_money':[],'p_exp':[],
           'p_yellowKey':[],'p_blueKey':[],'p_redKey':[],
           'class':[],'hp':[],'atk':[],'def':[],'money':[],'exp':[],
           'yellowKey':[],'blueKey':[],'redKey':[],'special':[],
           'z':[],'y':[],'x':[],'e':[],'choose':[]}

        for rate_, choose_index_list_ in enumerate(self.all_choose_index_list):
            if  rate_ > 0 and self.map_list[rate_].split('_v')[0] != self.map_list[rate_ - 1].split('_v')[0] or self.map_list[rate_].split('_v')[0] != self.env_name and rate_ == 0:
                del self.env
                self.env = Mota()
                self.env.build_env(self.map_list[rate_].split('_v')[0])
                self.env.create_nodes()
                print('Rebuild env')

            self.choose_index_list = choose_index_list_
            for index in self.choose_index_list:
                feasible_actions = self.env.get_feasible_actions()
                true_action = feasible_actions[index]
                actions = self.env.get_actions()
                # 狀態字典
                states = {}
                b = self.env.get_player_state()
                for i in range(len(actions)):
                    # 紀錄正確行動的索引值
                    if true_action == actions[i]:
                        true_index = i
                    self.env.step(actions[i])
                    a = self.env.get_player_state()
                    states[i] = a - b
                    self.env.back_step(1)
                # 添加數據
                for n in range(len(actions)):
                    df_dict['p_hp'].append(self.env.player.hp)
                    df_dict['p_atk'].append(self.env.player.atk)
                    df_dict['p_def'].append(self.env.player.def_)
                    df_dict['p_money'].append(self.env.player.money)
                    df_dict['p_exp'].append(self.env.player.exp)
                    df_dict['p_yellowKey'].append(self.env.player.items['yellowKey'])
                    df_dict['p_blueKey'].append(self.env.player.items['blueKey'])
                    df_dict['p_redKey'].append(self.env.player.items['redKey'])
                    df_dict['class'].append(actions[n].class_)
                    df_dict['hp'].append(states[n][0])
                    df_dict['atk'].append(states[n][1])
                    df_dict['def'].append(states[n][2])
                    df_dict['money'].append(states[n][3])
                    df_dict['exp'].append(states[n][4])
                    df_dict['yellowKey'].append(states[n][5])
                    df_dict['blueKey'].append(states[n][6])
                    df_dict['redKey'].append(states[n][7])
                    if actions[n].class_ == 'enemys':
                        df_dict['special'].append(actions[n].special)
                    else:
                        df_dict['special'].append(0)
                    pos = self.env.n2p[actions[n]]
                    df_dict['z'].append(pos[0])
                    df_dict['y'].append(pos[1])
                    df_dict['x'].append(pos[2])
                    if len(pos) == 3:
                        df_dict['e'].append(0)
                    else:
                        df_dict['e'].append(pos[3])

                    if n == true_index:
                        df_dict['choose'].append(1)
                    else:
                        df_dict['choose'].append(0)
                # 前進
                self.env.step(feasible_actions[index])
            self.env.reset()
            print('Env Reset')
        # 建立檔案
        df = pd.DataFrame(df_dict)
        return df
#============================================================================
#   訓練模型
#============================================================================
    def Modle_training(self):
        df = self.Output_featureData()
        self.delete_feature = ['z','y','x','e']
        df_train = df.drop(self.delete_feature, axis=1)
        df_train, self.labels_assigned = self.preprocess(df_train, label=['class'])
        Y_train = df_train.choose
        X_train = df_train.drop(['choose'], axis=1)
        self.model = RandomForestClassifier(n_estimators=500,
                               min_samples_split=2,
                               random_state=0).fit(X_train, Y_train)
        model_score = self.model.score(X_train, Y_train)
        print('訓練集準確率：  ', model_score)
        joblib.dump(self.model, 'model/MCTSfD_model.pkl')
        np.save('model/MCTSfD_labels.npy', self.labels_assigned)
        return model_score
#============================================================================
#   特徵工程
#============================================================================
    def feature_engineering(self, before_state, actions, assigned):
        feature_id = {'p_hp':0,'p_atk':1,'p_def':2,'p_money':3,'p_exp':4,
              'p_yellowKey':5,'p_blueKey':6,'p_redKey':7,
              'class':8,'hp':9,'atk':10,'def':11,'money':12,'exp':13,
              'yellowKey':14,'blueKey':15,'redKey':16,'special':17,
              'z':18,'y':19,'x':20,'e':21}
        delete_feature = [feature_id[i] for i in self.delete_feature]
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
        data = np.delete(data, delete_feature, axis=1)
        return data
#============================================================================
#   權重搜尋
#============================================================================
    def maximum_weight(self, array):
        max_num = np.max(array)
        max_index = np.where(array == max_num)[0]
        return max_num, max_index
#============================================================================
#   深度優先搜尋法
#============================================================================
    def DFS(self,best_index, actions, model, prediction_horizon = 2):
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
                weights = self.model.predict_proba(data)
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
#============================================================================
#   預測路線
#============================================================================
    def Predict_Route(self):
        # 重製環境
        del self.env
        self.env = Mota()
        self.env.build_env(self.env_name)
        self.env.create_nodes()
        print('Rebuild env')

        correct_num = 0
        average = 0
        startTime = time.perf_counter()
        # 遊戲開始
        self.all_choose_index_list = []
        self.map_list = []
        for file_name in os.listdir('player_route/'):
            map_name = file_name.split('_v')[0]
            if map_name == self.env_name:
                self.map_list.append(file_name)
                self.all_choose_index_list.append(np.load('player_route/' + file_name))
        for choose_index_list_ in self.all_choose_index_list:
            self.choose_index_list = choose_index_list_
            for choose_index in self.choose_index_list:
                actions = self.env.get_feasible_actions()
                # 選擇權重最高的行動
                player_state = self.env.get_player_state()
                data = self.feature_engineering(player_state, actions, self.labels_assigned)

                weights = self.model.predict_proba(data)
                _, best_index = self.maximum_weight(weights[:,1])
                if len(best_index) > 1:
                    # 預測行動之後的行動(預測視野)
                    best_index = self.DFS(best_index, actions, self.model)
                else:
                    best_index = best_index[0]
                # 比較結果
                if best_index == choose_index:
                    correct_num += 1
                self.env.step(actions[choose_index])
            self.env.reset()
            print('Reset Env')
            # 預測路線正確率
            average += correct_num / len(self.choose_index_list)
            print('預測路線正確率：', correct_num / len(self.choose_index_list), f'({correct_num}/{len(self.choose_index_list)})')
            correct_num = 0
        print('預測路線平均正確率：{:0.2f}'.format(average / len(self.all_choose_index_list)))
        print('預測花費時間：   {:0.4f}秒'.format(time.perf_counter() - startTime))
        print('(p.s.超過99%的時間都花費在model.predict上，而非feature_engineering)\n')
        return (average / len(self.all_choose_index_list)), (time.perf_counter() - startTime)
#============================================================================
#   主函式
#============================================================================
if __name__ == '__main__':
    model = MCTSfDm_Model('mapData_1', '強化地圖')
    model.Find_Env_List()
    model.Modle_training()
    model.Predict_Route()