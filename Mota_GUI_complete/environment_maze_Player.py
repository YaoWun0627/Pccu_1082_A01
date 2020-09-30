# -*- coding: utf-8 -*-
from import_tool import *

import database as db



#============================================================================
#  Player 玩家
#----------------------------------------------------------------------------
#  儲存玩家的狀態。
#============================================================================
class Player():
    #------------------------------------------------------------------------
    #  設定初始狀態
    #------------------------------------------------------------------------
    def reset(self, skw: dict):
        self.lv = skw['lv']                   # 等級
        self.maxhp = skw['maxhp']             # 最大生命
        self.hp = skw['hp']                   # 生命
        self.atk = skw['atk']                 # 攻擊力
        self.def_ = skw['def']                # 防禦力
        self.money = skw['money']             # 金幣
        self.exp = skw['exp']                 # 經驗值
        self.items = skw['items'].copy()      # 所持道具

#============================================================================
#  Mota_Maze 魔塔迷宮  V1.21      by Hung1   2020.5.2
#----------------------------------------------------------------------------
#  模擬魔塔環境，當給予動作時做出回饋。
#  基於Mota V1.1修改，將環境建立成迷宮。
#  self.observation = [z軸, y軸, x軸, 選項, 進入次數]
#  V1.0      2020.3.29
#  初版
#  V1.1      2020.4.16
#  添加self.features特徵值
#  可在常數中設定各結局的獎勵值，而非默認為0
#  step增加s_回傳值，每次都會複製一個新的對象
#  reset回傳的對象改成複製一個新的對象回傳
#  V1.2      2020.4.18
#  觀測值添加一維：進入次數
#  若step回傳invalid的訊息，觀測值將保留於上一個狀態，並且可繼續訓練
#  對其他無法繼續遊戲的狀態做限制，必須進行reset，讓使用者不會繼續輸入行動
#  step增加meaningless_actions_reward輸入參數，可以設定無意義行動的獎勵值，默認為0
#  修正step縮排問題，並且稍微改變其程式內容
#  刪除step和reset的return_state參數
#  V1.21     2020.5.2
#  step若不是continue，則self.observation包含回傳值一起彈回
#============================================================================
class Mota_Maze_Player():
    '''
    說明：
    本迷宮版魔塔環境固定輸入5個行動指令(右、下、左、上、確定)
    並且依照遊戲執行結果返回以下5種訊息其中一個：
    continue：遊戲還可以繼續輸入指令來進行下一步行動。
    death：在與怪物戰鬥中死亡。無法繼續遊戲。
    stop：在遊戲中進行事件卡關，比如NPC條件判斷、沒鑰匙開門、商店沒錢交易。無法繼續遊戲。
    clear：通關遊戲，進行成績結算。無法繼續遊戲。
    invalid：無效指令，比如撞牆、商店按左右鍵。"可以"繼續遊戲

    有幾個地方需特別注意，
    1.觀測值是落在當前處理的事件上，比如商店交易時，觀測值是商店的座標。
    2.商店可連續交易，新增離開的選項。
    3.人物在地圖移動中按確定鍵並不會返回invalid，其確定鍵功能為觸發當前觀測值的事件。(比如商店按離開選項後不移動，再按確定鍵會再進入商店)

    該類別可使用方法：
    build_env：輸入環境名稱，建立魔塔環境。
    step：輸入行動指令，返回獎勵值和訊息。
    reset：重設環境，返回初始觀測值。

    該類別變數的資料型態：
    observation：int一維陣列4元素，表示主角的四維座標(Z軸/Y軸/X軸/選項軸)。
    reward：int，由主角的8種狀態(生命/攻擊/防禦/金幣/經驗/黃鑰匙/藍鑰匙/紅鑰匙)計算出來的值。
    '''
    #------------------------------------------------------------------------
    #  常數
    #------------------------------------------------------------------------
    # 主角狀態的獎勵值比率(生命、攻擊、防禦、金幣、經驗、黃鑰匙、藍鑰匙、紅鑰匙)
    REWARD_RATE = np.array([1, 200, 200, 20, 20, 10, 10, 10])
    # 各類結局的獎勵值
    ENDDING_REWARD = {'death': -100, 'stop': -100, 'clear': 100, 'invalid': -200}
    #------------------------------------------------------------------------
    #  初始化
    #------------------------------------------------------------------------
    def __init__(self):
        self.env_data = None                  # 環境數據庫
        self.env_name = ''                    # 環境名稱
        self.env_map = None                   # 環境地圖
        self.features = 5                     # 特徵值
        self.observation = None               # 當前state的觀測值(4維座標+進入次數)
        self.map_enter_num = {}               # 地圖進入次數
        self.background_id = 0                # 背景圖塊ID
        self._original_env_map = None         # 初始地圖
        self._original_observation = None     # 初始觀測值
        self.player = Player()                # 儲存玩家狀態
        self.in_event_command = False         # 在事件選項中(如商店)
        self.event_commands = []              # 事件的選項列表
        self.new_enemys_id = {}               # 紀錄已替換的怪物事件ID
        self.limit_step_input = False         # 限制step輸入
        self.tile_path = 'pictures/baseTile.png'
        print('一個新的Mota_Maze物件被建立')
    #------------------------------------------------------------------------
    #  建立環境
    #------------------------------------------------------------------------
    def build_env(self, env_name: str):
        self.env_name = env_name
        self.env_data = db.load_data(env_name)  # 載入數據庫
        # 建立環境地圖
        self.env_map = np.array(self.env_data['floors']['map'])
        # 禁用(清除)指定事件位置
        for pos in self.env_data['floors']['disable']:
            self.change_map_id(pos, 0)
        # 建置玩家初始狀態
        self.player.reset(self.env_data['player'])
        # 建立初始觀測值
        player_id = None
        for key, value in self.env_data['maps'].items():
            if value['id'] == 'background':
                self.background_id = key
            if value['id'] == 'player':
                if player_id is None:  # 只取第一個出現player的ID
                    player_id = key
        player_pos = np.argwhere(self.env_map==player_id)[0]
        self.observation = np.append(player_pos, [0,0], axis=0)
        # 清除主角位置
        self.change_map_id(tuple(player_pos), 0)
        # 複製資料對象，用來快速重置
        self._original_env_map = self.env_map.copy()
        self._original_observation = self.observation.copy()
    #------------------------------------------------------------------------
    #  獲取主角狀態的陣列
    #------------------------------------------------------------------------
    def get_player_state(self) -> np.array:
        p = self.player
        return np.array([p.hp, p.atk, p.def_, p.money, p.exp,
                         p.items['yellowKey'], p.items['blueKey'], p.items['redKey']])
    #------------------------------------------------------------------------
    #  更改地圖編號
    #------------------------------------------------------------------------
    def change_map_id(self, pos: tuple, new_id: int):
        self.env_map[pos] = new_id
    #------------------------------------------------------------------------
    #  敵人事件處理
    #------------------------------------------------------------------------
    def activate_enemys(self, pos: tuple, event_id: int) -> str:
        # 戰鬥計算
        enemy = self.env_data['enemys'][event_id]
        p_damage = self.player.atk - enemy['def']
        if p_damage <= 0:
            ending = 'death'
            damage = 999999
        else:
            if enemy['hp'] % p_damage == 0:
                rounds = enemy['hp'] // p_damage - 1  # 主角自帶先攻一次
            else:
                rounds = enemy['hp'] // p_damage      # 主角自帶先攻一次
            e_damage = max(enemy['atk'] - self.player.def_, 0)
            damage = e_damage * rounds
            if enemy['special'] == 0:
                pass
            elif enemy['special'] == 11:  # 吸血屬性
                damage += self.player.hp // enemy['value']
            elif enemy['special'] == 22:  # 固傷屬性
                damage += enemy['damage']
            elif enemy['special'] == 1:   # 先攻屬性
                damage += e_damage
            elif enemy['special'] == 7:   # 破甲屬性
                damage += self.player.def_ * 0.9
            elif enemy['special'] == 4:   # 2連擊屬性
                damage *= 2
            elif enemy['special'] == 3:   # 堅固屬性
                if self.player.atk > enemy['def']:
                    new_def = self.player.atk - 1
                    p_damage = self.player.atk - new_def
                    if enemy['hp'] % p_damage == 0:
                        rounds = enemy['hp'] // p_damage - 1
                    else:
                        rounds = enemy['hp'] // p_damage
                    e_damage = max(enemy['atk'] - self.player.def_, 0)
                    damage = e_damage * rounds
                else:
                    pass
            elif enemy['special'] == 8:   # 反擊屬性
                damage += (self.player.atk * enemy['value']) * rounds
            # 計算結局
            ending = 'continue' if self.player.hp > damage else 'death'
        # 戰鬥處理
        self.player.hp -= damage
        if 'coin' in self.player.items:
            self.player.money += (enemy['money'] * 2)
        else:
            self.player.money += enemy['money']
        self.player.exp += enemy['exp']
        self.change_map_id(pos, self.background_id)
        return ending
    #------------------------------------------------------------------------
    #  道具事件處理
    #------------------------------------------------------------------------
    def activate_items(self, pos: tuple, event_id: int) -> str:
        # 道具處理
        item = self.env_data['items'][event_id]
        if item['cls'] == 'item':
            if event_id in self.player.items:
                self.player.items[event_id] += 1
            else:
                self.player.items[event_id] = 1
        elif item['cls'] == 'items':
            for key, value in item.items():
                if key in self.player.items:
                    self.player.items[key] += value
                elif key != 'cls':
                    self.player.items[key] = value
        else:
            for key, value in item.items():
                if key == 'hp':
                    self.player.hp += min(value, self.player.maxhp - self.player.hp)  # 溢血
                elif key == 'atk':
                    self.player.atk += value
                elif key == 'def':
                    self.player.def_ += value
                elif key == 'money':
                    self.player.money += value
                elif key == 'lv':
                    self.player.lv += value
                elif key == 'function':
                    for team in value:
                        val = eval(team['value'])
                        if team['name'] == 'player.hp':
                            # 使用function的hp不會受到溢血影響
                            self.player.hp = val
                        elif team['name'] == 'player.atk':
                            self.player.atk = val
                        elif team['name'] == 'player.def':
                            self.player.def_ = val
        self.change_map_id(pos, self.background_id)
        return 'continue'
    #------------------------------------------------------------------------
    #  地形事件處理
    #------------------------------------------------------------------------
    def activate_terrains(self, pos: tuple, event_id: int) -> str:
        ending = 'stop'
        if event_id == 'yellowDoor':
            if self.player.items['yellowKey'] == 0:
                ending = 'stop'
            else:
                self.player.items['yellowKey'] -= 1
                if self.player.items['yellowKey'] >= 0:
                    ending = 'continue'
                self.change_map_id(pos, self.background_id)
        elif event_id == 'blueDoor':
            if self.player.items['blueKey'] == 0:
                ending = 'stop'
            else:
                self.player.items['blueKey'] -= 1
                if self.player.items['blueKey'] >= 0:
                    ending = 'continue'
                self.change_map_id(pos, self.background_id)
        elif event_id == 'redDoor':
            if self.player.items['redKey'] == 0:
                ending = 'stop'
            else:
                self.player.items['redKey'] -= 1
                if self.player.items['redKey'] >= 0:
                    ending = 'continue'
                self.change_map_id(pos, self.background_id)

        return ending
    #------------------------------------------------------------------------
    #  NPC事件處理
    #  pos: 依照npc類別會傳入 三維座標 或 四維座標。
    #------------------------------------------------------------------------
    def activate_npcs(self, pos: tuple) -> str:
        if pos not in self.env_data['npcs']:
            commands = []
        else:
            commands = self.env_data['npcs'][pos]
        ending = 'continue'
        # 商店判斷
        if commands == 'shop':
            self.open_shop(pos)
            return ending
        # 指令處理
        for command in commands:
            # 條件句
            if command['type'] == 'if':
                if not eval(command['condition'].replace('player', 'self.player')):
                    ending = 'stop'
                    break
            # 增加數值
            elif command['type'] == 'addValue':
                if command['name'] == 'player.money':
                    self.player.money += command['value']
                elif command['name'] == 'player.hp':
                    self.player.hp += command['value']
                elif command['name'] == 'player.atk':
                    self.player.atk += command['value']
                elif command['name'] == 'player.def':
                    self.player.def_ += command['value']
                elif command['name'] == 'player.exp':
                    self.player.exp += command['value']
                elif command['name'] == 'player.lv':
                    self.player.lv += command['value']
            # 增加道具
            elif command['type'] == 'addItem':
                if command['name'] in self.player.items:
                    self.player.items[command['name']] += command['value']
                else:
                    self.player.items[command['name']] = command['value']
        # 不處理 not_activated 的指令
        # 改由判斷座標清除事件(非商店的所有三維事件)
        if len(pos) == 3:
            self.change_map_id(pos, self.background_id)
        return ending
    #------------------------------------------------------------------------
    #  事件後處理
    #------------------------------------------------------------------------
    def after_event(self, commands: list):
        for command in commands:
            # 破壞事件
            if command['type'] == 'open':
                self.change_map_id(command['loc'], self.background_id)
            # 啟用事件
            elif command['type'] == 'enable':
                z, y, x = command['loc']
                new_id = self.env_data['floors']['map'][z][y][x]
                self.change_map_id(command['loc'], new_id)
            # 更新怪物事件ID
            elif command['type'] == 'updateEnemys':
                enemys_lab = command.copy()
                enemys_lab.pop('type')
                reverse_dict = {v: k for k, v in self.new_enemys_id.items()}
                for key, value in enemys_lab.items():
                    if key in reverse_dict:
                        self.new_enemys_id[reverse_dict[key]] = value
                    else:
                        self.new_enemys_id[key] = value
    #------------------------------------------------------------------------
    #  觸發商店事件
    #------------------------------------------------------------------------
    def open_shop(self, pos: tuple):
        # 觀測值選項軸設為第一個
        self.observation[3] = 1
        # 建立商店
        self.in_event_command = True
        self.event_commands.clear()
        # 搜尋選項的座標(四維)
        i = 1
        while pos + (i,) in self.env_data['npcs']:
            self.event_commands.append(pos + (i,))
            i += 1
        self.event_commands.append('exit')
    #------------------------------------------------------------------------
    #  輸入行動，返回訊息
    #  action: 輸入0~4，右/下/左/上/確定
    #  meaningless_actions_reward : 無意義行動的獎勵值，默認為0
    #  返回獎勵值int和訊息str
    #  共有 continue, death, stop, clear, invalid 五種訊息
    #------------------------------------------------------------------------
    def step(self, action: int, return_state: bool = True,
             meaningless_actions_reward: int = 0) -> (int, str):
        ending = 'continue'
        # 禁止輸入行動
        if self.limit_step_input:
            raise RuntimeError('The mota environment need reset.')
        # 行動前的主角狀態
        before_state = self.get_player_state()
        # 紀錄行動前的觀測值
        old_observation = self.observation.copy()
        # 事件選項處理
        if self.in_event_command:
            if action == 1: # 下
                if self.observation[3] < len(self.event_commands):
                    self.observation[3] += 1
            elif action == 3: # 上
                if self.observation[3] > 1:
                    self.observation[3] -= 1
            elif action == 4: # 確定
                # 離開商店
                if self.event_commands[self.observation[3] - 1] == 'exit':
                    self.observation[3] = 0
                    self.in_event_command = False
                # 進行交易
                else:
                    ending = self.activate_npcs(tuple(self.observation[:4]))
                    self.in_event_command = False
            else:
                ending = 'invalid'
            # 座標(取四軸)
            pos = tuple(self.observation[:4])
        # 地圖行動處理
        else:
            if action == 0: # 右
                self.observation[2] += 1
                if self.observation[2] >= self.env_data['floors']['width']:
                    self.observation[2] -= 1
                    ending = 'invalid'
            elif action == 1: # 下
                self.observation[1] += 1
                if self.observation[1] >= self.env_data['floors']['height']:
                    self.observation[1] -= 1
                    ending = 'invalid'
            elif action == 2: # 左
                self.observation[2] -= 1
                if self.observation[2] < 0:
                    self.observation[2] += 1
                    ending = 'invalid'
            elif action == 3: # 上
                self.observation[1] -= 1
                if self.observation[1] < 0:
                    self.observation[1] += 1
                    ending = 'invalid'
            elif action == 4: # 確定
                pass  # 原地觸發事件(如商店等)
            else:
                ending = 'invalid'
            # 座標(只取三軸)
            pos = tuple(self.observation[:3])
            event = self.env_data['maps'][self.env_map[pos]]
            # 撞牆判定
            if event['cls'] == 'except' and event['id'] != 'background':
                ending = 'invalid'
            # 觸發事件
            elif event['cls'] == 'enemys':
                # 判斷怪物數據是否已更換
                if event['id'] in self.new_enemys_id:
                    ending = self.activate_enemys(pos, self.new_enemys_id[event['id']])
                else:
                    ending = self.activate_enemys(pos, event['id'])
            elif event['cls'] == 'items':
                ending = self.activate_items(pos, event['id'])
            elif event['cls'] == 'terrains':
                if event.get('noPass', True):
                    ending = self.activate_terrains(pos, event['id'])
            elif event['cls'] == 'npcs':
                ending = self.activate_npcs(pos)
            elif event['cls'] == 'flag':
                ending = 'clear'
            # 事件後處理(全局處理)
            if pos in self.env_data['floors']['afterEvent']:
                commands = self.env_data['floors']['afterEvent'][pos]
                self.after_event(commands)
            # 傳送點處理
            if pos in self.env_data['floors']['changeFloor']:
                new_pos = self.env_data['floors']['changeFloor'][pos]
                self.observation = np.append(new_pos, [0, 0], axis=0)
        # 計算獎勵值
        if ending in Mota_Maze_Player.ENDDING_REWARD:
            reward = Mota_Maze_Player.ENDDING_REWARD[ending]
        else:
            # 行動後的主角狀態
            after_state = self.get_player_state()
            reward = np.sum((after_state - before_state) * Mota_Maze_Player.REWARD_RATE)
        # 無意義行動獎勵值
        if reward == 0:
            reward = meaningless_actions_reward
        # 限制行動輸入
        if ending != 'invalid' and ending != 'continue' and ending != 'stop':
            self.limit_step_input = True
        # 觀測值進入次數
        self.observation[4] = self.map_enter_num.get(pos, 0)

        if ending == 'continue':
            # 計算進入次數
            if pos in self.map_enter_num:
                self.map_enter_num[pos] += 1
            else:
                self.map_enter_num[pos] = 1
            # 回傳值
            observation = self.observation.copy()
            return observation, reward, ending
        else:
            # 觀測值退回，回傳觀測值
            self.observation = old_observation
            return self.observation.copy(), reward, ending
    #------------------------------------------------------------------------
    #  重設環境
    #------------------------------------------------------------------------
    def reset(self) -> np.array:
        # 環境地圖
        self.env_map = self._original_env_map.copy()
        # 玩家狀態
        self.player.reset(self.env_data['player'])
        # 替換事件ID
        self.new_enemys_id.clear()
        # 觀測值
        self.observation = self._original_observation.copy()
        # 限制step輸入
        self.limit_step_input = False
        # 地圖進入次數
        self.map_enter_num.clear()
        return self.observation.copy()
    #------------------------------------------------------------------------
    #  獲取行動的對象訊息
    #  (本方法只供測試使用)
    #------------------------------------------------------------------------
    def _get_actions_target(self) -> pd.DataFrame:
        data = []
        if self.in_event_command:
            data.append(['右', '----'])
            data.append(['下', '選項下移'])
            data.append(['左', '----'])
            data.append(['上', '選項上移'])
            data.append(['確定', '執行選項'])
            # 建立表格
            return pd.DataFrame(data, columns=['cmd','message'])
        else:
            pos = tuple(self.observation + [0,0,1,0,0])[:3]
            if self.observation[2] < self.env_data['floors']['width']:
                event = self.env_data['maps'][self.env_map[pos]]
                data.append(['右', event['cls'], event['id']])
            else:
                data.append(['右', '----', '----'])
            pos = tuple(self.observation + [0,1,0,0,0])[:3]
            if self.observation[1] < self.env_data['floors']['height']:
                event = self.env_data['maps'][self.env_map[pos]]
                data.append(['下', event['cls'], event['id']])
            else:
                data.append(['下', '----', '----'])
            pos = tuple(self.observation - [0,0,1,0,0])[:3]
            if self.observation[2] >= 0:
                event = self.env_data['maps'][self.env_map[pos]]
                data.append(['左', event['cls'], event['id']])
            else:
                data.append(['左', '----', '----'])
            pos = tuple(self.observation - [0,1,0,0,0])[:3]
            if self.observation[2] >= 0:
                event = self.env_data['maps'][self.env_map[pos]]
                data.append(['上', event['cls'], event['id']])
            else:
                data.append(['上', '----', '----'])
            event = self.env_data['maps'][self.env_map[tuple(self.observation[:3])]]
            data.append(['確定', event['cls'], event['id']])
            # 建立表格
            return pd.DataFrame(data, columns=['cmd','class','id'])

    def draw_map(self, picture_name: str) -> int:
        layer = self.env_data['floors']['layer']
        width = self.env_data['floors']['width']
        height = self.env_data['floors']['height']
        map_ = self.env_data['floors']['map']
        maps = self.env_data['maps']
        tile = self.env_data['icons']
        tileImage = Image.open(self.tile_path)
        # 地板平鋪式背景
        bg = Image.new('RGBA', (width*32, height*32))
        bf = tileImage.crop((0, 0, 32, 32))
        for i in range(0, width*32, 32):
            for j in range(0, height*32, 32):
                bg.paste(bf, (i, j))
        # 依序畫出每一樓層
        for f in range(0, layer):
            fin = bg.copy()
            for i in range(0, width):
                for j in range(0, height):
                    # 獲取圖塊
                    n = map_[f][j][i]
                    if n == 0:
                        continue
                    if maps[n]['id'] == 'player':
                        n = 0
                    tile_id = tile[maps[n]['id']]
                    local = (tile_id%16*32, tile_id//16*32, tile_id%16*32+32, tile_id//16*32+32)
                    et = tileImage.crop(local)
                    # 分離通道並進行透明度拼貼
                    r,g,b,a = et.split()
                    fin.paste(et, (i*32, j*32), mask = a)
            fin.save(picture_name + str(f) + '.png')
        return layer

    def draw_map_layer(self, layer):
        #layer = self.env_data['floors']['layer']
        width = self.env_data['floors']['width']
        height = self.env_data['floors']['height']
        map_ = self.env_map
        maps = self.env_data['maps']
        tile = self.env_data['icons']
        tileImage = Image.open(self.tile_path)
        # 地板平鋪式背景
        bg = Image.new('RGBA', (width*32, height*32))
        bf = tileImage.crop((0, 0, 32, 32))
        for i in range(0, width*32, 32):
            for j in range(0, height*32, 32):
                bg.paste(bf, (i, j))
        # 依序畫出每一樓層
        fin = bg.copy()
        for i in range(0, width):
            for j in range(0, height):
                # 獲取圖塊
                n = map_[layer][j][i]
                if n == 0:
                    continue
                if maps[n]['id'] == 'player':
                    n = 0
                tile_id = tile[maps[n]['id']]
                local = (tile_id%16*32, tile_id//16*32, tile_id%16*32+32, tile_id//16*32+32)
                et = tileImage.crop(local)
                # 分離通道並進行透明度拼貼
                r,g,b,a = et.split()
                fin.paste(et, (i*32, j*32), mask = a)
        fin.save('pictures/MT_{:d}.png'.format(layer))
#測試
if __name__ == '__main__':
    print('★ =========歡迎使用文字版(迷宮版)魔塔環境========= ★')
    print('提醒：在操作時本遊戲時，建議以圖片版魔塔進行對照遊玩～')
    # 前置處理
    mota = Mota_Maze_Player()
    #mota.build_env('24層魔塔')
    mota.build_env('standard_map')
    print('環境名稱：', mota.env_name)
    index = ['hp','atk','def','money','exp','yellowKey','blueKey','redKey']
    playerDf = pd.DataFrame([0]*len(index), index=index, columns=['value'])
    step_count = 0
    # 預載開始(24層魔塔)
    #choose_index_list = [
    #3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 0, 3, 3, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3,
    #2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 0, 0, 0, 3, 3,
    #2, 2, 2, 2, 3, 3, 3, 0, 3, 3, 2, 3, 3, 1, 0, 0, 3, 1, 1, 2, 1, 1, 0, 1, 0,
    #0, 0, 0, 1, 0, 1, 1, 2, 1, 1, 0, 0, 3, 1, 2, 2, 3, 3, 0, 3, 3, 2, 3, 3, 3,
    #2, 3, 3, 3, 1, 4, 1, 1, 4, 1]
    # 預載開始(mapData_3)
    #choose_index_list = [
    #1, 1, 1, 2, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 0,
    #1, 1, 1, 1, 1, 0, 0, 0, 3, 1, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 3, 3, 0, 0, 0,
    #0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 3,
    #3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 2, 3, 3, 3, 2, 2,
    #2, 2, 1, 1, 1, 2, 0, 0, 2, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 0, 0, 1, 1, 1, 1,
    #1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 3, 3,
    #2, 0, 0, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,
    #3, 3, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 2, 3, 2, 2, 1, 1, 0, 2, 3,
    #3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]#,3]
    #for index in choose_index_list:
    #    mota.step(index)
    # 重置測試
    #mota.reset()
    #for index in choose_index_list:
    #    mota.step(index)
    #step_count = len(choose_index_list)
    # 預載結束

    choose_index_list = []
    p = mota.player
    ending = 'continue'
    while ending == 'continue' or ending == 'invalid' :
        print('主角狀態：')
        playerDf.value = [p.hp, p.atk, p.def_, p.money, p.exp,
                          p.items['yellowKey'], p.items['blueKey'], p.items['redKey']]
        print(playerDf.T)
        print('已執行步數：', step_count)
        print('觀測值(座標)：', mota.observation)
        print('行動選項資訊：')
        print(mota._get_actions_target())
        index = int(input('請輸入行動index(-1結束)：'))

        if index == -1:
            break
        else:
            choose_index_list.append(index)
            _, reward, ending = mota.step(index)
            step_count += 1
            print('該次行動獎勵值：', reward)

        print('-------------------------------------------------')
        print(mota.env_map)
    print('★ ==================遊戲結束================== ★')
    print('你的結局是：', ending)
    print('已行動選項順序：', choose_index_list)
