from import_tool import *
import copy
import os
import tkinter as tk
import numpy as np
import database as db
from ToolTip import Tooltip
from tkinter import scrolledtext, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from showmap import Frame_Scrollbar
from environment_maze_Player import Mota_Maze_Player
from environment_node_Player import Mota_Player
from environment_node import Mota
from window_resize import win_resize
from String_ import *

global_image = None
radioValue = None

class window_Player(tk.Toplevel):
    def __init__(self):
        super(window_Player, self).__init__()
        # 設定視窗參數
        self.title(Player_window_string['Initialize']['title'])
        winWidth = 806
        winHeight = 606
        self.x = (self.winfo_screenwidth() - winWidth) // 2
        self.y = (self.winfo_screenheight() - winHeight) // 3
        self.geometry('%sx%s+%s+%s' % (winWidth, winHeight, self.x, self.y))
        self.minsize(winWidth, winHeight)
        win_resize(self)
        # 中文字體
        self.ChiFonts = font.Font(family=Player_window_string['Initialize']['font_style_Chinese'], weight=tkFont.BOLD)
        # 遊戲開始視窗
        self.start_window()

    def start_window(self):
        # 遊戲開始按鈕
        S_11 = tk.PhotoImage(file ='pictures/S_07.png')
        self.S_11 = S_11.subsample(14,14)
        self.start_window_btn = tk.Button(self)
        self.start_window_btn.config(text = Player_window_string['Button_text']['text_1'], command=self.choose_environment, image = self.S_11,compound = tk.LEFT, bg ='#C4A98B')
        self.start_window_btn.place(y=-350, rely=1, relx=0.42, width=150)
        self.start_window_btn['font'] = self.ChiFonts

    def choose_environment(self):
        # 隱藏遊戲開始按鈕
        self.start_window_btn.place_forget()
        # 選擇節點版魔塔模式
        S_01 = tk.PhotoImage(file ='pictures/S_01.png')
        self.S_01 = S_01.subsample(14,14)
        self.node_mode_btn = tk.Button(self)
        self.node_mode_btn.config(text = Player_window_string['Button_text']['text_2'], command=self.node_window,image =self.S_01 ,compound = tk.LEFT,bg ='#C4A98B')
        self.node_mode_btn.place(x=-56, y=-300, rely=1, relx=0.43, width=200)
        self.node_mode_btn['font'] = self.ChiFonts
        Tooltip(self.node_mode_btn, text = Show_learning_string['tips_text']['text_17'])
        # 選擇迷宮版魔塔模式
        S_02 = tk.PhotoImage(file ='pictures/S_02.png')
        self.S_02 = S_02.subsample(14,14)
        self.maze_mode_btn = tk.Button(self)
        self.maze_mode_btn.config(text=Player_window_string['Button_text']['text_3'], command = self.maze_window,image = self.S_02,compound = tk.LEFT,bg ='#C4A98B' )
        self.maze_mode_btn.place(x=-56, y=-370, rely=1, relx=0.43, width=200)
        self.maze_mode_btn['font'] = self.ChiFonts
        Tooltip(self.maze_mode_btn, text = Show_learning_string['tips_text']['text_18'])

    def base_window(self):
        # 建立地圖畫布
        self.frm = tk.LabelFrame(self, text='Map')
        self.frm.place(height=-100, relwidth=0.6, relheight=1.0)
        self.canvas = tk.Canvas(self.frm, height=32*15, width=32*15)
        # 下樓按鈕
        self.down_btn = ttk.Button(self, text= Player_window_string['Button_text']['text_4'],command=self.floor_down)
        self.down_btn.place(y=-35, rely=1.0, relx=0.0, relwidth=0.2)
        self.down_btn.config(state='disabled')
        # 上樓按鈕
        self.up_btn = ttk.Button(self, text= Player_window_string['Button_text']['text_5'], command=self.floor_up)
        self.up_btn.place(y=-35, rely=1.0, relx=0.4, relwidth=0.2)
        self.up_btn.config(state='disabled')
        # 選擇地圖按鈕
        Q_02 = tk.PhotoImage(file ='pictures/Q_02.png')
        self.Q_02 = Q_02.subsample(6,6)
        self.choose_map_btn = tk.Button(self)
        self.choose_map_btn.config(text=Player_window_string['Button_text']['text_6'], command = self.choose_map, image = self.Q_02, bg ='#C4A98B', compound = tk.LEFT)
        self.choose_map_btn.place(x=40, y=80, width=-80, height=30, relx=0.6, relwidth=0.4)
        self.choose_map_btn['font'] = self.ChiFonts
        # 顯示資訊欄位
        self.info_frame = tk.LabelFrame(self, text=Player_window_string['Button_text']['text_7'])
        self.info_frame.place(x=10, y=160, width=-20, height=-170, relx=0.6, relwidth=0.4, relheight=1.0)
        self.info_scr = scrolledtext.ScrolledText(self.info_frame)
        self.info_scr.pack(fill='both', expand=True)
        if self.mode == 'maze':
            self.info_scr.insert(tk.END, Player_window_string['info_scr']['text_1'])
        elif self.mode == 'node':
            self.info_scr.insert(tk.END, Player_window_string['info_scr']['text_2'])
        # 主角狀態欄位
        self.player_state_columns = ("LV", "HP","ATK", "DEF", "MON", "EXP", "YK", "BK", "RK")
        self.player_state_tree = ttk.Treeview(self, height=1)
        self.player_state_tree['columns'] = self.player_state_columns
        self.player_state_tree['show'] = "headings"
        for col in self.player_state_columns:
            self.player_state_tree.column(col, width=(32*15)//9, minwidth=(32*15)//9, anchor='center', stretch=False)
            self.player_state_tree.heading(col, text=col, anchor='center')
        self.player_state_tree.place(x=0, y=-90, rely=1, relx=0, relwidth=0.6)
        # 建立遊戲資訊欄位
        self.game_info_nb = ttk.Notebook(self)
        self.game_info_tab = Frame_Scrollbar(self.game_info_nb)
        self.game_info_nb.add(self.game_info_tab, text=Player_window_string['Button_text']['text_9'])
        # 判斷是否開啟遊戲資訊欄位
        self.open_game_info = True
        # 建立遊戲資訊欄位按鈕
        self.game_info_btn = ttk.Button(self, text=Player_window_string['Button_text']['text_9'], command=self.game_info)
        self.game_info_btn.config(state="disabled")
        self.game_info_btn.place(y=-35, rely=1.0, relx=0.2, relwidth=0.2)
        # 建立遊戲資訊欄位滾動條
        self.enemys_info_scrollbar = ttk.Scrollbar(self.game_info_tab.contents, orient=tk.HORIZONTAL)
        self.items_info_scrollbar = ttk.Scrollbar(self.game_info_tab.contents, orient=tk.HORIZONTAL)
        # 建立遊戲資訊表格
        self.enemys_info_tree = ttk.Treeview(self.game_info_tab.contents)
        self.enemys_info_tree['show'] = "headings"
        self.items_info_tree = ttk.Treeview(self.game_info_tab.contents)
        self.items_info_tree['show'] = "headings"

    def floor_down(self):
        self.floor_index -= 1
        global global_image
        self.canvas.delete(self.image)
        global_image = tk.PhotoImage(file='pictures/MT_{:d}.png'.format(self.floor_index))
        self.image = self.canvas.create_image(0, 0, anchor='nw', image = global_image)
        self.change_floor_status()

    def floor_up(self):
        self.floor_index += 1
        global global_image
        self.canvas.delete(self.image)
        global_image = tk.PhotoImage(file='pictures/MT_{:d}.png'.format(self.floor_index))
        self.image = self.canvas.create_image(0, 0, anchor='nw', image = global_image)
        self.change_floor_status()

    def game_info(self):
        # 敵人和物品所需要的資訊
        enemys_columns = ["id", "hp", "atk", "def", "money", "exp", "special", "damage", "value"]
        items_columns = ["id", "hp", "atk", "def", "money", "exp", "yellowKey", "blueKey", "redKey", "function"]
        # 取得地圖上的資訊
        map_ = self.env.env_data['floors']['map'][self.floor_index]
        maps = self.env.env_data['maps']
        enemys_target = self.env.env_data['enemys']
        items_target = self.env.env_data['items']
        show_num = {i for l in map_ for i in l}
        # 找出這層樓的敵人id
        show_enemys_num = sorted([k for k in show_num if maps[k]['cls'] == 'enemys'])
        show_enemys_list = [{**maps[k], **enemys_target[maps[k]['id']]} for k in show_enemys_num]
        # 找出這層樓的物品id
        show_items_num = sorted([k for k in show_num if maps[k]['cls'] == 'items'])
        show_items_list = [{**maps[k], **items_target[maps[k]['id']]} for k in show_items_num]
        # 將敵人資訊取出並放到list中
        enemys_data = []
        items_data = []
        for d in show_enemys_list:
            row = []
            for c in enemys_columns:
                row.append(str(d.get(c, '-')))
            enemys_data.append(row)
        # 將物品資訊取出並放到list中
        for d in show_items_list:
            row = []
            for c in items_columns:
                row.append(str(d.get(c, '-')))
            items_data.append(row)
        # 設置敵人表格行名稱和限制高度
        self.enemys_info_tree['columns'] = tuple(enemys_columns)
        self.enemys_info_tree['height'] = len(enemys_data)
        # 設置物品表格行名稱和限制高度
        self.items_info_tree['columns'] = tuple(items_columns)
        self.items_info_tree['height'] = len(items_data)
        # 設置敵人和物品表格
        for col in tuple(enemys_columns):
            self.enemys_info_tree.column(col, width=100, minwidth=100, anchor='center', stretch=False)
            self.enemys_info_tree.heading(col, text=col, anchor='center')
        for col in tuple(items_columns):
            self.items_info_tree.column(col, width=100, minwidth=100, anchor='center', stretch=False)
            self.items_info_tree.heading(col, text=col, anchor='center')
        # 寫入和更新敵人和物品資訊
        for item in self.enemys_info_tree.get_children():
            self.enemys_info_tree.delete(item)
        for item in self.items_info_tree.get_children():
            self.items_info_tree.delete(item)
        for enemys_value in enemys_data:
            self.enemys_info_tree.insert('', 'end', value=enemys_value)
        for items_value in items_data:
            self.items_info_tree.insert('', 'end', value=items_value)
        # 設置敵人和物品的滾動條
        self.enemys_info_tree.configure(xscrollcommand=self.enemys_info_scrollbar.set)
        self.enemys_info_scrollbar.configure(command=self.enemys_info_tree.xview)
        self.items_info_tree.configure(xscrollcommand=self.items_info_scrollbar.set)
        self.items_info_scrollbar.configure(command=self.items_info_tree.xview)
        # 開啟遊戲資訊
        if self.open_game_info:
            self.game_info_nb.place(height=-40, relwidth=0.6, relheight=1.0)
            self.enemys_info_tree.pack(fill="both")
            self.enemys_info_scrollbar.pack(fill='x')
            self.items_info_tree.pack(fill="both")
            self.items_info_scrollbar.pack(fill='x')
            self.game_info_btn.config(text = Player_window_string['Button_text']['text_10'])
            self.up_btn.config(state="disabled")
            self.down_btn.config(state="disabled")
        # 關閉遊戲資訊
        else:
            self.game_info_nb.place_forget()
            self.enemys_info_tree.pack_forget()
            self.enemys_info_scrollbar.pack_forget()
            self.items_info_tree.pack_forget()
            self.items_info_scrollbar.pack_forget()
            self.game_info_btn.config(text=Player_window_string['Button_text']['text_9'])
            self.change_floor_status()
        # 開關遊戲資訊
        self.open_game_info = not self.open_game_info

    def change_floor_status(self):
        down_string = Player_window_string['Button_text']['text_11']
        up_string = Player_window_string['Button_text']['text_12']
        if self.floor_index > 0:
            # 遊戲進行中
            if self.play_state:
                # 遊戲進行中關閉按鈕
                self.down_btn.config(text=f'{down_string}({self.floor_index - 1}F)', state='disabled')
            else:
                self.down_btn.config(text=f'{down_string}({self.floor_index - 1}F)', state='normal')
        else:
            self.down_btn.config(text= down_string, state='disabled')
        if self.floor_max - self.floor_index > 1:
            # 遊戲進行中
            if self.play_state:
                # 遊戲進行中關閉按鈕
                self.up_btn.config(text=f'{up_string}({self.floor_index + 1}F)', state='disabled')
            else:
                self.up_btn.config(text=f'{up_string}({self.floor_index + 1}F)', state='normal')
        else:
            self.up_btn.config(text = up_string, state='disabled')

    def choose_map(self):
        # 隱藏選擇地圖按鈕
        self.choose_map_btn.place_forget()
        # 選擇地圖標籤
        self.choose_map_lab = tk.Label(self, text = Player_window_string['Button_text']['text_6'],bg='#C4A98B', font=(Player_window_string['Initialize']['font_style_Chinese'], 12, tkFont.BOLD))
        self.choose_map_lab.place(x=20, y=65, width=-80, height=30, relx=0.7, relwidth=0.3)
        # 取得地圖名稱
        self.choose_map_btn.config(text = Player_window_string['Button_text']['text_13'], command=self.get_map_name_and_build_env)
        self.choose_map_btn.place(x=500, y=100, width=-30, relwidth=0.15)
        # 地圖的下拉式選單
        self.map_comb = ttk.Combobox(self, state='readonly')
        self.map_comb.config(values=db.FILE_NAME)
        self.map_comb.set(db.FILE_NAME[0])
        self.map_comb.place(x=4, y=100, width=-45, height=25, relx=0.75, relwidth=0.25)

    def get_map_name_and_build_env(self):
        # 隱藏地圖標籤、按鈕、下拉式選單
        self.choose_map_lab.place_forget()
        self.choose_map_btn.place_forget()
        self.map_comb.config(state="disabled")
        self.map_comb.place_forget()
        # 開始遊玩按鈕
        self.play_btn = tk.Button(self)
        # 遊戲是否進行
        self.play_state = False
        if self.mode == "maze":
            self.play_btn.config(text= Player_window_string['Button_text']['text_14'], command=self.maze_mode_playing, bg='#C4A98B')
            self.play_btn['font'] = self.ChiFonts
        elif self.mode == "node":
            self.play_btn.config(text= Player_window_string['Button_text']['text_14'], command=self.node_mode_playing, bg='#C4A98B')
            self.play_btn['font'] = self.ChiFonts
        self.play_btn.place(x=40, y=80, width=-80, height=30, relx=0.6, relwidth=0.4)
        # 建立環境
        self.map_name = self.map_comb.get()
        # 如果沒有選出地圖將關閉頁面
        if self.map_name == '':
            messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['messagebox_text']['message']['text_1'])
            self.destroy()
        self.info_scr.insert(tk.END, self.map_name + Player_window_string['info_scr']['text_3'])
        if self.mode == "maze":
            self.env.build_env(self.map_name)
        elif self.mode == "node":
            self.env.build_env(self.map_name)
            self.env.create_nodes()
            self.env_.build_env(self.map_name)
            self.env_.create_nodes()
            self.bg_list = []
        # 啟用遊戲資訊按鈕
        self.game_info_btn.config(state="normal")
        # 初始化角色狀態
        player_state = self.env.player
        self.player_state_tree.insert('', 'end', value=[player_state.lv, player_state.hp, player_state.atk, \
            player_state.def_, player_state.money, player_state.exp, \
            player_state.items['yellowKey'], player_state.items['blueKey'], player_state.items['redKey']])
        # 繪製地圖
        self.create_map()

    def maze_mode_playing(self):
        # 遊戲開始
        self.info_scr.insert(tk.END, Player_window_string['info_scr']['text_4'])
        # 隱藏開始遊玩、上下樓按鈕
        self.play_btn.place_forget()
        self.play_state = True
        self.up_btn.config(state="disabled")
        self.down_btn.config(state="disabled")
        # 停止遊玩按鈕
        self.stop_btn = tk.Button(self)
        self.stop_btn.config(text= Player_window_string['Button_text']['text_15'], command=self.maze_mode_stopping, bg='#C4A98B')
        self.stop_btn.place(x=40, y=80, width=-80, height=30, relx=0.6, relwidth=0.4)
        self.stop_btn['font'] = self.ChiFonts
        # 從第一層開始遊玩
        global global_image
        self.canvas.delete(self.image)
        global_image = tk.PhotoImage(file='pictures/MT_0.png')
        self.image = self.canvas.create_image(0, 0, anchor='nw', image = global_image)
        self.floor_index = 0
        # 玩家圖像
        self.player_image = self.canvas.create_image(self.player_pos[2]*32, self.player_pos[1]*32, anchor='nw', image=self.player_image_file)
        # 建立四個方向按鈕和確定按鈕
        self.up_bind = self.bind("<Up>", self.move_up)
        self.down_bind = self.bind("<Down>", self.move_down)
        self.left_bind = self.bind("<Left>", self.move_left)
        self.right_bind = self.bind("<Right>", self.move_right)
        self.confirm_bind = self.bind("<Return>", self.shop_confirm)

    def node_mode_playing(self):
        # 創建顯示能行動節點編號資訊欄位
        self.show_action_index_labelFrm = tk.LabelFrame(self)
        self.show_action_index_labelFrm.place(x=10, y=70, width=-20, height=75, relx=0.6, relwidth=0.4)
        action_index_label = tk.Label(self.show_action_index_labelFrm, text = Player_window_string['Button_text']['text_17'], font=(Player_window_string['Initialize']['font_style_Chinese'], 12, tkFont.BOLD))
        action_index_label.pack()
        # 遊戲開始
        self.info_scr.insert(tk.END, Player_window_string['info_scr']['text_4'])
        # 隱藏開始遊玩、上下樓按鈕
        self.play_btn.place_forget()
        self.play_state = True
        self.up_btn.config(state="disabled")
        self.down_btn.config(state="disabled")
        # 停止遊玩按鈕
        self.stop_btn = tk.Button(self)
        self.stop_btn.config(text=Player_window_string['Button_text']['text_15'], command=self.node_mode_stopping, bg='#C4A98B')
        self.stop_btn.place(x=40, y=20, width=-80, height=30, relx=0.6, relwidth=0.4)
        self.stop_btn['font'] = self.ChiFonts
        # 從第一層開始遊玩
        global global_image
        self.canvas.delete(self.image)
        global_image = tk.PhotoImage(file='pictures/MT_0.png')
        self.image = self.canvas.create_image(0, 0, anchor='nw', image = global_image)
        self.floor_index = 0
        # 玩家圖像
        self.player_image = self.canvas.create_image(self.player_pos[2]*32, self.player_pos[1]*32, anchor='nw', image=self.player_image_file)
        # 選擇節點和節點在info的資訊
        self.open_shop = False
        self.info_scr.insert(tk.END, Player_window_string['info_scr']['text_5'] + Player_window_string['info_scr']['text_6'] + \
            Player_window_string['info_scr']['text_7'])
        self.choose_node()

    def choose_node(self):
        # 取得可行動節點
        self.actions = self.env.get_feasible_actions()
        self.actions_ = self.env_.get_feasible_actions()
        # 取得可行動節點的位置
        self.action_list = [self.env.n2p[action] for action in self.actions]
        self.all_mask = []
        self.all_oval = []
        # 建立整個可選擇節點的圖樣
        for index, action in enumerate(self.actions):
            self.info_scr.insert(tk.END, '{:^5d}\t   {:<10s}\t{:<s}\n'.format(index, str(self.env.n2p[action]), action.id))
            next_step = self.env.n2p[action]
            mask = Image.open("pictures/mask.png")
            mask = ImageTk.PhotoImage(mask)
            if self.floor_index == next_step[0] and not self.open_shop:
                mask = self.canvas.create_image(next_step[2]*32, next_step[1]*32, anchor='nw', image=mask, tags=f"{index}")
                self.all_mask.append(mask)
            self.click_bind = self.canvas.tag_bind(mask, "<Button-1>", lambda event, choose=mask : self.click(event, choose))
            self.show_action_index_bind = self.canvas.tag_bind(mask, "<Enter>", lambda event, choose=mask : self.show_action_index(event, choose))
            self.destroy_action_index_bind = self.canvas.tag_bind(mask, "<Leave>", self.destroy_action_index)
            if self.floor_index == next_step[0] and not self.open_shop:
                oval = self.canvas.create_oval(next_step[2]*32, next_step[1]*32, next_step[2]*32+32, next_step[1]*32+32, outline='#77e91b', width=3, stipple="gray25")
                self.all_oval.append(oval)

    def move_up(self, event):
        # 紀錄發生事件前位置
        self.pre_pos = copy.copy(self.player_pos)
        # 往上走一步
        self.player_pos[1] -= 1
        self.action = 3
        ob, reward, ending = self.env.step(self.action)
        # 紀錄發生事件後位置
        self.player_pos = ob[:3]
        # 角色移動
        self.player_move(ob, ending)

    def move_down(self, event):
        # 紀錄發生事件前位置
        self.pre_pos = copy.copy(self.player_pos)
        # 往下走一步
        self.player_pos[1] += 1
        self.action = 1
        ob, reward, ending = self.env.step(self.action)
        # 紀錄發生事件後位置
        self.player_pos = ob[:3]
        # 角色移動
        self.player_move(ob, ending)

    def move_left(self, event):
        # 紀錄發生事件前位置
        self.pre_pos = copy.copy(self.player_pos)
        # 往左走一步
        self.player_pos[2] -= 1
        self.action = 2
        ob, reward, ending = self.env.step(self.action)
        # 紀錄發生事件後位置
        self.player_pos = ob[:3]
        # 角色移動
        self.player_move(ob, ending)

    def move_right(self, event):
        # 紀錄發生事件前位置
        self.pre_pos = copy.copy(self.player_pos)
        # 往右走一步
        self.player_pos[2] += 1
        self.action = 0
        ob, reward, ending = self.env.step(self.action)
        # 紀錄發生事件後位置
        self.player_pos = ob[:3]
        # 角色移動
        self.player_move(ob, ending)

    def show_action_index(self, event, choose):
        # 顯示行動編號到show_action_index_labelFrm
        action = str(self.canvas.gettags(choose)[0])
        self.show_action_index_label = tk.Label(self.show_action_index_labelFrm, text = action)
        self.show_action_index_label.pack()

    def destroy_action_index(self, event):
        # 移除在show_action_index_labelFrm的行動編號
        self.show_action_index_label.destroy()

    def click(self, event, choose):
        # 初始化參數
        self.pre_pos = copy.copy(self.player_pos)
        action = int(self.canvas.gettags(choose)[0])
        ending = self.env.step(self.actions[action])
        # 紀錄玩家遊玩路徑
        for node in self.actions_:
            if self.env_.n2p[node] == self.env.n2p[self.actions[action]]:
                index = self.actions_.index(node)
                ending_ = self.env_.step(self.actions_[index])
                self.save_action.append(index)
        print(self.save_action)

        next_pos = self.action_list[action]
        self.player_pos = list(next_pos)
        # 進入商店
        if tuple(next_pos[:3]) in self.env.env_data['npcs'] and self.env.env_data['npcs'][tuple(next_pos[:3])] == "shop":
            self.open_shop = True
            self.node_open_shop()
        # 樓層移動
        if next_pos in self.env.env_data['floors']['changeFloor']:
            self.player_pos = list(self.env.env_data['floors']['changeFloor'][next_pos])
            self.floor_index = self.player_pos[0]
            self.canvas.delete('all')
            self.image_file = tk.PhotoImage(file='pictures/MT_{:d}.png'.format(self.player_pos[0]))
            self.image = self.canvas.create_image(0, 0, anchor='nw', image = self.image_file)
            for bg in self.bg_list:
                if bg[0] == self.floor_index:
                    self.floor_image = self.canvas.create_image(bg[2]*32, bg[1]*32, anchor='nw', image=self.floor_image_file)
            self.change_floor_status()
        # 事件觸發
        self.incident(next_pos, ending)
        # 貼上角色圖像
        self.canvas.delete(self.player_image)
        self.player_image = self.canvas.create_image(self.player_pos[2]*32, self.player_pos[1]*32, anchor='nw', image=self.player_image_file)
        # 更新角色狀態
        for item in self.player_state_tree.get_children():
            self.player_state_tree.delete(item)
        player_state = self.env.player
        self.player_state_tree.insert('', 'end', value=[player_state.lv, player_state.hp, player_state.atk, \
            player_state.def_, player_state.money, player_state.exp, \
            player_state.items['yellowKey'], player_state.items['blueKey'], player_state.items['redKey']])
        # 刪除上一步未選擇的圖案
        for mask in self.all_mask:
            self.canvas.delete(mask)
        for oval in self.all_oval:
            self.canvas.delete(oval)
        # 更新行動編號
        self.show_action_index_label.destroy()
        # 遊戲未結束
        if ending == "continue":
            self.info_scr.delete(f'end-{len(self.actions)+1}l', 'end-1l')
            self.choose_node()

    def shop_confirm(self, event):
        # 商店物品確認
        self.action = 4
        ob, reward, ending = self.env.step(self.action)

    def maze_open_shop(self, observation):
        # 商店選項文字
        shop_text = ''
        # 商店物品的值
        shop_value = 1
        # 選到的商店物品選項
        self.commodity_index = 1
        # 商店物品個數
        options_c = 0
        # 單選按鈕集合
        global radioValue
        radioValue = tk.IntVar()
        radioValue.set(1)
        self.commodity = radioValue.get()
        # 商店視窗
        self.shop_window = tk.Toplevel(self)
        # 選取商店物品
        def get_value():
            self.commodity = radioValue.get()
            times = abs(self.commodity - self.commodity_index)
            for _ in range(times):
                if self.commodity > self.commodity_index:
                    action = 1
                    ob, reward, ending = self.env.step(action)
                elif self.commodity < self.commodity_index:
                    action = 3
                    ob, reward, ending = self.env.step(action)
                elif self.commodity == self.commodity_index:
                    pass
            self.commodity_index = self.commodity
        # 產生單選按鈕文字
        for pos, content in self.env.env_data['npcs'].items():
            if tuple(observation[:3]) == pos[:3] and len(pos) == 4:
                options_c += 1
                shop_text = ''
                for goods in content:
                    if goods['type'] == 'addValue' and goods['name'] == 'player.money':
                        shop_text = f"MON Cost : {goods['value']}  -->  "
                    elif goods['type'] == 'addValue' and goods['name'] == 'player.exp':
                        shop_text = f"EXP Cost : {goods['value']}  -->  "
                    elif goods['type'] == 'addValue' and goods['name'] == 'player.hp':
                        shop_text += f"HP  : +{goods['value']}\t"
                    elif goods['type'] == 'addValue' and goods['name'] == 'player.atk':
                        shop_text += f"ATK : +{goods['value']}\t"
                    elif goods['type'] == 'addValue' and goods['name'] == 'player.def':
                        shop_text += f"DEF : +{goods['value']}\t"
                    elif goods['type'] == 'addValue' and goods['name'] == 'player.exp':
                        shop_text += f"EXP : +{goods['value']}\t"
                    elif goods['type'] == 'addValue' and goods['name'] == 'player.lv':
                        shop_text += f"LV  : +{goods['value']}\t"
                    else:
                        pass
                # 產生單選按鈕
                shop = tk.Radiobutton(self.shop_window, text=shop_text, variable=radioValue, value=shop_value, command=get_value)
                shop.grid(column=0, row=shop_value, sticky="W")
                shop_value += 1
        # 購買按鈕
        purchase_btn = tk.Button(self.shop_window)
        purchase_btn.config(text="Buy", command=lambda:self.maze_complete_the_purchase(self.commodity))
        purchase_btn.grid(column=0, row=shop_value, sticky="W")
        # 離開商店按鈕
        exit_shop_btn = tk.Button(self.shop_window)
        exit_shop_btn.config(text="Exit", command=lambda:self.maze_exit_shop(options_c))
        exit_shop_btn.grid(column=1, row=shop_value, sticky="W")
        # 暫時停止移動
        self.unbind("<Up>", self.up_bind)
        self.unbind("<Down>", self.down_bind)
        self.unbind("<Left>", self.left_bind)
        self.unbind("<Right>", self.right_bind)
        self.unbind("<Return>", self.confirm_bind)

    def node_open_shop(self):
        # 商店選項文字
        shop_text = ''
        # 商店物品的值
        shop_value = 1
        # 選到的商店物品選項
        self.commodity_index = 1
        # 商店物品個數
        options_c = 0
        # 單選按鈕集合
        global radioValue
        radioValue = tk.IntVar()
        radioValue.set(1)
        # 商店視窗
        self.shop_window = tk.Toplevel(self)
        # 選取商店物品
        self.commodity = 1
        def get_value():
            self.commodity = radioValue.get()
        # 產生單選按鈕文字
        for pos, content in self.env.env_data['npcs'].items():
            if tuple(self.player_pos[:3]) == pos[:3] and len(pos) == 4:
                options_c += 1
                shop_text = ''
                for goods in content:
                    if goods['type'] == 'addValue' and goods['name'] == 'player.money':
                        shop_text = f"MON Cost : {goods['value']}  -->  "
                    elif goods['type'] == 'addValue' and goods['name'] == 'player.exp':
                        shop_text = f"EXP Cost : {goods['value']}  -->  "
                    elif goods['type'] == 'addValue' and goods['name'] == 'player.hp':
                        shop_text += f"HP  : +{goods['value']}\t"
                    elif goods['type'] == 'addValue' and goods['name'] == 'player.atk':
                        shop_text += f"ATK : +{goods['value']}\t"
                    elif goods['type'] == 'addValue' and goods['name'] == 'player.def':
                        shop_text += f"DEF : +{goods['value']}\t"
                    elif goods['type'] == 'addValue' and goods['name'] == 'player.exp':
                        shop_text += f"EXP : +{goods['value']}\t"
                    elif goods['type'] == 'addValue' and goods['name'] == 'player.lv':
                        shop_text += f"LV  : +{goods['value']}\t"
                    else:
                        pass
                # 產生單選按鈕
                shop = tk.Radiobutton(self.shop_window, text=shop_text, variable=radioValue, value=shop_value, command=get_value)
                shop.grid(column=0, row=shop_value, sticky="W")
                shop_value += 1
        # 購買按鈕
        purchase_btn = tk.Button(self.shop_window)
        purchase_btn.config(text="Buy", command=lambda:self.node_complete_the_purchase(self.commodity))
        purchase_btn.grid(column=0, row=shop_value, sticky="W")
        # 離開商店按鈕
        exit_shop_btn = tk.Button(self.shop_window)
        exit_shop_btn.config(text="Exit", command=self.node_exit_shop)
        exit_shop_btn.grid(column=1, row=shop_value, sticky="W")

    def maze_complete_the_purchase(self, variable):
        # 確認購買
        action = 4
        ob, reward, ending = self.env.step(action)
        # 是否成功購買提示
        if ending == "continue":
            messagebox.showwarning(Player_window_string['messagebox_text']['titel']['text_1'], Player_window_string['messagebox_text']['message']['text_1'])
        elif ending == "stop":
            messagebox.showwarning(Player_window_string['messagebox_text']['titel']['text_1'], Player_window_string['messagebox_text']['message']['text_2'])
        # 更新角色狀態
        for item in self.player_state_tree.get_children():
            self.player_state_tree.delete(item)
        player_state = self.env.player
        self.player_state_tree.insert('', 'end', value=[player_state.lv, player_state.hp, player_state.atk, \
            player_state.def_, player_state.money, player_state.exp, \
            player_state.items['yellowKey'], player_state.items['blueKey'], player_state.items['redKey']])
        # 啟用方向鍵
        self.up_bind = self.bind("<Up>", self.move_up)
        self.down_bind = self.bind("<Down>", self.move_down)
        self.left_bind = self.bind("<Left>", self.move_left)
        self.right_bind = self.bind("<Right>", self.move_right)
        self.confirm_bind = self.bind("<Return>", self.shop_confirm)
        # 關閉商店視窗
        self.shop_window.destroy()

    def node_complete_the_purchase(self, variable):
        # 是否成功購買提示
        if (tuple(self.player_pos) + (variable,)) in self.env.env_data['npcs']:
            for command in self.env.env_data['npcs'][tuple(self.player_pos) + (variable,)]:
                if command['type'] == 'if':
                    if eval(command['condition'].replace('player', 'self.env.player')):
                        # 確認購買
                        action = variable
                        ending = self.env.step(self.actions[action])
                        # 紀錄玩家遊玩路徑
                        for node in self.actions_:
                            if self.env_.n2p[node] == self.env.n2p[self.actions[action]]:
                                index = self.actions_.index(node)
                                ending = self.env_.step(self.actions_[index])
                                self.save_action.append(index)
                                print(self.save_action)
                        messagebox.showwarning(Player_window_string['messagebox_text']['titel']['text_1'], Player_window_string['messagebox_text']['message']['text_1'])
                        self.open_shop = False
                        self.info_scr.delete(f'end-{len(self.actions)+1}l', 'end-1l')
                        # 繼續遊戲
                        self.choose_node()
                    else:
                        messagebox.showwarning(Player_window_string['messagebox_text']['titel']['text_1'], Player_window_string['messagebox_text']['message']['text_2'])
                        self.open_shop = False
                        self.info_scr.delete(f'end-{len(self.actions)+1}l', 'end-1l')
                        # 刪除玩家遊玩路徑
                        # self.save_action.pop()
                        # print(self.save_action)
                        # 繼續遊戲
                        self.choose_node()
        # 更新角色狀態
        for item in self.player_state_tree.get_children():
            self.player_state_tree.delete(item)
        player_state = self.env.player
        self.player_state_tree.insert('', 'end', value=[player_state.lv, player_state.hp, player_state.atk, \
            player_state.def_, player_state.money, player_state.exp, \
            player_state.items['yellowKey'], player_state.items['blueKey'], player_state.items['redKey']])
        # 關閉商店視窗
        self.shop_window.destroy()

    def maze_exit_shop(self, options_c):
        # 離開商店
        self.shop_window.destroy()
        state_value = self.env.observation[3]
        times = (options_c - state_value) + 1
        for _ in range(times):
            action = 1
            ob, reward, ending = self.env.step(action)
        action = 4
        ob, reward, ending = self.env.step(action)
        # 啟用方向鍵
        self.up_bind = self.bind("<Up>", self.move_up)
        self.down_bind = self.bind("<Down>", self.move_down)
        self.left_bind = self.bind("<Left>", self.move_left)
        self.right_bind = self.bind("<Right>", self.move_right)
        self.confirm_bind = self.bind("<Return>", self.shop_confirm)

    def node_exit_shop(self):
        # 離開商店
        self.shop_window.destroy()
        self.open_shop = False
        # self.save_action.pop()
        self.env.back_step(1)
        self.info_scr.delete(f'end-{len(self.actions)+1}l', 'end-1l')
        self.choose_node()

    def player_move(self, observation, ending):
        # 進入商店
        if tuple(observation[:3]) in self.env.env_data['npcs'] and self.env.env_data['npcs'][tuple(observation[:3])] == "shop":
            self.maze_open_shop(observation)
        # 樓層移動
        if observation[0] != self.pre_pos[0]:
            time.sleep(0.5)
            self.canvas.delete('all')
            self.env.draw_map_layer(self.pre_pos[0])
            self.image_file = tk.PhotoImage(file='pictures/MT_{:d}.png'.format(observation[0]))
            self.image = self.canvas.create_image(0, 0, anchor='nw', image = self.image_file)
            self.floor_index = observation[0]
            self.change_floor_status()
        # 事件觸發
        self.incident(observation, ending)
        # 貼上角色圖像
        self.canvas.delete(self.player_image)
        self.player_image = self.canvas.create_image(self.player_pos[2]*32, self.player_pos[1]*32, anchor='nw', image=self.player_image_file)
        # 更新角色狀態
        for item in self.player_state_tree.get_children():
            self.player_state_tree.delete(item)
        player_state = self.env.player
        self.player_state_tree.insert('', 'end', value=[player_state.lv, player_state.hp, player_state.atk, \
            player_state.def_, player_state.money, player_state.exp, \
            player_state.items['yellowKey'], player_state.items['blueKey'], player_state.items['redKey']])

    def incident(self, observation, ending):
        # 返回上一步
        if ending == "invalid" or ending == "stop":
            self.player_pos = self.pre_pos
        # 成功觸發事件
        elif ending == "continue":
            map_position = self.env_map[self.player_pos[0]][self.player_pos[1]][self.player_pos[2]]
            if map_position != 0:
                if self.mode == 'node':
                    if self.env.p2n[(self.player_pos[0], self.player_pos[1], self.player_pos[2])].class_ == 'npcs':
                        pass
                    else:
                        self.floor_image = self.canvas.create_image(self.player_pos[2]*32, self.player_pos[1]*32, anchor='nw', image=self.floor_image_file)
                        self.bg_list.append((self.player_pos[0], self.player_pos[1], self.player_pos[2]))
                elif self.mode == 'maze':
                    try:
                        if self.env.env_data['npcs'][(self.player_pos[0], self.player_pos[1], self.player_pos[2])] == 'shop':
                            pass
                    except KeyError:
                        self.floor_image = self.canvas.create_image(self.player_pos[2]*32, self.player_pos[1]*32, anchor='nw', image=self.floor_image_file)
        # 結局
        elif ending == "clear" or ending == "death":
            if ending == "clear":
                messagebox.showwarning(Player_window_string['messagebox_text']['titel']['text_1'], Player_window_string['messagebox_text']['message']['text_3'])
                if self.mode == 'node':
                    self.save_action_to_npy()
                    self.info_scr.insert(tk.END, Player_window_string['info_scr']['text_8'], 'node_done')
                    self.info_scr.tag_config('node_done', foreground='red')
            elif ending == "death":
                messagebox.showwarning(Player_window_string['messagebox_text']['titel']['text_1'], Player_window_string['messagebox_text']['message']['text_4'])
            self.stop_btn.config(state="disabled")
            if self.mode == "maze":
                self.maze_mode_stopping()
            elif self.mode == "node":
                self.node_mode_stopping()

    def maze_mode_stopping(self):
        self.unbind("<Up>", self.up_bind)
        self.unbind("<Down>", self.down_bind)
        self.unbind("<Left>", self.left_bind)
        self.unbind("<Right>", self.right_bind)
        self.unbind("<Return>", self.confirm_bind)
        self.stop_btn.config(state="disabled")
        self.play_state = False
        self.back_show_learning = tk.Button(self)
        self.back_show_learning.config(text= Player_window_string['Button_text']['text_16'], command = lambda:self.destroy(), bg='#C4A98B')
        self.back_show_learning.place(x=40, y=20, width=-80, height=30, relx=0.6, relwidth=0.4)
        self.back_show_learning['font'] = self.ChiFonts

    def node_mode_stopping(self):
        for mask in self.all_mask:
            self.canvas.delete(mask)
        for oval in self.all_oval:
            self.canvas.delete(oval)
        self.stop_btn.place_forget()
        self.play_state = False
        self.back_show_learning = tk.Button(self)
        self.back_show_learning.config(text= Player_window_string['Button_text']['text_16'] ,command = lambda:self.destroy(), bg='#C4A98B')
        self.back_show_learning.place(x=40, y=20, width=-80, height=30, relx=0.6, relwidth=0.4)
        self.back_show_learning['font'] = self.ChiFonts

    def create_map(self):
        # 目前樓層
        self.floor_index = 0
        # 最大樓層
        self.floor_max = self.env.draw_map('pictures/MT_')
        # 繪製第一層
        self.image_file = tk.PhotoImage(file='pictures/MT_{:d}.png'.format(self.floor_index))
        self.canvas.delete('all')
        self.image = self.canvas.create_image(0, 0, anchor='nw', image = self.image_file)
        self.canvas.pack()
        # 上、下樓按鈕顯示
        self.change_floor_status()
        # 擷取玩家、地板圖像
        self.tileImage = Image.open('pictures/baseTile.png')
        self.player_icon = self.tileImage.crop((32*7, 0, 32*8, 32))
        self.player_image_file = ImageTk.PhotoImage(self.player_icon)
        self.floor_icon = self.tileImage.crop((0, 0, 32, 32))
        self.floor_image_file = ImageTk.PhotoImage(self.floor_icon)
        # 玩家起點
        self.player_pos = self.find_player_pos()
        self.pre_pos = None

    def find_player_pos(self):
        player_id = None
        # 初始化地圖
        self.env_map = np.array(self.env.env_data['floors']['map'])
        # 迭代地圖找出角色位置
        for key, value in self.env.env_data['maps'].items():
            if value['id'] == 'player':
                if player_id is None:
                    player_id = key
        player_pos = np.argwhere(self.env_map==player_id)[0]
        return player_pos

    def save_action_to_npy(self):
        # 設置檔名順序
        index = 0
        # 尋找檔名
        for file_name in os.listdir('player_route/'):
            map_name_temp = file_name.split('_v')[0]
            if map_name_temp == self.map_name:
                index += 1
        # 儲存成npy檔
        np.save('player_route/' + self.map_name + '_v' + str(index) + '_h' + str(self.env_.player.hp), np.array(self.save_action))

    def maze_window(self):
        # 模式名稱
        self.mode = "maze"
        # 隱藏模式按鈕
        self.maze_mode_btn.place_forget()
        self.node_mode_btn.place_forget()
        # 建立迷宮物件
        self.env = Mota_Maze_Player()
        # 建立基本視窗
        self.base_window()

    def node_window(self):
        # 模式名稱
        self.mode = "node"
        # 紀錄玩家遊玩路徑
        self.save_action = []
        # 隱藏模式按鈕
        self.maze_mode_btn.place_forget()
        self.node_mode_btn.place_forget()
        # 建立節點物件
        self.env = Mota_Player()
        self.env_ = Mota()
        # 建立基本視窗
        self.base_window()

if __name__ == "__main__":
    win = window_Player()
    win.mainloop()
