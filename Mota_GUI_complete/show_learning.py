from import_tool import *
from String_ import *
import database as db
import gc
import copy
from environment_maze import Mota_Maze
from window_resize import win_resize
from animation_environment import Mota
from Learning_function import learning
from Demonstration_model_predict import Demonstration_predict
from Demonstration_windows import Demonstration_window
from MCTSfDm_windows import MCTSfDm_window
from MCTSfDm_run import MCTSfDm_learning
from ToolTip import Tooltip
from showmap import Frame_Scrollbar
from results_plot import Save_File
from Player_window import window_Player

global_image = None
#============================================================================
#  顯示訓練過程主介面
#============================================================================
class show_learning(tk.Tk):
    def __init__(self):
        super(show_learning, self).__init__()
        # 設定視窗參數
        self.title(Show_learning_string['Initialize']['title'])
        win_resize(self)
        winWidth = 806
        winHeight = 539
        self.x = (self.winfo_screenwidth() - winWidth) // 2
        self.y = (self.winfo_screenheight() - winHeight) // 3
        self.geometry('%sx%s+%s+%s' % (winWidth, winHeight, self.x, self.y))
        self.minsize(winWidth, winHeight)
        self.floor_index = 0
        self.re = 0
        # 中文字體
        self.ChiFont = font.Font(family=Show_learning_string['Initialize']['font_style_Chinese'], weight=tkFont.BOLD)
        # 英文字體
        self.EngFont = font.Font(family=Show_learning_string['Initialize']['font_style_English'], weight=tkFont.BOLD)
        self.first_btn()
        self.iconbitmap('pictures/th170_kutaka.ico')
#============================================================================
#  訓練結果圖表按鈕
#============================================================================
    def show_map_function(self):
        if self.nb_displayed:
            self.nb.place_forget()
            self.result_chart_btn.config(text = Show_learning_string['Button_text']['Home']['text_1'])
        else:
            self.nb.place(height=-40, relwidth=0.6, relheight=1.0)
            self.map_function()
            self.result_chart_btn.config(text= Show_learning_string['Button_text']['Home']['text_2'])
            if not self.nb_refreshed:
                self.nb_refreshed = True
        self.nb_displayed = not self.nb_displayed
#============================================================================
#  結果圖表
#============================================================================
    def map_function(self):
        tk.Label(self.tab1.contents,
                 font=(Show_learning_string['Initialize']['font_style_Chinese'], 9, tkFont.BOLD), bg='lavender'
                 ).pack(anchor='nw', padx=5, pady=5)
#============================================================================
#  開始按鈕
#============================================================================
    def first_btn(self):
        self.btn = tk.Button(self)
        Q_01 = tk.PhotoImage(file ='pictures/Q_01.png')
        self.Q_01 = Q_01.subsample(6,6)
        self.btn.config(text= Show_learning_string['Button_text']['Home']['text_3'],bg='#C4A98B',image = self.Q_01,compound = tk.LEFT,command = self.choose_mode)
        self.btn['font'] = self.ChiFont
        self.btn.pack()
        self.btn.place(y=-310, rely=1, relx=0.43, width=110)
        Tooltip(self.btn, text= Show_learning_string['tips_text']['text_1'], wraplength=200)
#============================================================================
#  選擇版本按鈕
#============================================================================
    def choose_mode(self):
        # 刪除  first_btn
        self.btn.destroy()
        # 建立節點版魔塔視窗物件
        self.Node_btn = tk.Button(self)
        S_01 = tk.PhotoImage(file ='pictures/S_01.png')
        self.S_01 = S_01.subsample(14,14)
        self.Node_btn.config(text=Show_learning_string['Button_text']['Home']['text_4'],image = self.S_01,bg='#C4A98B',command=lambda : self.build_node_window(),compound = tk.LEFT)
        self.Node_btn['font'] = self.ChiFont
        self.Node_btn.place(y=-400, rely=1, relx=0.20, width=200) #y=-450, rely=1, relx=0.43, width=200
        Tooltip(self.Node_btn, text = Show_learning_string['tips_text']['text_2'])
        # 建立迷宮版魔塔視窗物件
        self.Maze_btn = tk.Button(self)
        S_02 = tk.PhotoImage(file ='pictures/S_02.png')
        self.S_02 = S_02.subsample(14,14)
        self.Maze_btn.config(text=Show_learning_string['Button_text']['Home']['text_5'],image = self.S_02,bg='#C4A98B',command=lambda : self.build_maze_window() ,compound = tk.LEFT)
        self.Maze_btn['font'] = self.ChiFont
        self.Maze_btn.place(y=-400, rely=1, relx=0.48, width=200)#0.18
        Tooltip(self.Maze_btn, text=Show_learning_string['tips_text']['text_3'])
        # 建立模仿學習模型視窗
        self.Demonstration_btn = tk.Button(self)
        S_03 = tk.PhotoImage(file='pictures/S_03.png')
        self.S_03 = S_03.subsample(16,16)
        self.Demonstration_btn.config(text = Show_learning_string['Button_text']['Home']['text_6'],image = self.S_03 ,bg ='#C4A98B', command = lambda:self.create_Demonstration(), compound = tk.LEFT)
        self.Demonstration_btn['font'] = self.ChiFont
        self.Demonstration_btn.pack()
        self.Demonstration_btn.place(y=-310, rely=1, relx=0.20, width=200)
        Tooltip(self.Demonstration_btn, text = Show_learning_string['tips_text']['text_4'])
        # 建立MCTSfDm模型視窗
        S_05 = tk.PhotoImage(file='pictures/S_05.png')
        self.S_05 = S_05.subsample(14,14)
        self.MCTSfDm_btn = tk.Button(self)
        self.MCTSfDm_btn.config(text = Show_learning_string['Button_text']['Home']['text_7'], bg='#C4A98B',image = self.S_05,command = lambda:self.MCTSfDm_env(), compound = tk.LEFT)
        self.MCTSfDm_btn['font'] = self.ChiFont
        self.MCTSfDm_btn.place(y=-310, rely=1, relx=0.48, width=200)
        Tooltip(self.MCTSfDm_btn, text= Show_learning_string['tips_text']['text_5'])
        # 建立玩家自玩視窗按鈕
        S_06 = tk.PhotoImage(file='pictures/S_06.png')
        self.S_06 = S_06.subsample(14,14)
        self.play_byself_btn = tk.Button(self)
        self.play_byself_btn.config(text =Show_learning_string['Button_text']['Home']['text_8'],image=self.S_06,bg='#C4A98B',command = lambda:self.call_player_window(), compound = tk.LEFT)
        self.play_byself_btn['font'] = self.ChiFont
        self.play_byself_btn.place(y=-220, rely=1, relx=0.48, width=200)
        Tooltip(self.play_byself_btn, text = Show_learning_string['tips_text']['text_6'])
        # 結束視窗物件
        self.end_btn = tk.Button(self)
        S_04 = tk.PhotoImage(file='pictures/S_04.png')
        self.S_04 = S_04.subsample(14,14)
        self.end_btn.config(text = Show_learning_string['Button_text']['Home']['text_9'],image = self.S_04, bg ='#C4A98B', command = self.Reset_GUI, compound = tk.LEFT)
        self.end_btn['font'] = self.ChiFont

        self.end_btn.place(y=-220, rely=1, relx=0.20, width=200)
        Tooltip(self.end_btn, text=Show_learning_string['tips_text']['text_7'])
#============================================================================
#  呼叫自行遊玩程式
#============================================================================
    def call_player_window(self):
        win = window_Player()
        win.mainloop()
#============================================================================
#  介面之元件宣告
#============================================================================
    def build_base_window(self):
        # 建立下樓按鈕
        self.down_btn = ttk.Button(self, text=Show_learning_string['Button_text']['build_win']['text_1'],command=self.floor_down)
        self.down_btn.place(y=-35, rely=1.0, relx=0.20, width=160)
        self.down_btn.config(state='disabled')
        Tooltip(self.down_btn, text= Show_learning_string['tips_text']['text_8'])
        # 建立上樓按鈕
        self.up_btn = ttk.Button(self, text=Show_learning_string['Button_text']['build_win']['text_2'], command=self.floor_up)
        self.up_btn.place(y=-35, rely=1.0, relx=0.40, width=160)
        self.up_btn.config(state='disabled')
        Tooltip(self.up_btn, text=Show_learning_string['tips_text']['text_9'])

        # 建立訓練結果按鈕
        self.nb = ttk.Notebook(self)
        self.tab1 = Frame_Scrollbar(self.nb, bg='lavender')
        self.nb.add(self.tab1, text=Show_learning_string['Button_text']['build_win']['text_3'])

        self.result_chart_btn = ttk.Button(self, text = Show_learning_string['Button_text']['build_win']['text_4'], command = lambda:self.show_map_function())
        self.result_chart_btn.place(y=-35, rely=1.0, relx=0.0, width=160)
        #self.result_chart_btn.config(state='disabled')
        self.nb_displayed = False
        self.nb_refreshed = False
        Tooltip(self.result_chart_btn, text=Show_learning_string['tips_text']['text_10'])
        self.result_chart_btn.config(state='disabled')

        if self.nb_displayed:
            self.show_map_function()

        # 建立確定地圖按鈕
        self.map_btn = tk.Button(self)
        # 建立重置按鈕
        self.reset_algorithm = tk.Button(self)
        self.reset_env = tk.Button(self)
        self.reset_mode = tk.Button(self)
        # 建立選擇地圖按鈕
        self.select_map_btn = tk.Button(self)
        Q_02 = tk.PhotoImage(file ='pictures/Q_02.png')
        self.Q_02 = Q_02.subsample(5,5)
        self.select_map_btn.config(text = Show_learning_string['Button_text']['select_env']['text_1'],image =self.Q_02 ,bg='#C4A98B', command=self.select_map_name,compound = tk.LEFT)
        self.select_map_btn['font'] = self.ChiFont
        self.select_map_btn.pack()
        self.select_map_btn.place(x=40, y=80, width=-80, height=30, relx=0.6, relwidth=0.4)

        self.comb = ttk.Combobox(self, state='readonly')
        self.map_lab = tk.Label(self, text = Show_learning_string['label_text']['select_env']['text_1'], bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 12, tkFont.BOLD))
        self.algorithm_lab = tk.Label(self, bg='#C4A98B',text=Show_learning_string['label_text']['select_env']['text_2'],font=(Show_learning_string['Initialize']['font_style_Chinese'], 8, tkFont.BOLD))
        self.rounds_lab = tk.Label(self, bg='#C4A98B',text= Show_learning_string['label_text']['select_env']['text_3'],font=(Show_learning_string['Initialize']['font_style_Chinese'], 9, tkFont.BOLD))
        self.en = ttk.Entry(self)

        self.labframe = tk.LabelFrame(self, text=Show_learning_string['label_text']['select_env']['text_4'], bg='#C4A98B')
        self.labframe.place(x=10, y=160, width=-20, height=-170, relx=0.6, relwidth=0.4, relheight=1.0)

        self.scr = scrolledtext.ScrolledText(self.labframe)
        self.scr.pack(fill='both', expand=True)

        im = Image.open('pictures/logo.jpg')
        self.img = ImageTk.PhotoImage(im)
        self.imLabel = tk.Label(self, image=self.img, bg ='#C4A98B')
        self.imLabel.place(y=5, relx=0.615, relwidth=0.37)
#============================================================================
#  建立節點介面環境
#============================================================================
    def build_node_window(self):
        # 建立畫圖區域
        self.frm = tk.LabelFrame(self, text='Map')
        self.frm.place(height=-40, relwidth=0.6, relheight=1.0)
        # 辨識目前魔塔版本
        self.mota_name = 'node'
        # 隱藏按鈕
        self.Node_btn.place_forget()
        self.Maze_btn.place_forget()
        self.Demonstration_btn.place_forget()
        self.MCTSfDm_btn.place_forget()
        self.end_btn.place_forget()
        self.play_byself_btn.place_forget()
        # 建立節點版環境
        self.env = Mota(self.frm)
        # 學習演算法
        self.ALGORITHM = ['Q-Learning', 'Q-Learning v2', 'Sarsa', 'MCTS', 'MCTS v2']
        # 建立基本視窗
        self.build_base_window()
        self.scr.insert(tk.END, Show_learning_string['scr_text']['create_mota']['text_1'])
#============================================================================
#  建立迷宮介面環境
#============================================================================
    def build_maze_window(self):
        # 建立畫圖區域
        self.frm = tk.LabelFrame(self, text='Map',bg ='#F2E9E0')
        self.frm.place(height=-40, relwidth=0.6, relheight=1.0)
        # 辨識目前魔塔版本
        self.mota_name = 'maze'
        # 隱藏按鈕
        self.Node_btn.place_forget()
        self.Maze_btn.place_forget()
        self.Demonstration_btn.place_forget()
        self.MCTSfDm_btn.place_forget()
        self.end_btn.place_forget()
        self.play_byself_btn.place_forget()
        # 建立迷宮版環境
        self.env = Mota_Maze()
        # 學習演算法
        self.ALGORITHM = ['Q-Learning', 'SarsaLambda', 'DeepQNetwork', 'DoubleDeepQNetwork','PrioritizedDQN','Actor-Critic']
        # 建立地圖畫布
        self.canvas = tk.Canvas(self.frm, height=32*15, width=32*15)
        self.build_base_window()
        self.scr.insert(tk.END, Show_learning_string['scr_text']['create_mota']['text_2'])
#============================================================================
#  建立模仿學習介面環境
#============================================================================
    def build_node_2_window(self):
        self.frm = ttk.LabelFrame(self, text='Map')
        self.frm.place(height=-40, relwidth=0.6, relheight=1.0)
        # 辨識目前魔塔版本
        self.mota_name = 'node_1'
        # 隱藏按鈕
        self.Node_btn.place_forget()
        self.Maze_btn.place_forget()
        self.Demonstration_btn.place_forget()
        self.MCTSfDm_btn.place_forget()
        self.end_btn.place_forget()
        self.play_byself_btn.place_forget()
        # 建立節點版環境
        self.env = Mota(self.frm)
        # 學習演算法
        self.ALGORITHM = ['Multilayer Perceptron', 'Random Forests','BoostTree', 'Support Vector Machine', 'K Nearest Neighbor']
        # 建立基本視窗
        self.build_base_window()
        self.scr.insert(tk.END, Show_learning_string['scr_text']['create_mota']['text_3'])
#============================================================================
#  建立MCTSfDm介面環境
#============================================================================
    def build_node_3_window(self):
        self.frm = ttk.LabelFrame(self, text='Map')
        self.frm.place(height=-40, relwidth=0.6, relheight=1.0)
        # 辨識目前魔塔版本
        self.mota_name = 'node_2'
        # 隱藏按鈕
        self.Node_btn.place_forget()
        self.Maze_btn.place_forget()
        self.Demonstration_btn.place_forget()
        self.MCTSfDm_btn.place_forget()
        self.end_btn.place_forget()
        self.play_byself_btn.place_forget()
        # 建立節點版環境
        self.env = Mota(self.frm)
        # 學習演算法
        self.ALGORITHM = ['MCTSfD','QLfD']
        # 建立基本視窗
        self.build_base_window()
        self.scr.insert(tk.END, Show_learning_string['scr_text']['create_mota']['text_4'])
#============================================================================
#  選擇地圖
#============================================================================
    def select_map_name(self):
        # 隱藏按鈕
        self.select_map_btn.place_forget()
        self.reset_algorithm.place_forget()
        self.reset_env.place_forget()
        self.reset_mode.place_forget()

        self.map_lab.place(x=40, y=65, width=-80, height=30, relx=0.7, relwidth=0.2)
        map_btn_img = tk.PhotoImage(file ='pictures/Start.png')
        self.map_btn_img = map_btn_img.subsample(4,4)
        self.map_btn.config(text = Show_learning_string['Button_text']['universal']['text_1'],bg='#C4A98B',image=self.map_btn_img, command=self.map_from_name,compound = tk.LEFT)
        self.map_btn['font'] = self.ChiFont
        self.map_btn.place(x =500, y=100, width=-30, relwidth=0.15)
        self.comb.set('')
        self.comb.config(values=db.FILE_NAME)
        self.comb.place(x=4, y=100, width=-45, height=20, relx=0.75, relwidth=0.25)
#============================================================================
#  選擇地圖環境 及 建構環境
#============================================================================
    def map_from_name(self):
        # name 是地圖名字
        name = self.comb.get()
        self.save_map_name = copy.copy(name)
        if name == '':
            messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['messagebox_text']['message']['text_1'])
            return
        self.update()
        # 創建環境
        if self.mota_name == 'node' or self.mota_name == 'node_1' or self.mota_name == 'node_2':
            if self.re:
                self.check_re()
            self.env.build_env(name)

        elif self.mota_name == 'maze':
            self.env.build_env(name)

        self.env_name = name
        self.map_btn.place_forget()
        self.map_lab.place_forget()
        self.select_map_btn.place_forget()
        self.comb.place_forget()

        self.comb.set('')
        self.build_map(name)
        self.scr.insert(tk.END, str(name) + Show_learning_string['scr_text']['create_mota']['text_5'])
        self.select_algorithm()
#============================================================================
#  選擇地圖環境 及 建構環境
#============================================================================
    def build_map(self, map_name):
        # 目前所在的樓層
        self.floor_index = 0
        # 將地圖貼在畫布上
        if self.mota_name == 'maze':
            self.floor_max = self.env.draw_map('pictures/MT_')
            self.image_file = tk.PhotoImage(file='pictures/MT_{:d}.png'.format(self.floor_index))
            self.canvas.delete('all')
            self.image = self.canvas.create_image(0, 0, anchor='nw', image = self.image_file)
            self.canvas.pack()

        elif self.mota_name == 'node' or self.mota_name == 'node_1'or self.mota_name == 'node_2':
            self.floor_max = self.env.get_layer()
            self.env.create_map()
            self.env.create_nodes()
            self.env.build_anima_frame()

        self.change_floor_status()
        self.tileImage = Image.open('pictures/baseTile.png')
        self.player_icon = self.tileImage.crop((32*7, 0, 32*8, 32))
        self.floor_icon = self.tileImage.crop((0, 0, 32, 32))
        self.player_image_file = ImageTk.PhotoImage(self.player_icon)
        self.floor_image_file = ImageTk.PhotoImage(self.floor_icon)
        self.pre_player_pos = self.find_player_pos()
        self.player_image = None
#============================================================================
#  選擇演算法
#============================================================================
    def select_algorithm(self):
        self.reset_mode.place_forget()
        self.reset_algorithm.place_forget()
        self.reset_env.place_forget()

        self.algorithm_lab.place(x=10, y=100, relx=0.6, relwidth=0.095)
        self.comb.config(values=self.ALGORITHM)
        self.comb.place(x=10, y=100, relx=0.7, relwidth=0.18)
        self.rounds_lab.place(x=10, y=130, relx=0.6, relwidth=0.095)
        self.en.place(x=10, y=130, relx=0.7, relwidth=0.18)
        if self.mota_name == 'node_1':
            self.en.config(state='disable')
        self.select_map_btn.config(text= Show_learning_string['Button_text']['universal']['text_1'], bg='#C4A98B',image = self.map_btn_img ,command=self.start_to_learn,state=tk.NORMAL)
        self.select_map_btn.place(y=115, width=-14, height=35, relx=0.9, relwidth=0.11)
        self.back_btn = tk.Button(self)
        Q_03 = tk.PhotoImage(file ='pictures/Q_03.png')
        self.Q_03 = Q_03.subsample(4,4)
        self.back_btn.config(text= self.env_name ,bg='#C4A98B',image = self.Q_03,command=self.back_select_env,compound = tk.RIGHT)
        self.back_btn.place(y=62, width=200, height=35, relx=0.615)
        self.back_btn['font'] = self.ChiFont

        self.animation_btn = tk.Button(self)
        self.animation_btn.config(text = Show_learning_string['Button_text']['universal']['text_5'],bg='#C4A98B',command= lambda:self.animetion_switch(),state='disable')
        self.animation_btn.place(y=62, width=100, height=35, relx=0.87)
        self.animation_btn['font'] = self.ChiFont
        self.switch_start = True
        self.switch_end = True

        Tooltip(self.back_btn, text=Show_learning_string['tips_text']['text_11'])
#============================================================================
#  返回環境
#============================================================================
    def back_select_env(self):
        if self.mota_name == 'node':
            self.re_build_node_env()
        elif self.mota_name == 'maze':
            self.canvas.delete('all')
            self.select_map_name()
        elif self.mota_name == 'node_1':
            self.re_build_node_env()
        elif self.mota_name == 'node_2':
            self.re_build_node_env()
        self.back_btn.place_forget()
        self.en.place_forget()
        self.select_map_btn.place(x =-220, y=100, width=-30, relwidth=0.15)
        self.algorithm_lab.place_forget()
        self.rounds_lab.place_forget()
        self.animation_btn.place_forget()
        self.up_btn.config(text= Show_learning_string['Button_text']['build_win']['text_2'], state='disabled')
        self.down_btn.config(text= Show_learning_string['Button_text']['build_win']['text_1'], state='disabled')
#============================================================================
#  開始學習按鈕
#============================================================================
    def start_to_learn(self):
        # 取得選項文字
        self.tab1.clear()
        name = self.comb.get()
        rounds = self.en.get()

        if self.mota_name == 'node' or self.mota_name == 'maze'or self.mota_name == 'node_2':
            if name == '' or rounds == '':
                messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['messagebox_text']['message']['text_2'])
                return
        if self.mota_name == 'node_1' :
            if name == '':
                messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['messagebox_text']['message']['text_3'])
                return
        # 隱藏按鈕
        self.comb.config(state='disable')
        self.en.config(state='disable')
        self.select_map_btn.config(state='disable')
        self.back_btn.config(state='disabled')
        # 停止鍵
        def interrupt():
            self.stop = True
            self.select_map_btn.config(state='disabled')
            self.scr.insert(tk.END, Show_learning_string['scr_text']['another']['text_1'] ,'scr_red')
            self.scr.tag_config('scr_red',foreground='red')
        self.stop = False
        stop_img = tk.PhotoImage(file ='pictures/Stop.png')
        self.stop_img = stop_img.subsample(4,4)
        self.select_map_btn.config(text=Show_learning_string['Button_text']['universal']['text_3'], state=tk.NORMAL,image = self.stop_img, command=interrupt, compound = tk.LEFT)
        if self.mota_name == 'node_1' :
            self.select_map_btn.config(state='disabled')

        # 從第一層開始訓練
        if self.mota_name == 'maze':
            global global_image
            self.canvas.delete(self.image)
            global_image = tk.PhotoImage(file='pictures/MT_0.png')
            self.image = self.canvas.create_image(0, 0, anchor='nw', image = global_image)
            if self.floor_max > 1:
                self.up_btn.config(text = Show_learning_string['Button_text']['universal']['text_4'] , state='disabled')
            else:
                self.up_btn.config(text= Show_learning_string['Button_text']['build_win']['text_2'], state='disabled')
            self.down_btn.config(text= Show_learning_string['Button_text']['build_win']['text_1'], state='disabled')

            self.player_image = self.canvas.create_image(self.pre_player_pos[2]*32, self.pre_player_pos[1]*32, anchor='nw', image=self.player_image_file)

        elif self.mota_name == 'node' or self.mota_name == 'node_1' or self.mota_name == 'node_2':
            self.env.anima_frame.change_floor(0)
            if self.floor_max > 1:
                self.up_btn.config(text= Show_learning_string['Button_text']['universal']['text_4'], state='disabled')
            else:
                self.up_btn.config(text = Show_learning_string['Button_text']['build_win']['text_2'], state='disabled')
            self.down_btn.config(text= Show_learning_string['Button_text']['build_win']['text_1'], state='disabled')

        # 開始訓練
        if self.mota_name == 'node' or self.mota_name == 'maze' or self.mota_name == 'node_2':
            # 顯示演算法
            self.scr.insert(tk.END, str(name) + Show_learning_string['scr_text']['another']['text_2'])
            # 顯示訓練回數
            self.scr.insert(tk.END, Show_learning_string['scr_text']['another']['text_3'] + str(rounds) + Show_learning_string['scr_text']['another']['text_4'])
            self.scr.insert(tk.END, f'剩下{rounds}回\n')
            # 顯示最高分數
            self.scr.insert(tk.END, Show_learning_string['scr_text']['another']['text_5'])
            self.scr.insert(tk.END, '\n')
            if self.mota_name == 'node' or self.mota_name == 'maze':
                # 迷宮與節點算法
                self.learning_for_mota(name, int(rounds))
            elif self.mota_name == 'node_2':
                # 模型預測
                self.MCTSfDm_Predict(name,self.env,int(rounds))
        # 模型預測
        if self.mota_name == 'node_1':
            self.model_predict(self.env, name)
#============================================================================
#  訓練動畫按鈕
#============================================================================
    def animetion_switch(self):
        if self.switch_start:
            self.animation_btn.config(text= Show_learning_string['Button_text']['universal']['text_2'])
            if self.mota_name == 'node':
                self.agent.off_animation()
            if self.mota_name == 'node_2':
                self.agent.off_model_animation()
        else:
            self.animation_btn.config(text= Show_learning_string['Button_text']['universal']['text_5'])
            if self.mota_name == 'node':
                self.agent.on_animation()
            if self.mota_name == 'node_2':
                self.agent.on_model_animation()

            if not self.switch_end:
                self.switch_end = True
        self.switch_start = not self.switch_start
#============================================================================
#  學習演算法
#============================================================================
    def learning_for_mota(self, algorithm_name, rounds):
        startTime = time.perf_counter()
        self.animation_btn.config(state='normal')
        max_hp = 0
        File_map = Save_File(self.env_name, algorithm_name)
        self.agent = learning(self.env, algorithm_name, int(rounds))
        if self.mota_name == 'maze':
            for ob, done, episode, score in self.agent.maze_mota_learning():

                if self.stop:
                    self.player_move(ob, True)
                    break
                if done:
                    File_map.sampling(score)
                    self.scr.delete('end-5l', 'end-1l')
                    self.scr.insert(tk.END, str(algorithm_name) + Show_learning_string['scr_text']['another']['text_2'])
                    self.scr.insert(tk.END, Show_learning_string['scr_text']['another']['text_3'] + str(rounds) +  Show_learning_string['scr_text']['another']['text_4'])
                    self.scr.insert(tk.END, f'剩下{rounds - episode - 1}回\n')
                    if score > max_hp:
                        max_hp = score
                    self.scr.insert(tk.END, f'目前最高分數 : {max_hp}\n')
                    self.scr.update()
                if not done:
                    self.scr.delete('end-2l', 'end-1l')
                endTime = time.perf_counter()
                get_time = endTime - startTime
                self.scr.insert(tk.END, '花費時間:{:0.3f}\n'.format(get_time))
                self.scr.update()

                if self.switch_start == True:
                    self.player_move(ob, done)

            data = File_map.get_data()
            fix = File_map.get_figure(data, color='coral', #color='steelblue'/'coral'
                                figsize=(7.5, 6), facecolor='lavender', dpi=64)
            fix_map = FigureCanvasTkAgg(fix, self.tab1.contents)
            fix_map.get_tk_widget().pack()

            self.scr.insert(tk.END, Show_learning_string['scr_text']['another']['text_6'],'scr_red')
            self.scr.tag_config('scr_red',foreground='red')

        elif self.mota_name == 'node':
            save_max_action = []
            for ob, done, episode, score, save_action in self.agent.Node_mota_learning():
                if self.stop:
                    self.env.reset()
                    self.update()
                    time.sleep(0.1)
                    break
                if done:
                    File_map.sampling(score)
                    self.scr.delete('end-5l', 'end-1l')
                    self.scr.insert(tk.END, str(algorithm_name) + Show_learning_string['scr_text']['another']['text_2'])
                    self.scr.insert(tk.END, Show_learning_string['scr_text']['another']['text_3'] + str(rounds) +  Show_learning_string['scr_text']['another']['text_4'])
                    self.scr.insert(tk.END, f'剩下{rounds - episode - 1}回\n')
                    if score > max_hp:
                        max_hp = score
                        save_max_action = save_action
                    self.scr.insert(tk.END, f'目前最高分數 : {max_hp}\n')
                    self.scr.update()
                if not done:
                    self.scr.delete('end-2l', 'end-1l')
                endTime = time.perf_counter()
                get_time = endTime - startTime
                self.scr.insert(tk.END, '花費時間:{:0.3f}\n'.format(get_time))
                self.scr.update()
                self.update()
                if self.agent.switch ==True:
                    time.sleep(0.1)

            data = File_map.get_data()
            fix = File_map.get_figure(data, color='coral',
                                figsize=(7.5, 6), facecolor='lavender', dpi=64)

            fix_map = FigureCanvasTkAgg(fix, self.tab1.contents)
            fix_map.get_tk_widget().pack()
            if max_hp > 0 and not self.stop:
                # 儲存通關路徑
                index = 0
                for file_name in os.listdir('player_route/'):
                    map_name = file_name.split('_v')[0]
                    if map_name == self.save_map_name:
                        index += 1
                print(save_max_action)
                np.save('player_route/' + self.save_map_name + '_v' + str(index) + '_h' + str(max_hp), np.array(save_max_action))
                # 儲存通關路徑提示
                self.scr.insert(tk.END, Show_learning_string['scr_text']['another']['text_8'],'save_action')
                self.scr.tag_config('save_action',foreground='red')
            else:
                # 儲存通關路徑提示
                self.scr.insert(tk.END, Show_learning_string['scr_text']['another']['text_9'],'save_action')
                self.scr.tag_config('save_action',foreground='red')
            # 結束訓練提示
            self.scr.insert(tk.END, Show_learning_string['scr_text']['another']['text_6'],'scr_red')
            self.scr.tag_config('scr_red',foreground='red')
        self.reset_select()
#============================================================================
#  模型預測
#============================================================================
    def model_predict(self, env, model_name):
        self.update()
        self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_1'] + model_name+'\n\n')
        model = Demonstration_predict(env, model_name)
        for time_, ending in model.predict_environment():
            self.scr.delete('end-2l', 'end-1l')
            self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_2'].format(time_) +'\n')
            self.update()
            time.sleep(0.1)
        self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_3'] + ending +'\n')
        self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_4'] + str(env.player.hp)+'\n')
        self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_5'] + str(len(env.observation))+'\n')
        self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_6'] + str(np.max([env.n2p[n][0] for n in env.observation]))+'\n')
        if ending == 'clear':
            self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_24'], 'model_predict')
            self.scr.tag_config('model_predict',foreground='red')
        else:
            self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_25'], 'model_predict')
            self.scr.tag_config('model_predict',foreground='red')
        self.reset_select()
#============================================================================
#  MCTSfDm model
#============================================================================
    def MCTSfDm_Predict(self, model_name, env, rounds):
        save_max_action = []
        startTime = time.perf_counter()
        self.animation_btn.config(state='normal')
        self.update()
        File_map = Save_File(self.env_name, model_name)
        self.agent = MCTSfDm_learning(self.env, model_name, rounds)
        max_hp = 0
        for ob, done, episode, score, save_action in self.agent.Learning_Algorithm():
            if self.stop:
                self.env.reset()
                self.update()
                time.sleep(0.1)
                break
            if done:
                File_map.sampling(score)
                self.scr.delete('end-5l', 'end-1l')
                self.scr.insert(tk.END, model_name + Show_learning_string['scr_text']['another']['text_2'])
                self.scr.insert(tk.END, Show_learning_string['scr_text']['another']['text_3'] + str(rounds) + Show_learning_string['scr_text']['another']['text_4'])
                self.scr.insert(tk.END, f'剩下{rounds - episode - 1}回\n')
                if score > max_hp:
                    max_hp = score
                    save_max_action = save_action
                self.scr.insert(tk.END, f'目前最高分數 : {max_hp}\n')
                self.scr.update()
                time.sleep(0.1)
            if not done:
                self.scr.delete('end-2l', 'end-1l')
            endTime = time.perf_counter()
            get_time = endTime - startTime
            self.scr.insert(tk.END, '花費時間:{:0.3f}\n'.format(get_time))
            self.scr.update()

            if self.agent.model_switch ==True:
                time.sleep(0.1)
        data = File_map.get_data()
        fix = File_map.get_figure(data, color='coral',
                            figsize=(7.5, 6), facecolor='lavender', dpi=64)
        fix_map = FigureCanvasTkAgg(fix, self.tab1.contents)
        fix_map.get_tk_widget().pack()
        if max_hp > 0 and not self.stop:
            # 儲存通關路徑
            index = 0
            for file_name in os.listdir('player_route/'):
                map_name = file_name.split('_v')[0]
                if map_name == self.save_map_name:
                    index += 1
            print(save_max_action)
            np.save('player_route/' + self.save_map_name + '_v' + str(index) + '_h' + str(max_hp), np.array(save_max_action))
            # 儲存通關路徑提示
            self.scr.insert(tk.END, Show_learning_string['scr_text']['another']['text_8'],'save_action')
            self.scr.tag_config('save_action',foreground='red')
        else:
            # 儲存通關路徑提示
            if self.stop:
                self.scr.insert(tk.END, Show_learning_string['scr_text']['another']['text_10'],'save_action')
                self.scr.tag_config('save_action',foreground='red')
            else:
                self.scr.insert(tk.END, Show_learning_string['scr_text']['another']['text_9'],'save_action')
                self.scr.tag_config('save_action',foreground='red')
        self.reset_select()
#============================================================================
#  上樓按鈕
#============================================================================
    def floor_up(self):
        self.floor_index += 1
        global global_image
        if self.mota_name == 'node' or self.mota_name == 'node_1'or self.mota_name == 'node_1':
            self.env.anima_frame.change_floor(self.floor_index)
        elif self.mota_name == 'maze':
            self.canvas.delete(self.image)
            global_image = tk.PhotoImage(file='pictures/MT_{:d}.png'.format(self.floor_index))
            self.image = self.canvas.create_image(0, 0, anchor='nw', image = global_image)
        self.change_floor_status()
#============================================================================
#  下樓按鈕
#============================================================================
    def floor_down(self):
        self.floor_index -= 1
        global global_image
        if self.mota_name == 'node' or self.mota_name == 'node_1'or self.mota_name == 'node_2':
            self.env.anima_frame.change_floor(self.floor_index)
        elif self.mota_name == 'maze':
            self.canvas.delete(self.image)
            global_image = tk.PhotoImage(file='pictures/MT_{:d}.png'.format(self.floor_index))
            self.image = self.canvas.create_image(0, 0, anchor='nw', image = global_image)
        self.change_floor_status()
#============================================================================
#  切換樓層狀態
#============================================================================
    def change_floor_status(self):
        if self.floor_index > 0:
            self.down_btn.config(text=f'下一層({self.floor_index - 1}F)', state='normal')
        else:
            self.down_btn.config(text='下一層', state='disabled')
        if self.floor_max - self.floor_index > 1:
            self.up_btn.config(text=f'上一層({self.floor_index + 1}F)', state='normal')
        else:
            self.up_btn.config(text='上一層', state='disabled')
#============================================================================
#  找出代理位置
#============================================================================
    def find_player_pos(self):
        player_id = None
        self.env_map = np.array(self.env.env_data['floors']['map'])
        for key, value in self.env.env_data['maps'].items():
            if value['id'] == 'player':
                if player_id is None:
                    player_id = key
        player_pos = np.argwhere(self.env_map==player_id)[0]
        return player_pos
#============================================================================
#  重新選擇功能
#============================================================================
    def reset_select(self):
        self.result_chart_btn.config(state='normal')
        self.animation_btn.place_forget()
        self.select_map_btn.place_forget()
        self.algorithm_lab.place_forget()
        self.rounds_lab.place_forget()
        self.en.place_forget()
        self.comb.place_forget()
        self.back_btn.place_forget()

        self.env.reset()
        Q_04 = tk.PhotoImage(file ='pictures/Q_04.png')
        self.Q_04 = Q_04.subsample(3,3)

        self.reset_algorithm.config(text= Show_learning_string['Button_text']['reset']['text_1'], bg='#C4A98B',image = self.Q_04,command=self.select_algorithm,state=tk.NORMAL,compound = tk.LEFT)
        self.reset_algorithm.place(y=65, width=-14, height=27, relx=0.66, relwidth=0.3)

        self.reset_algorithm['font'] = self.ChiFont

        if self.mota_name == 'node' or self.mota_name == 'node_1'or self.mota_name == 'node_2':
            self.reset_env.config(text= Show_learning_string['Button_text']['reset']['text_2'], bg='#C4A98B',image = self.Q_04,command=self.re_build_node_env,compound = tk.LEFT)
        elif self.mota_name == 'maze':
            self.reset_env.config(text= Show_learning_string['Button_text']['reset']['text_2'], bg='#C4A98B',image = self.Q_04,command=self.back_select_env,compound = tk.LEFT)
        self.reset_env['font'] = self.ChiFont
        self.reset_env.place(y=95, width=-14, height=27, relx=0.66, relwidth=0.3)

        self.reset_mode.config(text =Show_learning_string['Button_text']['reset']['text_3'], bg='#C4A98B', image =self.Q_04, command=self.reset_all,compound = tk.LEFT)
        self.reset_mode.place(y=125, width=-14, height=27, relx=0.66, relwidth=0.3)
        self.reset_mode['font'] = self.ChiFont

        self.comb.config(state=tk.NORMAL)
        self.en.config(state=tk.NORMAL)
#============================================================================
#  重製重新選擇功能
#============================================================================
    def reset_all(self):
        self.reset_algorithm.destroy()
        self.reset_env.destroy()
        self.reset_mode.destroy()
        self.choose_mode()
        self.nb.destroy()
        self.frm.destroy()
        self.labframe.destroy()
        self.imLabel.destroy()
        self.down_btn.destroy()
        self.up_btn.destroy()
        self.result_chart_btn.destroy()
#============================================================================
#  迷宮版角色移動
#============================================================================
    def player_move(self, ob, done):
        if not done:
            self.post_player_pos = ob[:3]
            # 樓層移動
            if self.post_player_pos[0] != self.pre_player_pos[0]:
                self.canvas.delete('all')
                self.env.draw_map_layer(self.pre_player_pos[0])
                self.image_file = tk.PhotoImage(file='pictures/MT_{:d}.png'.format(self.post_player_pos[0]))
                self.image = self.canvas.create_image(0, 0, anchor='nw', image = self.image_file)

            # 事件觸發
            player_position = self.env_map[self.post_player_pos[0]][self.post_player_pos[1]][self.post_player_pos[2]]
            if player_position != 0 and 'Shop' not in self.env.env_data['maps'][player_position]['id']:
                self.floor_image = self.canvas.create_image(self.post_player_pos[2]*32, self.post_player_pos[1]*32, anchor='nw', image=self.floor_image_file)
            # 玩家移動
            self.canvas.delete(self.player_image)
            self.player_image = self.canvas.create_image(self.post_player_pos[2]*32, self.post_player_pos[1]*32, anchor='nw', image=self.player_image_file)
            self.pre_player_pos = self.post_player_pos
        else:
            # 重製畫布
            self.pre_player_pos = self.find_player_pos()
            self.canvas.delete('all')
            # 重建地圖前暫停一段時間才不會出現PermissionError: [Errno 13] Permission denied
            # 錯誤訊息是寫入檔案時被其他程序開啟
            time.sleep(1)
            self.floor_max = self.env.draw_map('pictures/MT_')
            self.image_file = tk.PhotoImage(file='pictures/MT_0.png')
            self.image = self.canvas.create_image(0, 0, anchor='nw', image = self.image_file)
            self.player_image = self.canvas.create_image(self.pre_player_pos[2]*32, self.pre_player_pos[1]*32, anchor='nw', image=self.player_image_file)
            self.canvas.pack()
        self.canvas.update()
#============================================================================
#  模仿學習建立選擇演算法及參數
#============================================================================
    def create_Demonstration(self):
        self.build_node_2_window()
        win = Demonstration_window()
        win.mainloop()
#============================================================================
#  MCTSfDm算法
#============================================================================
    def MCTSfDm_env(self):
        self.build_node_3_window()
        win = MCTSfDm_window()
        win.mainloop()
#============================================================================
#  重建節點版環境
#============================================================================
    def re_build_node_env(self):
        self.re = 1
        self.frm.destroy()
        self.frm = ttk.LabelFrame(self, text='Map')
        self.frm.place(height=-40, relwidth=0.6, relheight=1.0)
        self.new_env = Mota(self.frm)

        self.nb = ttk.Notebook(self)
        self.tab1 = Frame_Scrollbar(self.nb, bg='lavender')
        self.nb.add(self.tab1, text=Show_learning_string['Button_text']['build_win']['text_3'])

        del self.env.anima_frame
        gc.collect()
        del self.env
        gc.collect()
        #del self.
        self.select_map_name()
#============================================================================
#  重建節點版環境的畫布
#============================================================================
    def check_re(self):
        self.re = 0
        self.env = self.new_env
        del self.new_env
        gc.collect()
#============================================================================
#  結束應用程式
#============================================================================
    def Reset_GUI(self):
        self.destroy()

if __name__ == '__main__':
    print(os.getcwd())
    window = show_learning()
    window.mainloop()
