from import_tool import *
from String_ import *
from window_resize import win_resize
from ToolTip import Tooltip
from trainData import Feature_Data
from model_function import Demonstration_model
import threading
import time
import database as db

class Demonstration_window(tk.Toplevel):
    def __init__(self):
        super(Demonstration_window, self).__init__()
        winWidth = 806
        winHeight = 606
        self.title(Show_learning_string['Initialize']['title1'])
        self.x = (self.winfo_screenwidth() - winWidth) // 2
        self.y = (self.winfo_screenheight() - winHeight) // 3
        win_resize(self)

        self.done = False

        self.ChiFont = font.Font(family=Show_learning_string['Initialize']['font_style_Chinese'], weight=tkFont.BOLD)

        self.geometry('%sx%s+%s+%s' % (700, 500, self.x, self.y))
        self.Demonstration_env_name= db.FILE_NAME
        self.labframe = tk.LabelFrame(self, text=Show_learning_string['label_text']['select_env']['text_4'] ,bg='#C4A98B')
        self.labframe.place(x=10, y=160, width=-20, height=-170, relx=0.6, relwidth=0.4, relheight=1.0)
        self.scr = scrolledtext.ScrolledText(self.labframe)
        self.scr.pack(fill='both', expand=True)
        self.select_model_lab = tk.Label(self,text =Show_learning_string['label_text']['demo']['text_1'], bg='#A07558',font=(Show_learning_string['Initialize']['font_style_Chinese'], 14, tkFont.BOLD))
        self.select_model_lab.place(x=-210, y=10, width=110, height=30, relx=0.7)

        self.select_train_env_comb = ttk.Combobox(self, state='readonly')
        self.select_train_env_comb.place(x=-250, y=70, width=-45, height=20, relx=0.75, relwidth=0.25)
        self.select_train_env_comb.config(values=self.Demonstration_env_name)

        self.select_test_env_comb = ttk.Combobox(self, state='readonly')
        self.select_test_env_comb.place(x=-250, y=120, width=-45, height=20, relx=0.75, relwidth=0.25)
        self.select_test_env_comb.config(values=self.Demonstration_env_name)

        self.select_train_env_lab = tk.Label(self, text =Show_learning_string['label_text']['demo']['text_2'], bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 14, tkFont.BOLD))
        self.select_train_env_lab.place(x=-300, y=70, width=110, height=20, relx=0.65)

        self.select_test_env_lab = tk.Label(self, text =Show_learning_string['label_text']['demo']['text_3'], bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 14, tkFont.BOLD))
        self.select_test_env_lab.place(x=-300, y=120, width=110, height=20, relx=0.65)

        self.route_lab = tk.Label(self, text = Show_learning_string['label_text']['demo']['text_5'], bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 14, tkFont.BOLD))
        self.route_lab.place(x=-300, y=170, width=110, height=20, relx=0.65)
        self.route_comb = ttk.Combobox(self, state='readonly')
        self.route_comb.place(x=-250, y=170, width=-45, height=20, relx=0.75, relwidth=0.25)
        self.route_comb.config(values=['單一地圖', '強化地圖', '強化泛化'])
        self.route_comb.current(0)

        self.quest_btn = tk.Button(self)
        Q_05 = tk.PhotoImage(file ='pictures/Q_05.png')
        self.Q_05 = Q_05.subsample(15,15)
        self.quest_btn.config(bg='#C4A98B',image = self.Q_05, relief=tk.RIDGE)
        self.quest_btn.place(x=-450, y=20, width=40, height=40, relx=0.7)
        a=Tooltip(self.quest_btn, text=Show_learning_string['tips_text']['text_19'], wraplength=400)

#============================================================================
#  模型訓練
#============================================================================
        def model(self):
            self.done = False
            t1 = threading.Thread(target = self.updata_scr)
            t1.setDaemon(True)
            t1.start()

            model_name = self.select_model_comb.get()
            if model_name == '':
                return messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['messagebox_text']['message']['text_3'])

            self.scr.delete('end-2l', 'end-1l')
            self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_7'] + model_name + '\n')
            self.select_model_comb.place_forget()
            self.model_lab.place_forget()
            self.regularlization = ['None','L1','L2']
            self.svm_kernel = ['linear', 'rbf', 'sigmoid', 'precomputed']
            self.knn_weights = ['uniform','distance']
            self.knn_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

            # self.quest_btn = tk.Button(self)

            # Q_05 = tk.PhotoImage(file ='pictures/Q_05.png')
            # self.Q_05 = Q_05.subsample(15,15)
            # self.quest_btn.config(bg='#C4A98B',image = self.Q_05, relief=tk.RIDGE)
            # self.quest_btn.place(x=-450, y=20, width=40, height=40, relx=0.7)
#============================================================================
#  感知機
#============================================================================
            if model_name == 'Multilayer Perceptron':
                self.update()
                self.mlp_dense_en1 = ttk.Entry(self)
                self.mlp_dense_en1.place(x=-200, y=60, relx=0.7, relwidth=0.18)
                self.mlp_dense_en2 = ttk.Entry(self)
                self.mlp_dense_en2.place(x=-200, y=90, relx=0.7, relwidth=0.18)
                self.mlp_dense_en3 = ttk.Entry(self)
                self.mlp_dense_en3.place(x=-200, y=120, relx=0.7, relwidth=0.18)
                self.mlp_epochs_en = ttk.Entry(self)
                self.mlp_epochs_en.place(x=-200, y=150, relx=0.7, relwidth=0.18)
                self.mlp_dense_lab1 = tk.Label(self, text =Show_learning_string['label_text']['mlp']['text_1'], bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.mlp_dense_lab1.place(x=-320, y=60, relx=0.7, relwidth=0.15)

                self.mlp_dense_lab2 = tk.Label(self, text =Show_learning_string['label_text']['mlp']['text_2'], bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.mlp_dense_lab2.place(x=-320, y=90, relx=0.7, relwidth=0.15)
                self.mlp_dense_lab3 = tk.Label(self, text =Show_learning_string['label_text']['mlp']['text_3'], bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.mlp_dense_lab3.place(x=-320, y=120, relx=0.7, relwidth=0.15)
                self.mlp_dense_lab3 = tk.Label(self, text =Show_learning_string['label_text']['mlp']['text_4'] ,bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.mlp_dense_lab3.place(x=-320, y=150, relx=0.7, relwidth=0.15)

                self.regularlization_lab = tk.Label(self, text =Show_learning_string['label_text']['mlp']['text_5'], bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.regularlization_lab.place(x=-30, y=60, relx=0.7, relwidth=0.12)

                self.regularlization_comb = ttk.Combobox(self, state='readonly')
                self.regularlization_comb.place(x=10, y=60, width=-45, height=20, relx=0.75, relwidth=0.25)
                self.regularlization_comb.config(values=self.regularlization)
                self.regularlization_comb.current(0)

                Tooltip(self.quest_btn, text=Show_learning_string['tips_text']['text_12'], wraplength=200)

                def Get_mlp_Hyperparameter(self):
                    def threading_1():
                        while not self.done:
                            x1 = self.mlp_dense_en1.get()
                            x2 = self.mlp_dense_en2.get()
                            x3 = self.mlp_dense_en3.get()
                            x4 = self.mlp_epochs_en.get()
                            df = self.regularlization_comb.get()
                            if x1 == '' or x2 == '' or x3 == '' or x4 == '' :
                                return messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['messagebox_text']['message']['text_4'])

                            model = Demonstration_model(df)
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['another']['text_7'])
                            self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_23'])

                            train_score, test_score = model.mlp_model(int(x1),int(x2),int(x3),int(x4))

                            self.scr.delete('end-2l', 'end-1l')
                            self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_8'])
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_9'].format(train_score))
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_10'].format(test_score))
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_26'], 'mlp_done')
                            self.scr.tag_config('mlp_done', foreground='red')
                            self.done = True
                            self.select_all_btn.config(state='disabled')
                    a1 = threading.Thread(target = threading_1)
                    a1.setDaemon(True)
                    a1.start()
                self.select_all_btn.config(text =Show_learning_string['Button_text']['universal']['text_1'], command = lambda :Get_mlp_Hyperparameter(self))
#============================================================================
#  隨機森林
#============================================================================
            if model_name == 'Random Forests':
                self.rfc_en1 = ttk.Entry(self)
                self.rfc_en1.place(x=-200, y=60, relx=0.7, relwidth=0.18)
                self.rfc_en2 = ttk.Entry(self)
                self.rfc_en2.place(x=-200, y=90, relx=0.7, relwidth=0.18)
                self.rfc_en3 = ttk.Entry(self)
                self.rfc_en3.place(x=-200, y=120, relx=0.7, relwidth=0.18)

                self.rfc_lab1 = tk.Label(self, text =Show_learning_string['label_text']['rfc']['text_1'],bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.rfc_lab1.place(x=-320, y=60, relx=0.7, relwidth=0.15)
                self.rfc_lab2 = tk.Label(self, text =Show_learning_string['label_text']['rfc']['text_2'],bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.rfc_lab2.place(x=-320, y=90, relx=0.7, relwidth=0.15)
                self.rfc_lab3 = tk.Label(self, text =Show_learning_string['label_text']['rfc']['text_3'],bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.rfc_lab3.place(x=-320, y=120, relx=0.7, relwidth=0.17)

                Tooltip(self.quest_btn, text = Show_learning_string['tips_text']['text_13'], wraplength=200)

                def Get_rfc_Hyperparameter(self):
                    def threading_2():
                        while not self.done:
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['another']['text_7'])
                            self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_23'])
                            x1 = self.rfc_en1.get()
                            x2 = self.rfc_en2.get()
                            x3 = self.rfc_en3.get()
                            if x1 == '' or x2 == '' or x3 == '':
                                return messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['messagebox_text']['message']['text_4'])

                            model = Demonstration_model('None')
                            train_score, test_score = model.rfc_modle(int(x1), int(x2), int(x3))
                            self.scr.delete('end-2l', 'end-1l')
                            self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_8'])
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_9'].format(train_score))
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_10'].format(test_score))
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_26'], 'rfc_done')
                            self.scr.tag_config('rfc_done', foreground='red')
                            self.done = True
                            self.select_all_btn.config(state='disabled')

                    a2 = threading.Thread(target = threading_2)
                    a2.setDaemon(True)
                    a2.start()
                self.select_all_btn.config(text =Show_learning_string['Button_text']['universal']['text_1'], command = lambda :Get_rfc_Hyperparameter(self))
#============================================================================
#  梯度提升樹
#============================================================================
            if model_name == 'BoostTree':
                self.BT_en1 = ttk.Entry(self)
                self.BT_en1.place(x=-200, y=60, relx=0.7, relwidth=0.18)
                self.BT_en2 = ttk.Entry(self)
                self.BT_en2.place(x=-200, y=90, relx=0.7, relwidth=0.18)
                self.BT_en3 = ttk.Entry(self)
                self.BT_en3.place(x=-200, y=120, relx=0.7, relwidth=0.18)

                self.BT_lab1 = tk.Label(self, text = Show_learning_string['label_text']['BoostTree']['text_1'],bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.BT_lab1.place(x=-320, y=60, relx=0.7, relwidth=0.15)
                self.BT_lab2 = tk.Label(self, text = Show_learning_string['label_text']['BoostTree']['text_2'],bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.BT_lab2.place(x=-320, y=90, relx=0.7, relwidth=0.15)
                self.BT_lab3 = tk.Label(self, text = Show_learning_string['label_text']['BoostTree']['text_3'],bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.BT_lab3.place(x=-320, y=120, relx=0.7, relwidth=0.15)

                Tooltip(self.quest_btn, text = Show_learning_string['tips_text']['text_14'] , wraplength=300)

                def Get_BT_Hyperparameter(self):
                    def threading_3():
                        while not self.done:
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['another']['text_7'])
                            self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_23'])
                            x1 = self.BT_en1.get()
                            x2 = self.BT_en2.get()
                            x3 = self.BT_en3.get()
                            if x1 == '' or x2 == '' or x3 == '':
                                return messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['messagebox_text']['message']['text_4'])

                            model = Demonstration_model('None')
                            train_score, test_score = model.BoostTree_model(int(x1), float(x2), int(x3))
                            self.scr.delete('end-2l', 'end-1l')
                            self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_8'])
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_9'].format(train_score))
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_10'].format(test_score))
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_26'], 'BT_done')
                            self.scr.tag_config('BT_done', foreground='red')
                            self.done = True
                            self.select_all_btn.config(state='disabled')
                    a3 = threading.Thread(target = threading_3)
                    a3.setDaemon(True)
                    a3.start()
                self.select_all_btn.config(text = Show_learning_string['Button_text']['universal']['text_1'], command = lambda :Get_BT_Hyperparameter(self))
#============================================================================
#  SVM
#============================================================================
            if model_name == 'Support Vector Machine':
                self.svm_lab = tk.Label(self, text = Show_learning_string['label_text']['svm']['text_1'], bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.svm_lab.place(x=-320, y=60, width=150, height=20, relx=0.65)
                self.svm_comb = ttk.Combobox(self, state='readonly')
                self.svm_comb.place(x=-250, y=60, width=-45, height=20, relx=0.75, relwidth=0.25)
                self.svm_comb.config(values=self.svm_kernel)
                self.svm_comb.current(2)

                self.regularlization_lab = tk.Label(self, text = Show_learning_string['label_text']['svm']['text_2'], bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.regularlization_lab.place(x=-30, y=60, relx=0.7, relwidth=0.12)

                self.regularlization_comb = ttk.Combobox(self, state='readonly')
                self.regularlization_comb.place(x=10, y=60, width=-45, height=20, relx=0.75, relwidth=0.25)
                self.regularlization_comb.config(values=self.regularlization)
                self.regularlization_comb.current(0)

                Tooltip(self.quest_btn, text= Show_learning_string['tips_text']['text_15'], wraplength=200)

                def Get_svm_Hyperparameter(self):
                    def threading_4():
                        while not self.done:
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['another']['text_7'])
                            self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_23'])
                            x1 = self.svm_comb.get()
                            if x1 == '' :
                                return messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['messagebox_text']['message']['text_4'])
                            df = self.regularlization_comb.get()
                            model = Demonstration_model(df)
                            train_score, test_score = model.svm_model(x1)
                            self.scr.delete('end-2l', 'end-1l')
                            self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_8'])
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_9'].format(train_score))
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_10'].format(test_score))
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_26'], 'svm_done')
                            self.scr.tag_config('svm_done', foreground='red')
                            self.done = True
                            self.select_all_btn.config(state='disabled')
                    a4 = threading.Thread(target = threading_4)
                    a4.setDaemon(True)
                    a4.start()
                self.select_all_btn.config(text = Show_learning_string['Button_text']['universal']['text_1'], command = lambda :Get_svm_Hyperparameter(self))
#============================================================================
#  KNN
#============================================================================
            if model_name == 'K Nearest Neighbor':
                self.knn_lab1 = tk.Label(self, text = Show_learning_string['label_text']['knn']['text_1'],bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.knn_lab1.place(x=-320, y=60, width=150, height=20, relx=0.65)
                self.knn_en1 = ttk.Entry(self)
                self.knn_en1.place(x=-215, y=60, relx=0.7, relwidth=0.19)

                self.knn_lab2 = tk.Label(self, text =Show_learning_string['label_text']['knn']['text_2'],bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.knn_lab2.place(x=-320, y=90, width=150, height=20, relx=0.65)
                self.knn_comb1 = ttk.Combobox(self, state='readonly')
                self.knn_comb1.place(x=-250, y=90, width=-45, height=20, relx=0.75, relwidth=0.25)
                self.knn_comb1.config(values=self.knn_weights)
                self.knn_comb1.current(0)

                self.knn_lab3 = tk.Label(self, text =Show_learning_string['label_text']['knn']['text_3'],bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.knn_lab3.place(x=-320, y=120, width=150, height=20, relx=0.65)
                self.knn_comb2 = ttk.Combobox(self, state='readonly')
                self.knn_comb2.place(x=-250, y=120, width=-45, height=20, relx=0.75, relwidth=0.25)
                self.knn_comb2.config(values=self.knn_algorithm)
                self.knn_comb2.current(0)

                self.knn_lab4 = tk.Label(self, text =Show_learning_string['label_text']['knn']['text_4'],bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 10, tkFont.BOLD))
                self.knn_lab4.place(x=-320, y=150, width=150, height=20, relx=0.65)
                self.knn_en2 = ttk.Entry(self)
                self.knn_en2.place(x=-215, y=150, relx=0.7, relwidth=0.19)

                Tooltip(self.quest_btn, text= Show_learning_string['tips_text']['text_16'] , wraplength=250)

                def Get_knn_Hyperparameter(self):
                    def threading_5():
                        while not self.done:
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['another']['text_7'])
                            self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_23'])
                            x1 = self.knn_en1.get()
                            x2 = self.knn_comb1.get()
                            x3 = self.knn_comb2.get()
                            x4 = self.knn_en2.get()
                            if x1 == '' or x2 == '' or x3 == '' or x4 == '' :
                                return messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['messagebox_text']['message']['text_4'])
                            model = Demonstration_model('None')
                            train_score, test_score = model.knn_model(int(x1), x2, x3, int(x4))
                            self.scr.delete('end-2l', 'end-1l')
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_8'])
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_9'].format(train_score))
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_10'].format(test_score))
                            self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_26'], 'knn_done')
                            self.scr.tag_config('knn_done', foreground='red')
                            self.done = True
                            self.select_all_btn.config(state='disabled')
                    a5 = threading.Thread(target = threading_5)
                    a5.setDaemon(True)
                    a5.start()
                self.select_all_btn.config(text =Show_learning_string['Button_text']['universal']['text_1'], command = lambda :Get_knn_Hyperparameter(self))

        def select_model(self):
            self.done = True
            self.skip_btn.config(state='disabled')
            model_name = ['Multilayer Perceptron', 'Random Forests','BoostTree', 'Support Vector Machine', 'K Nearest Neighbor']
            self.select_train_env_lab.place_forget()
            self.select_test_env_lab.place_forget()
            self.route_lab.place_forget()
            self.select_train_env_comb.place_forget()
            self.select_test_env_comb.place_forget()
            self.route_comb.place_forget()
            self.select_model_comb = ttk.Combobox(self, state='readonly')
            self.select_model_comb.place(x=-250, y=120, width=-40, height=25, relx=0.75, relwidth=0.25)
            self.select_model_comb.config(values=model_name)
            self.model_lab = tk.Label(self, text = Show_learning_string['scr_text']['create_model']['text_11'] , bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 14, tkFont.BOLD))
            self.model_lab.place(x=-320, y=120, width=120, height=20, relx=0.65)
            self.select_all_btn.config(text = Show_learning_string['Button_text']['universal']['text_1'], command= lambda :model(self))
            Tooltip(self.quest_btn, text=Show_learning_string['tips_text']['text_20'], wraplength=200)
#============================================================================
#  取得訓練&測試資料
#============================================================================
        def Get_data(self):
            def threading_():
                while not self.done:
                    train_env_name = self.select_train_env_comb.get()
                    test_env_name = self.select_test_env_comb.get()
                    route_mode = self.route_comb.get()

                    if train_env_name == '' or test_env_name =='':
                        return messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['messagebox_text']['message']['text_5'])

                    self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_12'])

                    train_feature_data = Feature_Data('train', train_env_name, route_mode)
                    data_exist = train_feature_data.Find_Env_List()
                    if data_exist:
                        for rate in train_feature_data.Output_Excel():
                            self.scr.delete('end-2l', 'end-1l')
                            self.scr.insert(tk.END, f'訓練資料集特徵數據生成中...{rate}%\n')
                            if rate == 100:
                                self.scr.insert(tk.END, f'等待訓練資料集特徵數據輸出...\n')
                            self.update()
                        self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_13'])
                        self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_14'])
                    else:
                        return messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['messagebox_text']['message']['text_6'])

                    test_feature_data = Feature_Data('test', test_env_name, route_mode)
                    data_exist = test_feature_data.Find_Env_List()
                    if data_exist:
                        for rate in test_feature_data.Output_Excel():
                            self.scr.delete('end-2l', 'end-1l')
                            self.scr.insert(tk.END, f'測試資料集特徵數據生成中...{rate}%\n')
                            if rate == 100:
                                self.scr.insert(tk.END, f'等待測試資料集特徵數據輸出...\n')

                            self.update()
                        self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_15'])
                        self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_16'])
                    else:
                        return messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['messagebox_text']['message']['text_6'])
                    self.done = True
                select_model(self)

            a = threading.Thread(target = threading_)
            a.setDaemon(True)
            a.start()

        self.select_all_btn = tk.Button(self)
        a1 = tk.PhotoImage(file='pictures/Start.png')
        self.a1 = a1.subsample(4,4)
        self.select_all_btn.config(text =Show_learning_string['Button_text']['universal']['text_1'], command= lambda :Get_data(self),compound = tk.LEFT, image=self.a1,bg='#C4A98B')
        self.select_all_btn.place(x=10, y=-260, rely=1, relx=0.43, width=100)
        self.select_all_btn['font'] = self.ChiFont
        self.skip_btn = tk.Button(self)
        a2 =tk.PhotoImage(file ='pictures/L_01.png')
        self.a2 = a2.subsample(14,14)
        self.skip_btn.config(text =Show_learning_string['Button_text']['universal']['text_6'], command= lambda :select_model(self),compound = tk.LEFT,image= self.a2,bg='#C4A98B')
        self.skip_btn.place(x=-120 ,y=-260, rely=1, relx=0.43,width=100)
        self.skip_btn['font'] = self.ChiFont
        S_04 = tk.PhotoImage(file ='pictures/S_04.png')
        self.S_04 = S_04.subsample(14,14)
        self.destory_self_btn = tk.Button(self)
        self.destory_self_btn.config(text =Show_learning_string['Button_text']['universal']['text_7'], command= lambda :self.destroy(),compound = tk.LEFT, image=self.S_04,bg='#C4A98B')
        self.destory_self_btn.place(x=-250 ,y=-260, rely=1, relx=0.43, width=100)
        self.destory_self_btn['font'] = self.ChiFont

        t = threading.Thread(target = self.updata_scr)
        t.setDaemon(True)
        t.start()

    def updata_scr(self):
        self.done = False
        while not self.done:
            print('Demonstration 線程正在執行中!')
            time.sleep(2)
            #self.update()


if __name__ == "__main__":
    win = Demonstration_window()
    win.mainloop()
