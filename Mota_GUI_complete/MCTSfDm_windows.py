import database as db
from import_tool import *
from String_ import *
from window_resize import win_resize
from MCTSfDm_model_create import MCTSfDm_Model

class MCTSfDm_window(tk.Toplevel):

    def __init__(self):
        super(MCTSfDm_window, self).__init__()
        winWidth = 806
        winHeight = 606
        self.x = (self.winfo_screenwidth() - winWidth) // 2
        self.y = (self.winfo_screenheight() - winHeight) // 3
        self.title('MCTSfD')
        self.ChiFont = font.Font(family=Show_learning_string['Initialize']['font_style_Chinese'], weight=tkFont.BOLD)
        win_resize(self)
        self.geometry('%sx%s+%s+%s' % (700, 500, self.x, self.y))
        self.Confirm_btn = tk.Button(self)
        self.labframe = tk.LabelFrame(self, text=Show_learning_string['label_text']['select_env']['text_4'] ,bg='#C4A98B')
        self.labframe.place(x=10, y=160, width=-20, height=-170, relx=0.6, relwidth=0.4, relheight=1.0)

        self.scr = scrolledtext.ScrolledText(self.labframe)
        self.scr.pack(fill='both', expand=True)
        #
        self.model_lab = tk.Label(self,text =Show_learning_string['label_text']['demo']['text_4'], bg='#A07558',font=(Show_learning_string['Initialize']['font_style_Chinese'], 14, tkFont.BOLD))
        self.model_lab.place(x=-270, y=10, width=300, height=30, relx=0.7)

        self.train_env_lab = tk.Label(self, text =Show_learning_string['label_text']['demo']['text_2'], bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 14, tkFont.BOLD))
        self.train_env_lab.place(x=-300, y=70, width=100, height=20, relx=0.65)

        self.train_env_comb = ttk.Combobox(self, state='readonly')
        self.train_env_comb.place(x=-250, y=70, width=140, height=20, relx=0.75)
        self.train_env_comb.config(values=db.FILE_NAME)
        self.train_env_comb.set(db.FILE_NAME[0])

        self.route_lab = tk.Label(self, text = Show_learning_string['label_text']['demo']['text_5'], bg='#C4A98B',font=(Show_learning_string['Initialize']['font_style_Chinese'], 14, tkFont.BOLD))
        self.route_lab.place(x=-300, y=120, width=100, height=20, relx=0.65)
        self.route_comb = ttk.Combobox(self, state='readonly')
        self.route_comb.place(x=-250, y=120, width=140, height=20, relx=0.75)
        self.route_comb.config(values=['單一地圖', '強化地圖', '強化泛化'])
        self.route_comb.current(0)

        def create_model(self):
            def threading_1():
                while not self.done:
                    env_value = self.train_env_comb.get()
                    route_mode = self.route_comb.get()
                    if env_value == '':

                        return messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['scr_text']['create_model']['text_17'])

                    self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_18'] +env_value+ Show_learning_string['scr_text']['create_model']['text_19'])
                    self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_27'])
                    model = MCTSfDm_Model(env_value, route_mode)
                    data_exist = model.Find_Env_List()
                    if data_exist:
                        model_score = model.Modle_training()
                        self.scr.insert(tk.END, Show_learning_string['scr_text']['create_model']['text_9'].format(model_score))
                        rate, star = model.Predict_Route()
                        self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_20'].format(rate))
                        self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_21'].format(star))
                        self.scr.insert(tk.END,Show_learning_string['scr_text']['create_model']['text_26'], 'mctsfd_done')
                        self.scr.tag_config('mctsfd_done', foreground='red')
                        self.Confirm_btn.config(state='disabled')
                        self.done = True
                    else:
                        return messagebox.showwarning(Show_learning_string['messagebox_text']['title']['text_1'], Show_learning_string['messagebox_text']['message']['text_6'])

            t1 = threading.Thread(target =  threading_1)
            t1.setDaemon(True)
            t1.start()

        a1 = tk.PhotoImage(file='pictures/Start.png')
        self.a1 = a1.subsample(4,4)
        self.Confirm_btn.config(text = Show_learning_string['Button_text']['universal']['text_1'],command=lambda:create_model(self),image= self.a1,compound = tk.LEFT,bg='#C4A98B')
        self.Confirm_btn.place(x=10, y=-310, rely=1, relx=0.43, width=100)
        self.Confirm_btn['font'] = self.ChiFont

        S_04 = tk.PhotoImage(file ='pictures/S_04.png')
        self.S_04 = S_04.subsample(14,14)
        self.close_window = tk.Button(self)
        self.close_window.config(text =Show_learning_string['Button_text']['universal']['text_7'], command= lambda :self.destroy(),compound = tk.LEFT, image=self.S_04,bg='#C4A98B')
        self.close_window.place(x=-120 ,y=-310, rely=1, relx=0.43, width=100)
        self.close_window['font'] = self.ChiFont

        t = threading.Thread(target = self.updata_scr)
        t.setDaemon(True)
        t.start()

    def updata_scr(self):
        self.done = False
        while not self.done:
            print('MCTSfD 線程正在執行中!')
            time.sleep(2)
            #self.update()

if __name__ == "__main__":
    win = MCTSfDm_window()
    win.mainloop()