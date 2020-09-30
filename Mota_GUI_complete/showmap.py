# -*- coding: utf-8 -*-
'''
可以根據框架(Frame)的大小產生滾動條。    by Hung1    2020/1/22 優化版本
以下本套件功能：
  可以兼容Notebook等，能夠正常運行
  新增兩軸自動滾動軸
  在畫框內可用鼠標代替滾動軸功能
  解決鼠標無法在範圍內正常運行
  當內容超出畫框範圍時，內容框架跟隨畫框調整大小
  解決畫框內容無法跟著縮放的問題
  解決框架內容不足時還能往上滾動畫面
參考程式：
  滾動條運作(基礎)
  https://stackoverflow.com/questions/46215310/scrollbar-checkbutton-and-canvas-in-python-2-7
  滾動軸連動畫框(基礎)
  https://stackoverflow.com/questions/48027035/tkinter-scrollbars-appear-but-dont-work
  按鈕調整畫框內容大小
  https://www.codenong.com/13197469/
  雙框架獨立滾動軸
  https://stackoverflow.com/questions/51538818/python-tkinter-binding-child-widgets
  監控事件列表
  https://blog.csdn.net/liuxu0703/article/details/60604637
  調整框架寬度
  https://t.codebug.vip/questions-1870701.htm
'''
import tkinter as tk
from tkinter import ttk

class AutoScrollbar(ttk.Scrollbar):
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with this widget')

    def place(self, **kw):
        raise tk.TclError('Cannot use place with this widget')

class Frame_Scrollbar(tk.Frame):
    def __init__(self, mainframe, bg='light gray'):
        tk.Frame.__init__(self, master=mainframe)
        # 子框架填滿
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        # 畫布
        self.canvas = tk.Canvas(self, background=bg)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        # 內容框架
        self.contents = tk.Frame(self, background=bg)
        self.contents.rowconfigure(0, minsize=200, weight=1)  # 適用grid
        self.contents.columnconfigure(0, minsize=200, weight=1)
        self.contents.columnconfigure(1, weight=1)
        self.window_id = self.canvas.create_window(2, 2, window=self.contents, anchor='nw') #(2,2)是防止底色沒對齊
        # 限制滾動範圍
        self.contents.bind('<Configure>', self._scroll_area)
        # 滾動條
        vbar = AutoScrollbar(self, orient="vertical")
        hbar = AutoScrollbar(self, orient='horizontal')
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky='we')
        vbar.config(command=self.canvas.yview)
        hbar.config(command=self.canvas.xview)
        self.canvas.config(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        # 內容框架連動縮放
        self.canvas.bind('<Configure>', self._change_zoom)
        # 範圍內啟用滾動條
        self.bind("<Enter>", self._set_binds_canvas)
        # 離開範圍內禁用滾動條
        self.bind("<Leave>", self._unset_binds_canvas)
    
    def clear(self):
        # 銷毀框架中的所有小部件
        for widget in self.contents.winfo_children():
            widget.destroy()
        # 將內容框架縮小
        self.contents.configure(height=1, width=1)
        
    def _scroll_area(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def _change_zoom(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width-4) #-4是防止底色沒對齊
        
    def _set_binds_canvas(self, event):
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        
    def _unset_binds_canvas(self, event):
        self.unbind_all("<MouseWheel>")
        
    def _on_mousewheel(self, event):
        # 當內容不足時，禁止任何滾動處理
        if self.canvas.yview() != (0.0, 1.0):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Frame_Scrollbar Test")
    root.geometry("800x400")
    notebook = ttk.Notebook(root)
    tab = Frame_Scrollbar(notebook)
    # 添加內容
    for i in range(30):
        #label = tk.Label(tab.contents, text="label #{}".format(i), width=30)
        #label.pack(side="top", fill="x", padx=1, pady=1)
        #label.pack(fill='x', expand=True)
        
        fra = tk.Frame(tab.contents, bg='blue')
        label = tk.Label(fra, text="label #{}".format(i), width=30)
        label.pack(fill='x', expand=True, side='left')
        label = tk.Label(fra, text="label #{}".format(i), width=30)
        label.pack(fill='x', expand=True, side='right')
        fra.pack(fill='x', expand=True, side='bottom')
    #for i in range(30):
    #    label = ttk.Label(tab.contents, text="This is a label "+str(i))
    #    label.grid(column=1, row=i, sticky='w')
    #    text = ttk.Entry(tab.contents)
    #    text.grid(column=2, row=i, sticky='w')
    # 其他裝飾框架
    notebook.add(tab, text='tag')
    tab2 = tk.Frame(notebook, bg='blue')
    notebook.add(tab2, text='tag')
    notebook.place(relwidth=0.6, relheight=1.0)
    frm2 = ttk.LabelFrame(root, text='Information')
    scr = tk.scrolledtext.ScrolledText(frm2)
    scr.pack(fill='both', expand=True)
    frm2.place(relx=0.6, relwidth=0.4, relheight=1.0)
    scr.insert(tk.END, '\n【選擇要\n創建環境\n的遊戲目錄】\n')
    scr.insert(tk.END, 'html5魔塔\n的遊戲根\n目錄，將讀取\n其子目\n\
               \n\n\n\n\n\n\n錄\n下的js\n檔，建立\n成遊\n\n\n\n\n\n\n\
               \n\n\n\n\n戲環境。\n\n')
    # 銷毀部件
    #root.after(5000, lambda: tab.clear())
    root.mainloop()