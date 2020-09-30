# -*- coding: utf-8 -*-
# Advanced zoom example. Like in Google Maps.
# It zooms only a tile, but not the whole image. So the zoomed tile occupies
# constant memory and not crams it with a huge resized image for the large zooms.
'''
【畫布滾動縮放】                        by Hung1  2020/1/19
增加一些新功能，微調內部程式，解決字體縮放問題。
原程式碼出處：
https://stackoverflow.com/questions/41656176/tkinter-canvas-zoom-move-pan/48137257#48137257
畫有透明度的矩形
https://stackoverflow.com/questions/54637795/how-to-make-a-tkinter-canvas-rectangle-transparent
python界面上的圖片縮放，根據窗口大小
https://blog.csdn.net/sinat_27382047/article/details/80138733
'''
from import_tool import *


class AutoScrollbar(ttk.Scrollbar):
    ''' A scrollbar that hides itself if it's not needed.
        Works only if you use the grid geometry manager '''
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

class Zoom_Advanced(tk.Frame):
    ''' Advanced zoom of the image '''
    def __init__(self, mainframe, path):
        ''' Initialize the main Frame '''
        #保存快取對象
        self.fp_cache = {}
        self.text_cache = []
        tk.Frame.__init__(self, master=mainframe)
        # Vertical and horizontal scrollbars for canvas
        vbar = AutoScrollbar(self.master, orient='vertical')
        hbar = AutoScrollbar(self.master, orient='horizontal')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='we')
        # Create canvas and put image on it
        self.canvas = tk.Canvas(self.master, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        # 使用藍色背景，更好聚焦圖片
        self.canvas.configure(background="white")
        self.canvas.grid(row=0, column=0, sticky='nswe')
        #self.canvas.update()  # wait till canvas is created 不需暫停更新
        vbar.configure(command=self.scroll_y)  # bind scrollbars to the canvas
        hbar.configure(command=self.scroll_x)
        # Make the canvas expandable
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', self.show_image)  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.move_from)
        self.canvas.bind('<B1-Motion>',     self.move_to)
        self.canvas.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>',   self.wheel)  # only with Linux, wheel scroll down
        self.canvas.bind('<Button-4>',   self.wheel)  # only with Linux, wheel scroll up
        self.image = self.picture_cache(path)  # open image (cache by Hung1)
        self.width, self.height = self.image.size
        self.imscale = 1.0  # scale for the canvaas image
        self.delta = 1.3  # zoom magnitude
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)
        # Plot some optional random rectangles for the test purposes
        # 若圖片小於窗口大小，將圖片置中
        self.canvas.update() # 畫面更新
        x = max(0, self.canvas.winfo_width() // 2 - self.width // 2)
        y = max(0, self.canvas.winfo_height() // 2 - self.height // 2)
        self.canvas.scan_dragto(x, y, gain=1)
        self.canvas.update() # 畫面更新
        self.show_image()

    def scroll_y(self, *args, **kwargs):
        ''' Scroll canvas vertically and redraw the image '''
        self.canvas.yview(*args, **kwargs)  # scroll vertically
        self.show_image()  # redraw the image

    def scroll_x(self, *args, **kwargs):
        ''' Scroll canvas horizontally and redraw the image '''
        self.canvas.xview(*args, **kwargs)  # scroll horizontally
        self.show_image()  # redraw the image

    def move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.show_image()  # redraw the image

    def wheel(self, event):
        ''' Zoom with mouse wheel '''
        # 解決圖片縮放後，切換圖片會帶來文字位移的問題
        x = 0#self.canvas.canvasx(event.x)
        y = 0#self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]: pass  # Ok! Inside the image
        else: return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down
            i = min(self.width, self.height)
            if int(i * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale = round(self.imscale / self.delta, 2)  # 避免精度偏移
            scale        = round(scale / self.delta, 2)  # 避免精度偏移
        if event.num == 4 or event.delta == 120:  # scroll up
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale = round(self.imscale * self.delta, 2)  # 避免精度偏移
            scale        = round(scale * self.delta, 2)  # 避免精度偏移
        self.canvas.scale('all', x, y, scale, scale)  # rescale all canvas objects
        # 文字可以跟著縮放
        for text, font_size in self.text_cache:
            self.canvas.itemconfig(text, font='Calibri %d italic' % (font_size+round(self.imscale**2)))
        self.show_image()

    def show_image(self, event=None):
        ''' Show image on the Canvas '''
        bbox1 = self.canvas.bbox(self.container)  # get image area
        # Remove 1 pixel shift at the sides of the bbox1
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (self.canvas.canvasx(0),  # get visible area of the canvas
                 self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  # get scroll region box
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # whole image in the visible area
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # whole image in the visible area
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # set scroll region
        x1 = max(bbox2[0] - bbox1[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            x = min(int(x2 / self.imscale), self.width)   # sometimes it is larger on 1 pixel...
            y = min(int(y2 / self.imscale), self.height)  # ...and sometimes not
            image = self.image.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def picture_cache(self, path):
        ''' 將圖檔以open()開啟，以便於之後釋放內存 '''
        if self.fp_cache.__contains__(path):
            fp = self.fp_cache[path]
        else:
            fp = open(path,'rb')
            self.fp_cache[path] = fp
        return Image.open(fp)

    def clear_cache(self):
        ''' 清除快取，釋放內存 '''
        for _, fp in self.fp_cache.items():
            fp.close()
        self.fp_cache.clear()

    def draw_text(self, text, x, y, color='red', font_size=16):
        ''' 繪製文字 '''
        # 解決圖片縮放後，切換圖片會帶來文字對不上位置的問題
        tex = self.canvas.create_text(x*self.imscale, y*self.imscale, text=text, font='Calibri %d italic' % font_size, fill=color)
        self.text_cache.append((tex, font_size))
        self.canvas.update()

    def clear_text(self):
        ''' 清除文字 '''
        for text, _ in self.text_cache:
            self.canvas.delete(text)
        self.text_cache.clear()

    def change_image_path(self, new_path, freeze_size = True):
        ''' 改變顯示圖片 '''
        self.image = self.picture_cache(new_path)
        if not freeze_size:
            self.width, self.height = self.image.size
            self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)
        self.show_image()
        self.canvas.update()

    def delete_canvas(self):
        ''' 刪除畫布 '''
        self.clear_text()
        self.canvas.destroy()
        self.clear_cache()

if __name__ == '__main__':
    path = 'pictures/MT_0.png'  # place path to your image here
    root = tk.Tk()
    frm = ttk.LabelFrame(root, text='Compiled Map')
    frm.place(relwidth=0.6, relheight=1.0)
    app = Zoom_Advanced(frm, path=path)
    app.draw_text('10', 330, 330)
    app.draw_text('33', 362, 330, font_size=14)
    app.change_image_path('pictures/MT_1.png')
    root.mainloop()