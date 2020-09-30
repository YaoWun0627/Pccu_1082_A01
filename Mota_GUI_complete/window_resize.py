from import_tool import *

class win_resize():
    def __init__(self, root):
        self.im = Image.open('pictures/background_image.jpg')
        self.im_copy= self.im.copy()
        self.img = ImageTk.PhotoImage(self.im)
        self.imLabel = tk.Label(root, image=self.img)
        self.imLabel.pack(fill='both', expand='yes')
        self.imLabel.bind('<Configure>', self._resize_image)
    
    def _resize_image(self, event):
        new_width = event.width
        new_height = event.height
        self.im = self.im_copy.resize((new_width, new_height))
        self.img = ImageTk.PhotoImage(self.im)
        self.imLabel.configure(image = self.img)
       
if __name__ == '__main__':
    pass