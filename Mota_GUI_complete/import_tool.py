import gc
import warnings
from pip._internal import main as pip
#pip(['uninstall','numpy','--yes'])
#gc.collect()
'''
pip(['uninstall','pd']) 
pip(['uninstall','tensorflow'])
pip(['uninstall','Pillow'])
'''

try:
    import numpy as np
except ImportError:
    pip(['install', '--user','numpy'])   
    import numpy as np
except ModuleNotFoundError:
    print("找不到 numpy 請重新安裝")
    
try:
    import pandas as pd
except ImportError:
    pip(['install', '--user','pd'])   
    import pandas as pd
except ModuleNotFoundError:
    print("找不到 pandas 請重新安裝")
      
try:
    import tensorflow as tf
    from tensorflow.keras.optimizers import RMSprop
except ImportError:
    pip(['install', '--user','tensorflow==1.15.0'])
    import tensorflow as tf
    from tensorflow.keras.optimizers import RMSprop
except ModuleNotFoundError:
    print("找不到 tensorflow 請重新安裝")
    
try:
    from keras.models import Sequential, load_model
    from keras.layers import Dense, Dropout
except ImportError:
    pip(['install', '--user','keras==2.3.1'])
    from keras.models import Sequential, load_model
    from keras.layers import Dense, Dropout 
except ModuleNotFoundError:
    print("找不到 keras 請重新安裝")

    
try:
    from sklearn import svm, preprocessing 
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.externals import joblib
    from sklearn.preprocessing import Normalizer   
except ImportError:
    pip(['install', '--user','scikit-learn==0.21.3'])
    from sklearn import svm, preprocessing 
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.externals import joblib
    from sklearn.preprocessing import Normalizer
except ModuleNotFoundError:
    print("找不到 sklearn 請重新安裝")
    
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    pip(['install', '--user','matplotlib'])    
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ModuleNotFoundError:
    print("找不到 matplotlib 請重新安裝")
    
try:
    from PIL import Image, ImageTk
except ImportError:
    pip(['install', '--user','Pillow==6.2.0'])
    from PIL import Image, ImageTk
except ModuleNotFoundError:
    print("找不到 Pillow 請重新安裝")
    
try:
    import tkinter as tk
    import tkinter.font as font
    from tkinter import scrolledtext, messagebox
    from tkinter import font as tkFont
    from tkinter import ttk
    
except ImportError:
    pip(['install', '--user','tkinter'])
    import tkinter as tk
    import tkinter.font as font
    from tkinter import scrolledtext, messagebox
    from tkinter import font as tkFont
    from tkinter import ttk
except ModuleNotFoundError:
    print("找不到 tkinter 請重新安裝")

try:
    import io
except ImportError:
    pip(['install', '--user','io'])
    import io    
except ModuleNotFoundError:
    print("找不到 io 請重新安裝")

try:
    import time
except ImportError:
    pip(['install', '--user','time'])
    import time    
except ModuleNotFoundError:
    print("找不到 time 請重新安裝")

try:
    import os 
except ImportError:
    pip(['install', '--user','os-win'])
    import os    
except ModuleNotFoundError:
    print("找不到 os 請重新安裝") 


try:
    import random 
except ImportError:
    pip(['install', '--user','random'])
    import random    
except ModuleNotFoundError:
    print("找不到 random 請重新安裝")

try:
    import threading 
except ImportError:
    pip(['install', '--user','threading'])
    import threading    
except ModuleNotFoundError:
    print("找不到 threading 請重新安裝")

    