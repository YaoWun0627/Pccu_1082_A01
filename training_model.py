# -*- coding: utf-8 -*-
"""
【訓練模型】                        by Hung1  2020/4/29
將訓練集資料，以分類統計的演算法或是人工神經網路來訓練出一個模型。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from sklearn import svm, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def preprocess(df, labels_assigned=None, label=[]):
    """
    預處理，對特徵集的資料作轉換
    輸入：
    df: 要進行預處理的資料
    labels_assigned: 可選，可以按照傳入的映射表作轉換。默認為None
    輸出：
    df: 預處理後的資料
    assigned: 當沒有使用映射表，會回傳一個由此資料產生的映射表
    """
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
            df[col] = df[col].map(lambda x: assigned[x]) #!!!映射表若沒有對應將會出錯
        return df

def regularlization(df, scaler='MinMax'):
    """
    正規化，將特徵集的數值進行正規化
    輸入：
    df: 要進行正規化的資料
    scaler: 正規化的類型，可選擇'MinMax','MaxAbs','Standard'。默認為'MinMax'
    """
    # 正規化0~1
    if scaler == 'MinMax':
        min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(df)
    # 正規化-1~1
    elif scaler == 'MaxAbs':
        max_abs_scaler = preprocessing.MaxAbsScaler()
        df = max_abs_scaler.fit_transform(df)
    # 常態分佈(平均值=0，標準差=1)
    elif scaler == 'Standard':
        standard_scaler = preprocessing.StandardScaler()
        df = standard_scaler.fit_transform(df)
    else:
        raise RuntimeError('scaler parameter error')
    return df

#-------------------------------------------------------------------

select_model = 'rfc'
# 產生訓練集和測試集
network = 'policy'
#network = 'value'
df_train = pd.read_excel(f'model/{network}_training.xls')
df_test = pd.read_excel(f'model/{network}_test.xls')
df_train, labels_assigned = preprocess(df_train, label=['class','class2'])
df_test = preprocess(df_test, labels_assigned, label=['class','class2'])
Y_train = df_train.choose
X_train = df_train.drop(['choose'], axis=1)
Y_test = df_test.choose
X_test = df_test.drop(['choose'], axis=1)

# 開始建立&訓練模型
if select_model == 'mlp':
    # 建立MLP模型
    print('mlp model:')
    model = Sequential()
    model.add(Dense(256,
                    input_dim=28,
                    kernel_initializer='uniform', #normal
                    activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128,
                    kernel_initializer='uniform',
                    activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256,
                    kernel_initializer='uniform',
                    activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))
    # 模型資訊
    #model.summary()
    # 訓練模型
    model.compile(loss='binary_crossentropy', #mean_squared_error
                  optimizer='adam',
                  metrics=['acc'])
    train_history = model.fit(X_train, Y_train,
                              epochs=50,#400
                              batch_size=32,#8
                              validation_split=0.1,
                              verbose=1,
                              shuffle=True)
                              #validation_data=(X_val, Y_val))
    # 訓練紀錄
    def show_train_history(train_history, train, validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('Train History')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    show_train_history(train_history, 'acc', 'val_acc')
    show_train_history(train_history, 'loss', 'val_loss')
    # 評估準確率
    scores = model.evaluate(X_train, Y_train, verbose=0)
    print('訓練集準確率：', scores[1])
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print('測試集準確率：', scores[1])
    # 預測結果
    test_pred = model.predict(X_test)
    # 儲存模型
    #model.save(f'model/{network}_mlp_model.h5')
    #np.save(f'model/{network}_mlp_model_labels.npy', labels_assigned)
    # 載入模型
    #model = load_model(f'model/{network}_mlp_model.h5')

elif select_model == 'rfc':
    # 建立隨機森林模型
    print('rfc model:')
    #rfc = RandomForestClassifier(n_estimators=1000,
    #                             min_samples_split=20,
    #                             min_samples_leaf=1,
    #                             oob_score=True,
    #                             random_state=1,
    #                             n_jobs=-1)
    #rfc = RandomForestClassifier(n_estimators=100,
    #                             min_samples_split=50,
    #                             oob_score=True,
    #                             random_state=2,
    #                             verbose=1)
    rfc = RandomForestClassifier(n_estimators=50,
                                 min_samples_split=30,
                                 oob_score=True,
                                 random_state=1)
    # 訓練模型
    rfc.fit(X_train, Y_train)
    # 評估準確率
    print('oob_score  ：', rfc.oob_score_)
    print('訓練集準確率：', rfc.score(X_train, Y_train))
    test_pred = rfc.predict(X_test)
    print('測試集準確率：', accuracy_score(Y_test, test_pred))
    # 預測結果
    test_pred_p = rfc.predict_proba(X_test)
    # 儲存模型
    #joblib.dump(rfc, f'model/{network}_rfc_model.pkl')
    #np.save(f'model/{network}_rfc_model_labels.npy', labels_assigned)
    # 載入模型
    #rfc2 = joblib.load(f'model/{network}_rfc_model.pkl')

elif select_model == 'svm':
    # 建立SVM模型
    print('svm model:')
    svmcal = svm.SVC(kernel='sigmoid', gamma='scale', coef0=0.9, probability=True)
    #svmcal = svm.SVC(kernel='linear', probability=True, decision_function_shape='ovr')
    #svmcal = svm.SVC(kernel='rbf', gamma=20, decision_function_shape='ovr')
    # 訓練模型
    svmcal.fit(X_train, Y_train)
    # 評估準確率
    print('訓練集準確率：', svmcal.score(X_train, Y_train))
    test_pred = svmcal.predict(X_test)
    print('測試集準確率：', accuracy_score(Y_test, test_pred))
    # 預測結果
    test_pred_p = svmcal.predict_proba(X_test)
    # 儲存模型
    #joblib.dump(svmcal, f'model/{network}_svm_model.pkl')
    #np.save(f'model/{network}_svm_model_labels.npy', labels_assigned)
    # 載入模型
    #svmcal2 = joblib.load(f'model/{network}_svm_model.pkl')

elif select_model == 'knn':
    # 建立KNN模型
    print('knn model:')
    #knn = KNeighborsClassifier()
    knn = KNeighborsClassifier(n_neighbors=20, weights= 'distance', algorithm='auto',
                               leaf_size=5, p=1, metric='minkowski', metric_params=None, n_jobs=-1)
    # 訓練模型
    knn.fit(X_train, Y_train)
    # 評估準確率
    print('訓練集準確率：', knn.score(X_train, Y_train))
    test_pred = knn.predict(X_test)
    print('測試集準確率：', accuracy_score(Y_test, test_pred))
    # 預測結果
    test_pred_p = knn.predict_proba(X_test)
    # 儲存模型
    #joblib.dump(knn, f'model/{network}_knn_model.pkl')
    #np.save(f'model/{network}_knn_model_labels.npy', labels_assigned)
    # 載入模型
    #knn2 = joblib.load(f'model/{network}_knn_model.pkl')
