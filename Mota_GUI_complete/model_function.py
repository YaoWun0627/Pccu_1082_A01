from import_tool import *

class Demonstration_model(object):
    def __init__(self, regularlization_value):
        train_data = pd.read_excel('model/train.xls')
        test_data = pd.read_excel('model/test.xls')

        df_train, self.labels_assigned = self.preprocess(train_data, label=['class'])
        df_test = self.preprocess(test_data, self.labels_assigned, label=['class'])
        self.regularlization_value = regularlization_value
        self.X_train = df_train.drop(['choose'], axis=1)
        self.Y_train = df_train.choose
        self.Y_test = df_test.choose
        self.X_test = df_test.drop(['choose'], axis=1)
        self.regularlization(self.X_train, self.regularlization_value)

    def preprocess(self, df, labels_assigned=None, label=[]):
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
                df[col] = df[col].map(lambda x: assigned[x])
            return df

    def regularlization(self, df, regularlization_value):
        if regularlization_value == 'None':
            df = df
        elif regularlization_value == 'L1':
            standard_scaler =  Normalizer(norm='l1')
            df = standard_scaler.fit_transform(df)
            print('使用L1正規化')
        if regularlization_value == 'L2':
            standard_scaler =  Normalizer(norm='l2')
            df = standard_scaler.fit_transform(df)
            print('使用L2正規化')
        return  df

    def mlp_model(self, x1, x2, x3, x4):
        print('mlp model:')
        model = Sequential()
        model.add(Dense(x1,
                        input_dim=18,
                        kernel_initializer='uniform',
                        activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(x2,
                        kernel_initializer='uniform',
                        activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(x3,
                        kernel_initializer='uniform',
                        activation='relu'))
        model.add(Dense(1,
                        kernel_initializer='uniform',
                        activation='sigmoid'))
        # 訓練模型
        model.compile(loss='binary_crossentropy',
                      optimizer='RMSprop',
                      metrics=['acc'])

        #t = threading.Thread(self.print_time())
        #t.start()
        print('------------')
        model.fit(self.X_train, self.Y_train,
                      epochs=x4,
                      batch_size=32,
                      validation_split=0.1,
                      verbose=2,
                      shuffle=True)

        self.done = True
        # 評估準確率
        train_scores = model.evaluate(self.X_train, self.Y_train, verbose=0)
        print('訓練集準確率：', train_scores[1])
        test_scores  = model.evaluate(self.X_test, self.Y_test, verbose=0)
        print('測試集準確率：', test_scores[1])
        # 儲存模型
        model.save(f'model/Multilayer Perceptron_model.h5')
        np.save(f'model/Multilayer Perceptron_model_labels.npy', self.labels_assigned)
        return train_scores[1], test_scores[1]

    def rfc_modle(self, x1, x2, x3):
        print('rfc model:')

        rfc = RandomForestClassifier(n_estimators=x1,
                                     min_samples_split=x2,
                                     oob_score=True,
                                     random_state=x3).fit(self.X_train, self.Y_train)

        train_scores = rfc.score(self.X_train, self.Y_train)
        print('訓練集準確率：', train_scores)
        test_pred = rfc.predict(self.X_test)
        test_scores = accuracy_score(self.Y_test, test_pred)
        print('測試集準確率：', test_scores)
        # 儲存模型
        joblib.dump(rfc,'model/Random Forests_model.pkl')
        np.save('model/Random Forests_model_labels.npy', self.labels_assigned)

        return train_scores, test_scores

    def BoostTree_model(self, x1, x2, x3):
        print('BoostTree model:')
        BT = GradientBoostingClassifier(n_estimators= x1,
                             loss = "exponential",
                             learning_rate= x2,
                             min_samples_split= x3,
                             random_state=0).fit(self.X_train, self.Y_train)

        train_scores = BT.score(self.X_train, self.Y_train)
        print('訓練集準確率：', train_scores)
        test_pred = BT.predict(self.X_test)
        test_scores = accuracy_score(self.Y_test, test_pred)
        print('測試集準確率：', test_scores)
        # 儲存模型
        joblib.dump(BT,'model/BoostTree_model.pkl')
        np.save('model/BoostTree_model_labels.npy', self.labels_assigned)
        return train_scores, test_scores

    def svm_model(self, x1):
        print('svm model:')
        svmcal = svm.SVC(kernel= x1, gamma='auto', coef0=1, probability=True).fit (self.X_train, self.Y_train)
        train_scores = svmcal.score(self.X_train, self.Y_train)
        print('訓練集準確率：', train_scores)
        test_pred = svmcal.predict(self.X_test)
        test_scores = accuracy_score(self.Y_test, test_pred)
        print('測試集準確率：', test_scores)
        joblib.dump(svmcal, 'model/Support Vector Machine_model.pkl')
        np.save('model/Support Vector Machine_model_labels.npy', self.labels_assigned)
        return train_scores, test_scores

    def knn_model(self, x1, x2, x3, x4):
         print('knn model:')
         knn = KNeighborsClassifier(n_neighbors=x1, weights= x2, algorithm= x3,
                           leaf_size=x4, p=1, metric='minkowski', metric_params=None, n_jobs=-1).fit(self.X_train, self.Y_train)
         train_scores = knn.score(self.X_train, self.Y_train)
         print('訓練集準確率：', train_scores)
         test_pred = knn.predict(self.X_test)
         test_scores = accuracy_score(self.Y_test, test_pred)
         print('測試集準確率：', test_scores)
         joblib.dump(knn, f'model/K Nearest Neighbor_model.pkl')
         np.save(f'model/K Nearest Neighbor_model_labels.npy', self.labels_assigned)

         return train_scores, test_scores

if __name__ == '__main__':

    model = Demonstration_model('None')
    model.mlp_model(512, 256, 128, 10)
