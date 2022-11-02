import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

import seaborn as sns
from DataPreprocess import DataPreprocess

dataset_root = "./dataset/"

class LSTM_remake:
    def __init__(self, _df):
        self.filename = None
        self.early_stop = None
        self.checkpoint = None
        self.dataframe_origin = _df
        self.column_list = ['MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP', 'OK']  # default column
        self.model_path = './model/'
        self.model = None

    def run(self):
        train, test = self.train_test_split(_info=True)
        train_sc, test_sc = self.minmax_scaler(train, test)

        X_train_over, y_train_over = self.SMOTE_make(train_sc)
        train_feature, train_label = self.make_dataset(X_train_over, y_train_over, 10)

        test_feature, test_label = test_sc[self.column_list[:-1]], test_sc[self.column_list[-1]]
        test_feature, test_label = self.make_dataset(test_feature, test_label, 10)

        x_train, x_val, y_train, y_val = train_test_split(train_feature, train_label, test_size=0.3)

        self.make_model(train_feature)
        self.model_compile()

        self.model_fit(x_train, y_train, x_val, y_val)
        self.model_predict(test_feature, test_label)



    def set_column_list(self, _columns):
        """
        feature로 사용할 columns 리스트 변경
        :param _columns: columns 리스트
        :return: None
        """
        self.column_list = _columns
        print(f"column_list is changed {self.column_list}")

    def train_test_split(self, _rate=0.7, _info=False):
        """
        특정 비율 만큼 train, test 분리하기 (default=0.7)
        :param _rate: 비율
        :param _info: 분리한 train, test 정보를 보고 싶을때
        :return: train, test
        """
        split_num = int(self.dataframe_origin.shape[0] * _rate)
        train_df = self.dataframe_origin[:split_num]
        test_df = self.dataframe_origin[split_num:]
        if _info:
            print(f"type(train)={type(train_df)}, {train_df.shape}")
            print(f"type(test)={type(test_df)}, {test_df.shape}")

        return train_df, test_df

    def minmax_scaler(self, _train, _test, _info=False):
        """
        MinMaxScaler 0~1사이의 값으로 변경 (음수가 있을 경우 -1~1 사이의 값으로 변경)
        :param _train:
        :param _test:
        :param _info: Scaler한 정보를 보고 싶은 경우
        :return: scaler된 train, test
        """
        scaler = preprocessing.MinMaxScaler()
        train_sc = scaler.fit_transform(_train)
        test_sc = scaler.transform(_test)
        if _info:
            print(f"type(train_sc)={type(train_sc)}, train_sc.shape={train_sc.shape}")
            print(f"type(test_sc)={type(test_sc)}, test_sc.shape={test_sc.shape}")
            print(f"train_sc\n{train_sc.head(10)}")
            print(f"test_sc\n{test_sc.head(10)}")

        return train_sc, test_sc

    def SMOTE_make(self, _df, _info=False):
        """
        DataFrame을 오버샘플링 시키는 메서드
        :param _df: DataFrame(train_sc)
        :param _info: 오버샘플링된 데이터의 정보를 보고 싶을때
        :return: 
        """
        X_train_values = _df[:, :-1]
        y_train_values = _df[:, -1]

        smote = SMOTE(random_state=0)
        X_train_over, y_train_over = smote.fit_resample(X_train_values, y_train_values)
        if _info:
            print(f"SMOTE 적용 전 {X_train_values.shape}, {y_train_values.shape}")
            print(f"SMOTE 적용 후 {X_train_over.shape}, {y_train_over.shape}")
            print(f"적용 후 레이블 값 분포\n{pd.Series(y_train_over).value_counts()}")

        return X_train_over, y_train_over

    def make_dataset(self, _data, _label, _window_size):
        """
        과거값을 통해 현재 종속변수(label)을 예측 하도록 데이터셋 설정
        :param _data: X_train
        :param _label: y_train
        :param _window_size: window size --> Sequence data의 크기
        :return: ndarray(feature list), ndarray(label list)
        """
        # DataFrame 형식(iloc 필요함)을 사용하기 위해서 변경이 필요
        if type(_data) == np.ndarray:
            _data = pd.DataFrame(_data, columns=[self.column_list[:-1]])
        if type(_label) == np.ndarray:
            _label = pd.DataFrame(_label, columns=[self.column_list[-1]])

        #
        feature_list = []
        label_list = []

        for i in range(len(_data) - _window_size):
            feature_list.append(np.array(_data.iloc[i:i + _window_size]))
            label_list.append(np.array(_label.iloc[i + _window_size]))
        return np.array(feature_list), np.array(label_list)

    def make_model(self, _train_feature):
        self.model = Sequential()
        self.model.add(LSTM(50,  # LSTM 레이어의 결과 50개 나오도록 지정
                            input_shape=(_train_feature.shape[1], _train_feature.shape[2]),
                            activation='tanh',
                            return_sequences=False)
                       )
        self.model.add(Dense(1, activation='sigmoid'))

    def model_compile(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.early_stop = EarlyStopping(monitor='val_loss', patience=5)
        self.filename = os.path.join(self.model_path, 'tmp_checkpoint.h5')
        self.checkpoint = ModelCheckpoint(self.filename, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    def model_fit(self, _x_train, _y_train, _x_valid, _y_valid, _show=False):
        history = self.model.fit(_x_train, _y_train,
                                 epochs=100,
                                 batch_size=250,
                                 validation_data=(_x_valid, _y_valid),
                                 callbacks=[self.early_stop, self.checkpoint])
        if _show:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='validation')
            plt.legend()
            plt.show()

    def model_predict(self, _test_feature, _test_label):
        self.model.load_weights(self.filename)
        pred = self.model.predict(_test_feature)

        pred_df = pd.DataFrame(pred, columns=[self.column_list[-1]])
        pred_df[self.column_list[-1]] = pred_df[self.column_list[-1]].apply(lambda x: 1 if x >= 0.5 else 0)
        print(pred_df[self.column_list[-1]].value_counts())

        classify = confusion_matrix(_test_label, pred_df)
        print(classify)

        p = precision_score(_test_label, pred_df)
        r = recall_score(_test_label, pred_df)
        f1 = f1_score(_test_label, pred_df)
        acc = accuracy_score(_test_label, pred_df)
        print(f"precision: %0.4f" % p)
        print(f"recall: %0.4f" % r)
        print(f"f1-score: %0.4f" % f1)
        print(f"accuracy: %0.4f" % acc)

