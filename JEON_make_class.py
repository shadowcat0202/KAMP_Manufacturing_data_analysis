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


class Model_LSTM:
    def __init__(self, _df, _split_rate=0.7):
        self.analysis_columns = ["MELT_TEMP", "MOTORSPEED", "MELT_WEIGHT", "INSP", "TAG"]
        self.__data_frame = _df[self.analysis_columns]
        self.__split_data_rate = int(_df.shape[0] * _split_rate)  # 기본 분리 비율 0.7
        self.__train = None
        self.__test = None
        self.__train_sc = None
        self.__test_sc = None

    def make_dataset(self, _info=False):
        self.__train = self.__data_frame[:self.__split_data_rate]
        self.__test = self.__data_frame[self.__split_data_rate:]
        if _info:
            print(f"dataset_split ====================================================")
            print(f"train.shape={self.__train.shape}\ntest.shape={self.__test.shape}")

    def make_MinMax_normalization(self, _info=False):
        scaler = preprocessing.MinMaxScaler()
        self.__train_sc = scaler.fit_transform(self.__train)
        self.__test_sc = scaler.transform(self.__test)
        if _info:
            print("MinMax 정규화 ===================================")
            print(f"train_sc:\n{self.__train_sc}")
            print(f"test_sc:\n{self.__test_sc}")

    def categorical_encoder(self, _col, _info=False):
        encoder = preprocessing.LabelEncoder()
        self.__data_frame[_col] = encoder.fit_transform(self.__data_frame[_col])
        self.__data_frame[_col] = self.__data_frame[_col].astype('float32')
        if _info:
            print(f"{_col} encoder ===================================")
            print(self.__data_frame[_col].value_counts())

    def get_dataframe(self):
        return self.__data_frame

    def set_analysis_columns(self, _columns):
        self.analysis_columns = _columns


def origin():
    # 데이터프레임 전처리용 클래스
    dp = DataPreprocess()

    # 전처리된 데이터프레임
    df = dp.df_prcd
    print(df.info())
    print(df)
    print("origin() finish")


class LSTM_remake:
    def __init__(self, _df):
        self.dataframe_origin = _df
        self.column_list = ['MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP', 'OK']  # default column
        self.model_path = './model/'
        self.model = None

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
        :return:
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

    def SMOTE_make(self, _df):
        X_train_values = _df[:,:-1]
        X_train_values = _df[:,:-1]


    def make_dataset(self, _data, _label, _window_size):
        """
        과거값을 통해 현재 종속변수(label)을 예측 하는 것
        :param _data: X_train
        :param _label: y_train
        :param _window_size: window size --> Sequence data의 크기
        :param _columns_list: column name list
        :return: ndarray(feature list), ndarray(label list)
        """
        # DataFrame 형식을 사용하기 위해서 변경이 필요
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
        arly_stop = EarlyStopping(monitor='val_loss', patience=5)
        filename = os.path.join(self.model_path, 'tmp_checkpoint.h5')
        checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    def model_fit(self, _x_train, _y_train, ):
        history = self.model.fit(x_train, y_train,
                            epochs=100,
                            batch_size=50,
                            validation_data=(x_valid, y_valid),
                            callbacks=[early_stop, checkpoint])
