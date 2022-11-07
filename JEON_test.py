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


def read_csv():
    _df = pd.read_csv(f"./dataset/dataset.csv", encoding='cp949')
    return _df


def categorical_encoder(_df):
    col_names = _df.columns.values.tolist()
    df_copy = _df.copy()
    for col_name in col_names:
        if df_copy[col_name].dtypes == 'int64':
            encoder = preprocessing.LabelEncoder()
            df_copy[col_name] = encoder.fit_transform(df_copy[col_name])
            df_copy[col_name] = df_copy[col_name].astype('float64')

    return df_copy


def make_dataset(_data, _label, _window_size, _columns_list):
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
        _data = pd.DataFrame(_data, columns=[_columns_list[:-1]])
    if type(_label) == np.ndarray:
        _label = pd.DataFrame(_label, columns=[_columns_list[-1]])

    #
    feature_list = []
    label_list = []

    for i in range(len(_data) - _window_size):
        feature_list.append(np.array(_data.iloc[i:i + _window_size]))
        label_list.append(np.array(_label.iloc[i + _window_size]))
    return np.array(feature_list), np.array(label_list)


def test_f():
    # 데이터프레임 전처리용 클래스
    dp = DataPreprocess()

    # 전처리된 데이터프레임
    df = dp.df_prcd
    print('df_prcd columns')
    df.info()
    # set_columns = ['MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP', 'CHG_MELT_WEIGHT', 'OK']
    set_columns = ['MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP', 'OK']
    dataset = df[set_columns]
    dataset = dataset.fillna(method='bfill')
    dataset.info()

    split_num = int(dataset.shape[0] * 0.7)
    train = dataset[:split_num]
    test = dataset[split_num:]

    # train = categorical_encoder(train)    # 타입 문제인지 확인하려고 만들었던 함수(타입 문제가 아니라 값의 음수 문제)

    scaler = preprocessing.MinMaxScaler()
    train_sc = scaler.fit_transform(train)  # 'CHG_MELT_WEIGHT' 0 또는 nan 존재
    test_sc = scaler.transform(test)

    X_train_values = train_sc[:, :-1]
    y_train_values = train_sc[:, -1]

    smote = SMOTE(random_state=0)
    X_train_over, y_train_over = smote.fit_resample(X_train_values, y_train_values)
    # print(f"SMOTE 적용 전 {X_train_values.shape}, {y_train_values.shape}")
    # print(f"SMOTE 적용 후 {X_train_over.shape}, {y_train_over.shape}")
    # print(f"적용 후 레이블 값 분포\n{pd.Series(y_train_over).value_counts()}")

    # 데이터 프레임
    train_feature, train_label = make_dataset(X_train_over, y_train_over, 10, set_columns)
    test_sc = pd.DataFrame(test_sc, columns=[set_columns])

    test_feature = test_sc[set_columns[:-1]]
    test_label = test_sc[set_columns[-1]]
    test_feature, test_label = make_dataset(test_feature, test_label, 10, set_columns)
    print(test_feature.shape)
    print(test_label.shape)

    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.3)

    model = Sequential()
    model.add(LSTM(50,  # LSTM 레이어의 결과 50개 나오도록 지정
                   input_shape=(train_feature.shape[1], train_feature.shape[2]),
                   activation='tanh',
                   return_sequences=False)
              )
    model.add(Dense(1, activation='sigmoid'))

    model_path = './model/'
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    filename = os.path.join(model_path, 'tmp_checkpoint.h5')
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    history = model.fit(x_train, y_train,
                        epochs=100,
                        batch_size=500,
                        validation_data=(x_valid, y_valid),
                        callbacks=[early_stop, checkpoint])

    model.load_weights(filename)
    pred = model.predict(test_feature)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.show()

    pred_df = pd.DataFrame(pred, columns=[set_columns[-1]])
    pred_df[set_columns[-1]] = pred_df[set_columns[-1]].apply(lambda x: 1 if x >= 0.5 else 0)

    print(pred_df[set_columns[-1]].value_counts())

    classify = confusion_matrix(test_label, pred_df)
    print(classify)

    p = precision_score(test_label, pred_df)
    r = recall_score(test_label, pred_df)
    f1 = f1_score(test_label, pred_df)
    acc = accuracy_score(test_label, pred_df)
    print(f"precision: %0.4f" % p)
    print(f"recall: %0.4f" % r)
    print(f"f1-score: %0.4f" % f1)
    print(f"accuracy: %0.4f" % acc)


def exist_missing(_df):
    return _df.isnull().sum() != 0


def exist_duplicates(_df):
    return _df.duplicated().sum() != 0


def pandas_test():
    df = pd.read_csv('./dataset/dataset.csv')
    print(exist_missing(df))


def find_cycle(_df):
    print(_df.info())

    MW = _df[['DATE_TIME', 'MELT_WEIGHT']]
    # print(MELT_WEIGHT)

    MELT_WEIGHT_under_200 = MW['MELT_WEIGHT'] < 200

    cycle_info = []
    MW['CYCLE'] = False
    t1 = None
    t2 = None
    for idx in range(2, len(MW)):
        if not MELT_WEIGHT_under_200[idx-1] and MELT_WEIGHT_under_200[idx] and not MELT_WEIGHT_under_200[idx+1]:
            MELT_WEIGHT_under_200[idx] = False
        if not MELT_WEIGHT_under_200[idx-2] and not MELT_WEIGHT_under_200[idx-1] and \
                MELT_WEIGHT_under_200[idx] and MELT_WEIGHT_under_200[idx+1]:  # 구간의 시작
            t1 = idx
        elif MELT_WEIGHT_under_200[idx-2] and MELT_WEIGHT_under_200[idx-1] and \
                not MELT_WEIGHT_under_200[idx] and not MELT_WEIGHT_under_200[idx+1]:
            t2 = idx - 1
            # print(f'{t1}~{t2}')
            # 구간 찾음
            min_val_idx = MW.loc[t1:t2, 'MELT_WEIGHT'].idxmin()
            MW.loc[min_val_idx, 'CYCLE'] = True

    # print(MW['CYCLE'].value_counts())
    plt.plot(MW['DATE_TIME'], MW['MELT_WEIGHT'])
    # plt.plot(MW['DATE_TIME'], cy['MELT_WEIGHT'], 'ro')
    plt.scatter(MW[MW['CYCLE'] == True]['DATE_TIME'].tolist(),
                MW[MW['CYCLE'] == True]['MELT_WEIGHT'].tolist(),
                marker='o', color='red')
    plt.show()

def find_cycle_feat_bychoi(_df):
    mw = _df[['DATE_TIME', 'MELT_WEIGHT']]
    MELT_WEIGHT_under_200 = mw['MELT_WEIGHT'] < 200

    cycle_info = []
    mw['CYCLE'] = False
    t1 = None
    t2 = None
    for idx in range(1, len(mw)):
        if not MELT_WEIGHT_under_200[idx - 1] and MELT_WEIGHT_under_200[idx]:  # 구간의 시작
            t1 = idx
        elif MELT_WEIGHT_under_200[idx - 1] and not MELT_WEIGHT_under_200[idx]:
            t2 = idx - 1
            # print(f'{t1}~{t2}')
            # 구간 찾음
            min_val_idx = mw.loc[t1:t2, 'MELT_WEIGHT'].idxmin()
            mw.loc[min_val_idx, 'CYCLE'] = True

if __name__ == '__main__':
    # 데이터프레임 전처리용 클래스
    dp = DataPreprocess()
    # 전처리된 데이터프레임
    df = dp.df_prcd
    df = df.fillna(method='bfill')

    # test_f()
    find_cycle(df)
    # pandas_test()
