import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def data_Loader(_filename, _info=False):
    df_buf = pd.read_csv(_filename)
    if _info:
        print(df_buf.head(5))
        print(df_buf.tail(5))
        print(df_buf.info())
        print(df_buf.describe())
    return df_buf


def exist_missing(_df, _info=False):
    result = _df.isnull().sum() != 0
    if _info:
        print(result)
    return result


def exist_duplicates(_df, _info, _column=None):
    result = None
    if _column is not None:
        result = _df.duplicated().sum() != 0
    else:
        result = _df[_column].duplicated().sum() != 0
    if _info:
        print(result)
    return result


def num_to_sec(_df, _info=False, _from=None, _to=None):
    _df[_to] = _df[_from] * 6 % 60
    _df.drop(columns=[_from], inplace=True)
    if _info:
        print(f'[{_to}]')
        print(_df[_to])
    return _df

def make_24_time_combine_datetime_and_sec(_df, _info=False, _datetime_col=None, _sec_col=None):
    pass

def pandas_test():
    df_origin = data_Loader('./dataset/dataset.csv', _info=False)
    list_columns = df_origin.columns.tolist()
    list_var_columns = ["MELT_TEMP", "MOTORSPEED", "MELT_WEIGHT", "INSP"]

    df_copy = df_origin.copy()
    df_copy = num_to_sec(df_copy, _info=True, _from='NUM', _to='SEC_24')




if __name__ == '__main__':
    pandas_test()
