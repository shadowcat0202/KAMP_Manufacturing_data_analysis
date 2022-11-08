import pandas as pd
import sklearn

import matplotlib.pyplot as plt
import seaborn as sns
import tsfresh.examples

from DataPreprocess import DataPreprocess

if __name__ == '__main__':

    # 데이터프레임 전처리용 클래스
    dp = DataPreprocess()
    #
    # # 전처리된 데이터프레임
    df = dp.df_prcd

    print(df)

    df.dropna(axis=0, inplace=True)
