import pandas as pd
import sklearn

import matplotlib.pyplot as plt
import seaborn as sns
import tsfresh.examples

from DataPreprocess import DataPreprocess

#TODO - DELETE
from DataPreprocess_2DataCleaning import DataCleaning
from FindCycle_app_JEON import find_cycle

if __name__ == '__main__':

    # 데이터프레임 전처리용 클래스
    dp = DataPreprocess()
    # 전처리된 데이터프레임
    df = dp.df_prcd

    # NaN 제거:  각 행에 윈도우 적용, 이전 관측치 칼럼에 연결함으로써 발생하는 nan을 제거
    df.dropna(axis=0, inplace=True)
