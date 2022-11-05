"""
시간이 너무 오래걸리는 관계로 이 코드 사용 X
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

from DataPreprocess import DataPreprocess

# pandas option
pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)

# 데이터프레임 전처리용 클래스
dp = DataPreprocess()

# 전처리된 데이터프레임
df = dp.df_prcd

def trueMinima(df):
    for i in range(len(df)):
        if df.loc[i, "CYCLE_MINIMA"] == True:
            df.loc[i, "MINIMA_VALUE"] = df.loc[i, "MELT_WEIGHT"]


if __name__ == "__main__":
    cycle_df = df.drop(['OUTLIER_WGT(TB)', 'OUTLIER_WTG(MM_60)', 'OUTLIER_WTG(MM_30)',
                        'OUTLIER_WTG(MM_15)', 'OUTLIER_WTG(MM_9)', 'OUTLIER_WTG(MM_3)',
                        'MONTH', 'WEEK', 'DATE', 'WEEKDAY', 'HOUR'], axis=1)


    # 1. MELT_WEIGHT 값이 0 되는 지점을 찾아봅시다.
    zero_point = cycle_df[cycle_df["MELT_WEIGHT"] == 0]
    zero_Idx = list(zero_point["MELT_WEIGHT"].index) # 무게가 0인 행의 인덱스

    cycle_df["CYCLE_MINIMA"] = False # 컬럼 새로 생성
    cycle_df.loc[zero_Idx, "CYCLE_MINIMA"] = True # 무게가 0이라면 True로 바꿔줌

    """
    시각화를 통해 각 주기별 최솟값이 y=30 이하임을 확인함
    """
    # plt.plot(cycle_df["MELT_WEIGHT"])
    # plt.scatter(zero_point["MELT_WEIGHT"].index,zero_point["MELT_WEIGHT"], marker="*", c="red")
    # plt.axhline(y=40, color='green', linewidth=1)
    # plt.axhline(y=30, color='orange', linewidth=1)
    # plt.axhline(y=20, color='purple', linewidth=1)
    # plt.show()

    # 2. 구간의 Minumum 값 찾기
    bins = 300 # 구간
    cnt = 0
    for i in range(bins, len(cycle_df)-bins):
        temp_df = cycle_df.loc[i-bins:i+bins, "MELT_WEIGHT"] # 구간으로 자름
        minVal = temp_df.min() # 그 구간에서 최소값
        temp_min = temp_df[temp_df == temp_df.min()] # 최솟값을 가지는 행 or 데이터프레임

        # 0이 최솟값인 구간은 이미 CYCLE_MINIMA 컬럼이 True, ==> 지나갑시다.
        if minVal == 0:
            continue

        # minVal 이 30 이상이면 최솟값 아님, 다른 구간으로 얼른 넘어갑시다.
        if minVal >= 30:
            continue

        temp_minIdx = temp_min.index[0] # 최소값의 인덱스
        # cycle_df.loc[temp_minIdx, "CYCLE_MINIMA"] = True

        #이게 진짜 최소값인지 근처 값에서 최소면 진짜 최소
        """
        최솟값 후보의 인덱스가 temp_minIdx 일 때
        [temp_minIdx-bins:temp_minIdx+bins] 에서의 최솟값과 일치한다면
        이는 최솟값이 맞다고 판단
        """
        #
        tenAroundMin = cycle_df.loc[temp_minIdx-bins:temp_minIdx+bins, "MELT_WEIGHT"].min()
        #
        if minVal == tenAroundMin:
            cycle_df.loc[temp_minIdx, "CYCLE_MINIMA"] = True

    cycle_df["MINIMA_VALUE"] = np.nan

    print(cycle_df)
    trueMinima(cycle_df)

    plt.plot(cycle_df["MELT_WEIGHT"])
    plt.scatter(cycle_df["MINIMA_VALUE"].index,cycle_df["MINIMA_VALUE"], marker="*", c="red")
    plt.axhline(y=40, color='green', linewidth=1)
    plt.axhline(y=30, color='orange', linewidth=1)
    plt.axhline(y=20, color='purple', linewidth=1)
    plt.show()