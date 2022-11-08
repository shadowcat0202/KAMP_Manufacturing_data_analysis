import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import seaborn as sns


# pd.set_option("display.max_rows", None)

def find_cycle(_df):
    print(_df.info())

    MW = _df[['DATE_TIME', 'MELT_WEIGHT']]
    # print(MELT_WEIGHT)

    # print(MW.loc[573700:573850, "MELT_WEIGHT"])
    # print(MW.loc[610050:610250, "MELT_WEIGHT"])

    MELT_WEIGHT_under_200 = MW['MELT_WEIGHT'] < 200

    cycle_info = []
    MW['CYCLE'] = False
    t1 = None
    t2 = None
    for idx in range(1, len(MW)-300):
        if not MELT_WEIGHT_under_200[idx - 1] and MELT_WEIGHT_under_200[idx]:  # 구간의 시작
            t1 = idx
        elif MELT_WEIGHT_under_200[idx - 1] and not MELT_WEIGHT_under_200[idx]:
            t2 = idx - 1
            # print(f'{t1}~{t2}')
            # 구간 찾음
            min_val_idx = MW.loc[t1:t2, 'MELT_WEIGHT'].idxmin()
            if MW.loc[min_val_idx, "MELT_WEIGHT"] <= 30: # 최솟값은 30이하에 존재
                # 최솟값 앞 뒤로 300씩 구간에서의 최솟값 확인, 일치한다면 최솟값 맞음!
                section_300_min = MW.loc[min_val_idx - 300:min_val_idx + 300, "MELT_WEIGHT"].min()
                if MW.loc[min_val_idx, "MELT_WEIGHT"] == section_300_min:
                    MW.loc[min_val_idx, 'CYCLE'] = True

                # 해결해야 하는 부분 : 한 주기 안에서 최솟값이 중복되는 경우?

    # print(MW['CYCLE'].value_counts())
    # plt.plot(MW['DATE_TIME'], MW['MELT_WEIGHT'])
    # plt.plot(MW['DATE_TIME'].index, MW['MELT_WEIGHT'])
    # # plt.plot(MW['DATE_TIME'], cy['MELT_WEIGHT'], 'ro')
    # plt.scatter(MW[MW['CYCLE'] == True]['DATE_TIME'].index,
    #             # MW[MW['CYCLE'] == True]['DATE_TIME'].tolist(),
    #             MW[MW['CYCLE'] == True]['MELT_WEIGHT'].tolist(),
    #             marker='o', color='red')
    # plt.axhline(y=200, color='green', linewidth=1)
    # plt.axhline(y=100, color='orange', linewidth=1)
    # plt.axhline(y=30, color='purple', linewidth=1)
    # plt.show()

    _df["CYCLE"] = MW["CYCLE"]

    return _df


def coundCycle(_df):
    """
    :param _df:
    :return: 주기가 들어간 list
    """
    cycleIdxs = []

    for i in range(len(_df)):
        if _df.loc[i, "CYCLE"] == True:
            cycleIdxs.append(i)

    cycleList = []
    for i in range(1, len(cycleIdxs)):
        cycle = cycleIdxs[i] - cycleIdxs[i-1]
        cycleList.append(cycle)

    return cycleList



# if __name__ == '__main__':
#     from DataPreprocess import DataPreprocess
#     # 데이터프레임 전처리용 클래스
#     dp = DataPreprocess()
#     # 전처리된 데이터프레임
#     df = dp.df_prcd
#
#     # test_f()
#     df = find_cycle(df)
#     # pandas_test()
#
#     cycleList = coundCycle(df) #
#
#     # print(cycleList)
#     plt.style.use("ggplot")
#     plt.title("cycle histogram")
#     plt.hist(cycleList, bins=200)
#     plt.show()
