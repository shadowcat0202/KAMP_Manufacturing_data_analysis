import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DataPreprocess import DataPreprocess

base_df_src = "../data/dataset.csv"
base_df = pd.read_csv(base_df_src)

pd.set_option("display.max_columns", None)

base_df = base_df.set_index("NUM") # NUM 은 index 컬럼으로 설정
# base_df.drop("NUM", inplace=True, axis=1) # NUM 컬럼 날려버림


if __name__ == "__main__":
    print(base_df.head())
    print(base_df.info())
    # print(base_df.describe()) # 835200 rows, Null 값 없음

    """
               MELT_TEMP     MOTORSPEED    MELT_WEIGHT           INSP
    count  835200.000000  835200.000000  835200.000000  835200.000000
    mean      509.200623     459.782865     582.962125       3.194853
    std       128.277519     639.436413    1217.604433       0.011822
    min       308.000000       0.000000       0.000000       3.170000
    25%       430.000000     119.000000     186.000000       3.190000
    50%       469.000000     168.000000     383.000000       3.190000
    75%       502.000000     218.000000     583.000000       3.200000
    max       832.000000    1804.000000   55252.000000       3.230000

    1. MELT_TEMP : 통계값으로는 큰 이상 없어 보임

    2. MOTORSPEED : STD 값이 639, 평균이 459.8, 중앙값이 168
                    ==> 이상치 다수 분포
                    ==> 1000을 넘는 것 기준으로 이상치여부 판단
                    ==> 앞 뒤 간격 잘 볼 것

    3. MELT_WEIGHT : STD 값이 1217, 평균이 583.0, 중앙값이 383
                    ==> 역시 이상치 다수분포
                    ==> denomination 필요

    4. INSP : 3.17 ~ 3.23의 값
              ==> 표준편차도 0.012 에 불과
              ==> 과연 필요한 feature인가?
    """
    df_1 = base_df[["MELT_TEMP", "MOTORSPEED", "MELT_WEIGHT"]]


    ## 1. MELT_TEMP
    # print("==MELT_TEMP========================================================")
    # temp = base_df["MELT_TEMP"]
    # print(temp)
    # print("========================================================MELT_TEMP==")




    ## 2. MOTORSPEED
    # print("==MOTORSPEED========================================================")
    # speed = base_df["MOTORSPEED"]
    # print(speed.iloc[:10])
    # print("========================================================MOTORSPEED==")





    print("==MELT_WEIGHT========================================================")

    weight_sr = df_1["MELT_WEIGHT"]
    # print(weight_sr.iloc[1:])
    # print(len(weight_sr))
    print(weight_sr)


    # plt.plot(weight_sr.iloc[:2000])
    plt.plot(weight_sr.iloc[:10000])

    try:
        for i in range(1, 10000):
            devide_value = weight_sr[i] / weight_sr[i-1]

            if devide_value >= 8 and weight_sr[i] >= 1000:
                weight_sr[i] = weight_sr[i] // 10

    except ZeroDivisionError as e:
        print(e)
        pass

    # try:
    #     for i in range(1, 10000):
    #         devide_value = weight_sr[i] / weight_sr[i-1]
    #
    #         if devide_value >= 9:
    #             weight_sr[i] = weight_sr[i] // 10
    # except ZeroDivisionError as e:
    #     print(e)
    #     pass

    plt.plot(weight_sr.iloc[:10000])
    # plt.plot(weight_sr.iloc[:2000])
    plt.show()

    print("========================================================MELT_WEIGHT==")
